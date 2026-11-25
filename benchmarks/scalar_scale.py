"""Capture or replay the large 2D scalar studies used in the E4 revision."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.e3_complex_meshes import _problem_var
from sciagent import controller
from sciagent.evidence import sha256_file
from sciagent.gmsh import generate_unstructured_msh

ROOT = Path(__file__).resolve().parents[1]


def input_file(name, directory):
    spec = json.loads((ROOT / "specs/poisson_sin_sin_p1_tri.json").read_bytes())
    if name == "irregular-q3":
        mesh = generate_unstructured_msh("quad", nx=16, seed=77, jitter=0.22)
        (directory / "graded.msh").write_bytes(mesh)
        spec.update(schema_version="0.4", problem=_problem_var().model_dump(mode="json"),
                    external_mesh={"file": "graded.msh", "sha256": hashlib.sha256(mesh).hexdigest()})
        spec["execution"].update(mesh_type="quad", mesh_topology="unstructured",
                                 mesh_levels=[16, 32, 64], polynomial_order=3)
        spec["validation"] = {"reference_solve_method": "amg_pcg", "reference_solve_max_dofs": 100000,
                              "max_l2_error_finest": 1e-6}
    else:
        finest = int(name.removeprefix("p1-"))
        spec["execution"].update(mesh_levels=[finest//4, finest//2, finest])
        spec["validation"] = {"reference_solve_method": "amg_pcg", "reference_solve_max_dofs": 1700000,
                              "max_l2_error_finest": 2/finest**2}
        spec["policy"] = {"max_attempts": 1, "max_dofs_per_level": 1700000,
                          "per_level_timeout_seconds": 600, "verifier_timeout_seconds": 600, "verifier_memory_mb": 3072}
    path = directory / "study.json"
    path.write_text(json.dumps(spec, indent=2) + "\n")
    return path


def summarize(archive, pin):
    manifest = json.loads((archive / "manifest.json").read_bytes())
    receipt = controller.reverify(archive / "attempts/attempt_01", expected_digest=pin)
    if not receipt["integrity"]["passed"] or receipt["verdict"] != manifest["final_state"] or receipt["verdict"] != "ACCEPTED":
        raise ValueError("independent numerical replay failed")
    return {"manifest_sha256": pin, "state": manifest["final_state"], "replay": receipt["verdict"],
            "native_and_controller_seconds": manifest["total_wall_seconds"],
            "verification": manifest["attempts"][0]["verification"],
            "levels": [{k: lv[k] for k in ("n", "num_dofs", "l2_error", "h1_semi_error", "reference_rel_diff")}
                       for lv in receipt["levels"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=["p1-1024", "p1-1280", "irregular-q3"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--manifest-sha256", help="replay the run directory against this externally retained digest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.directory.resolve()
    if args.manifest_sha256:
        pin = args.manifest_sha256
    else:
        archive.mkdir(parents=True, exist_ok=False)
        run = controller.run_study(input_file(args.name, archive), archive / "studies")
        archive = Path(run["run_dir"])
        pin = sha256_file(archive / "manifest.json")
    result = summarize(archive, pin)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
