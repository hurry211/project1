"""Run and independently certify every reviewed scalar case/configuration pair."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import BilinearForm, LinearForm, asm
from skfem.helpers import dot, grad

from benchmarks.reference import Reference, relative_difference, tensor_basis
from sciagent import capability, controller
from sciagent.evidence import sha256_file

SPECS = Path(__file__).resolve().parents[1] / "specs"
CASES = {
    "sin_sin": SPECS / "poisson_sin_sin_p1_tri.json",
    "cos_exp": SPECS / "poisson_cos_exp_mixed_p1_quad.json",
    "sin_sin_variable": SPECS / "stage2" / "diffusion_mixed_p1_tri.json",
}


def check(ok, detail):
    if not ok:
        raise ValueError(detail)


def variable_reference(basis, field):
    """Project the exact gradient in the energy inner product.

    The top natural trace is included by the weak identity. This independent
    load uses neither the production source expression nor a boundary integral.
    """
    def analytic(x, y):
        sx, sy = np.sin(np.pi*x), np.sin(2*np.pi*y)
        return sx*sy, np.array([np.pi*np.cos(np.pi*x)*sy, 2*np.pi*sx*np.cos(2*np.pi*y)])

    @BilinearForm
    def stiffness(u, v, w):
        return (1 + w.x[0]) * dot(grad(u), grad(v))

    @LinearForm
    def load(v, w):
        _, g = analytic(*w.x)
        return (1 + w.x[0]) * dot(g, grad(v))

    x, y = basis.doflocs
    free = (x != 0) & (x != 1) & (y != 0)
    K, F = asm(stiffness, basis), asm(load, basis)
    solution = np.zeros(basis.N)
    solution[free] = spsolve(K[free][:, free], F[free])
    u, g = analytic(*basis.global_coordinates())
    uh = basis.interpolate(field)
    norms = (np.sqrt(np.sum((uh-u)**2 * basis.dx)),
             np.sqrt(np.sum(np.sum((uh.grad-g)**2, axis=0) * basis.dx)))
    return solution, np.array(norms)


def case_request(case, source, key, directory):
    mt, p, assembly, solver = key
    req = json.loads(CASES[case].read_text())
    req["execution"].update(mesh_type=mt, polynomial_order=p, assembly=assembly, solver=solver,
                            mesh_levels=[8, 16, 32])
    if source == "gmsh22":
        mesh = SPECS / "stage2" / f"unit_square_{mt}.msh"
        data = mesh.read_bytes()
        (directory / mesh.name).write_bytes(data)
        req.update(schema_version="0.3", external_mesh={"file": mesh.name, "sha256": hashlib.sha256(data).hexdigest()})
        for boundary in req["problem"]["boundaries"]:
            boundary["attribute"] *= 11
    path = directory / "study.json"
    path.write_text(json.dumps(req, indent=2) + "\n")
    return path


def certify(run, case, source, key, pinned):
    digest = sha256_file(run / "manifest.json")
    check(pinned is None or digest == pinned, f"{run}: retained manifest digest differs")
    manifest = json.loads((run / "manifest.json").read_text())
    check((manifest["final_state"], manifest.get("bound_case"), len(manifest["attempts"])) ==
          ("ACCEPTED", case, 1), f"{run}: {manifest['final_state']} / {manifest['final_reason']}")
    attempt = run / "attempts/attempt_01"
    replay = controller.reverify(attempt, expected_digest=digest)
    check(replay["integrity"]["passed"] and replay["verdict"] == "ACCEPTED", f"{run}: replay failed")
    req = json.loads((run / "request/study_spec.json").read_text())
    execution = req["execution"]
    mt, p, assembly, solver = key
    check((execution["mesh_type"], execution["polynomial_order"], execution["assembly"], execution["solver"]) == key
          and ("external_mesh" in req) == (source == "gmsh22"), f"{run}: request differs from matrix row")
    sides = (11, 22, 33, 44) if source == "gmsh22" else (1, 2, 3, 4)
    field_gaps, norm_gaps, norms = [], [], []
    check([lv["n"] for lv in replay["levels"]] == [8, 16, 32], f"{run}: level sequence differs")
    for lv in replay["levels"]:
        n = lv["n"]
        level = attempt / "levels" / f"n{n:04d}"
        ref = Reference(tensor_basis(execution, n), mixed=case == "cos_exp")
        field, order = ref.native_field(level)
        ref.check_topology(level, order, n, side_tags=sides)
        if case == "sin_sin_variable":
            solution, errors = variable_reference(ref.basis, field)
        else:
            solution, errors = ref.solve(), np.array(ref.errors(field))
        gap = relative_difference(field, solution)
        norm_gap = float(np.max(abs(np.array([lv["l2_error"], lv["h1_semi_error"]]) / errors - 1)))
        check(gap < 1e-7 and norm_gap < 1e-6, f"{run}/{n}: field gap {gap}, norm gap {norm_gap}")
        field_gaps.append(gap)
        norm_gaps.append(norm_gap)
        norms.append(errors)
    slope = np.log2(norms[-2] / norms[-1])
    check(np.all(np.diff(norms, axis=0) < 0) and p+1-0.2 <= slope[0] <= p+1+0.5 and
          p-0.2 <= slope[1] <= p+0.5, f"{run}: oracle convergence {slope}")
    return {"case": case, "mesh_source": source, "mesh_type": mt, "order": p, "assembly": assembly,
            "solver": solver, "manifest_sha256": digest, "max_relative_field_difference": max(field_gaps),
            "max_relative_norm_difference": max(norm_gaps), "finest_l2": norms[-1][0],
            "finest_h1_semi": norms[-1][1], "finest_orders": slope.tolist()}


def measure(directory, previous):
    rows = {}
    check(set(CASES) == set(capability.CASE_IDS), "each reviewed case requires independent oracle coverage")
    for case in CASES:
        for source in capability.MESH_SOURCES:
            for key in sorted(capability.STAGE2_CONFIGURATIONS):
                label = "_".join(map(str, (case, source, *key)))
                parent = directory / label
                if previous is None:
                    parent.mkdir(parents=True, exist_ok=False)
                    path = case_request(case, source, key, parent)
                    run = Path(controller.run_study(path, parent / "runs")["run_dir"])
                    pinned = None
                else:
                    matches = list((parent / "runs").glob("*/manifest.json"))
                    check(len(matches) == 1, f"{parent}: expected one archived run")
                    run, pinned = matches[0].parent, previous["rows"][label]["manifest_sha256"]
                rows[label] = certify(run, case, source, key, pinned)
                print(f"PASS {label}", flush=True)
    return {"schema_version": 1, "mfem_version": capability.MFEM_VERSION, "mfem_commit": capability.MFEM_COMMIT,
            "build_flags": capability.REVIEWED_FLAGS, "mesh_levels": [8, 16, 32], "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="fresh output root or existing capture")
    parser.add_argument("--report", type=Path, required=True, help="JSON output")
    parser.add_argument("--replay", type=Path, help="use retained archives and previously saved report digests")
    args = parser.parse_args()
    previous = json.loads(args.replay.read_text()) if args.replay else None
    result = measure(args.directory, previous)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
