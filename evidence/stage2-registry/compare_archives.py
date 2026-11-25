"""Replay the pinned Stage 1 and fresh run_demo archives on the same toolchain."""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sciagent.controller import reverify
from sciagent.evidence import inspect_manifest, sha256_file


def require(condition, location):
    if not condition:
        raise ValueError(f"Archive parity failed at {location}")


def compare(previous, current):
    baseline = json.loads((ROOT / "evidence/stage1/canonical-comparison.json").read_text())["studies"]
    counts = Counter()
    studies = {}
    canonical = []
    for manifest_path in sorted(current.glob("*/manifest.json")):
        new_dir = manifest_path.parent
        new_manifest, _, changed = inspect_manifest(new_dir)
        name = Path(new_manifest["spec_file"]).stem
        require(not changed and name in baseline and name not in studies, name)
        old_digest = baseline[name]["manifest_sha256"]
        old_dir = previous / baseline[name]["directory"]
        old_manifest, _, changed = inspect_manifest(old_dir, old_digest)
        require(not changed, old_dir)
        for key in ("final_state", "final_code", "frozen_sha256", "bound_case"):
            require(old_manifest.get(key) == new_manifest.get(key), (name, key))
        new_digest = sha256_file(manifest_path)
        studies[name] = {"state": new_manifest["final_state"], "code": new_manifest["final_code"],
                         "stage1_manifest": old_digest, "new_manifest": new_digest}
        for a, b in zip(old_manifest["attempts"], new_manifest["attempts"], strict=True):
            for key in ("attempt", "mesh_levels", "verdict", "failure_codes", "errors", "orders"):
                require(a[key] == b[key], (name, key))
            folder = Path("attempts") / f"attempt_{a['attempt']:02d}"
            for run, digest in ((old_dir, old_digest), (new_dir, new_digest)):
                verdict = reverify(run / folder, expected_digest=digest)
                require(verdict["integrity"]["passed"] and verdict["verdict"] == a["verdict"]
                        and verdict["failure_codes"] == a["failure_codes"], (name, run, folder))
            counts["paired_attempt_replays"] += 1
            for level in a["errors"]:
                for norm in ("l2", "h1_semi"):
                    if level[norm] is not None:
                        counts["matched_error_values"] += 1
                n = level["n"]
                base = folder / "levels" / f"n{n:04d}"
                old_level, new_level = old_dir / base, new_dir / base
                old_result, new_result = old_level / "result.json", new_level / "result.json"
                require(old_result.exists() == new_result.exists(), (name, base))
                if not old_result.exists():
                    continue
                left, right = json.loads(old_result.read_text()), json.loads(new_result.read_text())
                require("diagnostics" not in left, old_result)
                diagnostics = right.pop("diagnostics")
                require(diagnostics["schema_version"] == 1, new_result)
                for obj in (left, right):
                    for k in ("time_setup_ms", "time_solve_ms", "time_error_ms"):
                        del obj[k]
                require(left == right, (name, base, "native record"))
                counts["matched_native_results"] += 1
                for file in ("dof_values.txt", "element_dofs.txt", "elements.txt", "boundary.txt", "mesh.mesh", "solution.gf"):
                    require((old_level / file).is_file() and sha256_file(old_level / file) == sha256_file(new_level / file), (name, base, file))
                counts["byte_identical_field_exports"] += 1
                if name.startswith("poisson_") or (n == a["mesh_levels"][0] and name in
                                                   {"fault_early_stop_lie", "fault_wrong_source", "fault_nan_value"}):
                    canonical.append({"study": name, "n": n, **diagnostics})
    require(studies.keys() == baseline.keys(), "study set")
    return {"baseline": "52fed54", "study_count": len(studies), "counts": dict(counts),
            "studies": studies, "canonical_levels_and_fault_examples": canonical}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path, help="the 31 retained Stage 1 WSL2 runs")
    parser.add_argument("current", type=Path, help="fresh run_demo directory")
    args = parser.parse_args()
    print(json.dumps(compare(args.previous, args.current), indent=2, allow_nan=False))
