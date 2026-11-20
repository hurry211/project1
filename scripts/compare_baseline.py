"""Compare a fresh run_demo directory with the versioned Claude run summaries."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sciagent.evidence import inspect_manifest, sha256_file  # noqa: E402


def compare(runs_dir: Path) -> dict:
    historical = json.loads((ROOT / "evidence/CLAUDE_RUN_MANIFESTS.json").read_text())["runs"]
    expected = {name.split("_", 1)[1]: m for name, m in historical.items()}
    studies = {}
    differences = []
    relative_deltas = []
    for path in sorted(runs_dir.glob("*/manifest.json")):
        run, _, changed = inspect_manifest(path.parent)
        name = Path(run["spec_file"]).stem
        if changed or run.get("evidence_version") != 1:
            raise ValueError(f"{name}: incomplete or changed evidence inventory: {sorted(changed)}")
        if name in studies or name not in expected:
            raise ValueError(f"{name}: duplicate or unexpected study in {runs_dir}")
        old = expected[name]
        fields = ("final_state", "final_code", "frozen_sha256")
        if any(run.get(k) != old.get(k) for k in fields) or len(run["attempts"]) != len(old["attempts"]):
            differences.append(name)
        for a, b in zip(run["attempts"], old["attempts"]):
            if (a.get("mesh_levels") != b.get("mesh_levels") or a["verdict"] != b["verdict"]
                    or not set(b.get("failure_codes", [])) <= set(a.get("failure_codes", []))):
                differences.append(name)
        if old["final_state"] == run["final_state"] == "ACCEPTED":
            fresh, prior = run["attempts"][-1]["errors"], old["attempts"][-1]["errors"]
            for a, b in zip(fresh, prior, strict=True):
                if a["n"] != b["n"]:
                    differences.append(name)
                for key in ("l2", "h1_semi"):
                    delta = abs(a[key] - b[key]) / abs(b[key])
                    if not math.isfinite(delta):
                        raise ValueError(f"{name}: non-finite error comparison")
                    relative_deltas.append(delta)
        studies[name] = {"state": run["final_state"], "code": run["final_code"],
                         "attempts": len(run["attempts"]), "directory": path.parent.name,
                         "manifest_sha256": sha256_file(path)}
    return {"states": dict(sorted(Counter(s["state"] for s in studies.values()).items())),
            "expected_studies": len(expected), "observed_studies": len(studies),
            "missing": sorted(expected.keys() - studies.keys()), "mismatches": sorted(set(differences)),
            "accepted_final_norm_values": len(relative_deltas),
            "max_relative_norm_difference": max(relative_deltas, default=None), "studies": studies}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path)
    try:
        report = compare(parser.parse_args().runs_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Cannot compare runs: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    sys.exit(1 if report["missing"] or report["mismatches"] else 0)
