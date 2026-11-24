"""Alternate two independent verifier versions on one externally pinned archive."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.performance import describe
from sciagent.evidence import check_archive, sha256_file
from sciagent.spec import RetryPolicy, VERIFIER_MEMORY_MB, VERIFIER_TIMEOUT_SECONDS


def same_decision(observed, saved):
    def gates(checks):
        return [(c["name"], c["passed"], c["code"], c["threshold"]) for c in checks]

    if (observed["verdict"] != saved["verdict"] or observed["failure_codes"] != saved["failure_codes"]
            or observed["input_sha256"] != saved["input_sha256"]
            or gates(observed["study_checks"]) != gates(saved["study_checks"])
            or [x["n"] for x in observed["levels"]] != [x["n"] for x in saved["levels"]]):
        return False
    if gates([observed["accuracy_check"]] if observed["accuracy_check"] else []) != gates(
            [saved["accuracy_check"]] if saved["accuracy_check"] else []):
        return False
    for a, b in zip(observed["levels"], saved["levels"]):
        if (gates(a["checks"]) != gates(b["checks"]) or a["num_dofs"] != b["num_dofs"]
                or a["solver_iterations"] != b["solver_iterations"]):
            return False
        for key in ("l2_error", "h1_semi_error", "reference_rel_diff"):
            if not math.isclose(a[key], b[key], rel_tol=1e-7, abs_tol=1e-12):
                return False
    return True


def replay(root, attempt, policy, saved):
    memory = policy.verifier_memory_mb or VERIFIER_MEMORY_MB
    timeout = policy.verifier_timeout_seconds or VERIFIER_TIMEOUT_SECONDS
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env.pop("PYTHONPATH", None)
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "sciagent.verification_worker", str(attempt), str(memory)],
                          input="{}", cwd=root, env=env, text=True, capture_output=True, timeout=timeout)
    elapsed = time.perf_counter() - start
    proc.check_returncode()
    data = json.loads(proc.stdout)
    out = data["verdict"]
    return {"wall_seconds": elapsed, "peak_rss_kib": data["peak_rss_kib"], "version": out["verifier_version"],
            "verdict": out["verdict"], "equivalent": same_decision(out, saved),
            "levels": [{k: lv[k] for k in ("n", "l2_error", "h1_semi_error", "reference_rel_diff")}
                       for lv in out["levels"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two measured pairs are required")
    attempt = args.attempt.resolve()
    mode, saved, changed = check_archive(attempt, args.manifest_sha256)
    if changed or mode != "sha256-inventory" or saved["verdict"] != "ACCEPTED":
        raise ValueError("benchmark requires an intact, accepted archive with a complete inventory")
    policy = RetryPolicy.model_validate_json((attempt.parent.parent / "frozen/policy.json").read_text())
    roots = {"before": args.baseline_root.resolve(), "after": Path(__file__).resolve().parents[1]}
    hashes = {k: {p: sha256_file(root / p) for p in ("verifier/verifier.py", "verifier/fem.py")}
              for k, root in roots.items()}
    rows = []
    for i in range(-1, args.repeats):
        order = list(roots) if i % 2 == 0 else list(reversed(roots))
        row = {k: replay(roots[k], attempt, policy, saved) for k in order}
        if i < 0:
            warmup = row
        else:
            rows.append(row)
    if check_archive(attempt, args.manifest_sha256)[2]:
        raise ValueError("archive changed during measurement")
    passed = all(r[k]["equivalent"] for r in [warmup, *rows] for k in roots)
    report = {"passed": passed, "archive": attempt.parent.parent.name, "attempt": attempt.name,
              "manifest_sha256": args.manifest_sha256, "verifier_sources_sha256": hashes,
              "affinity": sorted(os.sched_getaffinity(0)),
              "budget_mb": policy.verifier_memory_mb or VERIFIER_MEMORY_MB,
              "timeout_seconds": policy.verifier_timeout_seconds or VERIFIER_TIMEOUT_SECONDS,
              "warmup": warmup, "samples": rows,
              "statistics": {k: {m: describe([r[k][m] for r in rows]) for m in ("wall_seconds", "peak_rss_kib")}
                             for k in roots},
              "paired_speedup": describe([r["before"]["wall_seconds"] / r["after"]["wall_seconds"] for r in rows])
              if passed else None}
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
