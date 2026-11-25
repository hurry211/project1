"""Measure replay wall time and worker peak RSS on byte-identical archived studies.

Run on Linux, optionally under taskset. Timings exclude interpreter startup,
imports, archive inventory checks, and verdict serialization. Peak RSS includes
the entire fresh worker process. This benchmark launches no MFEM computation.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
from statistics import median
import subprocess
import sys
import time

ARMS = ("baseline", "candidate")
THREAD_SETTINGS = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def worker(attempt):
    import resource
    from verifier import verifier

    if Path(verifier.__file__).resolve() != Path.cwd() / "verifier/verifier.py":
        raise RuntimeError("worker imported verifier outside selected source root")
    start = time.monotonic()
    verdict = verifier.verify_attempt(attempt, require_run_records=True)
    duration = time.monotonic() - start
    return {"seconds": duration, "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "verdict": verdict["verdict"], "verdict_sha256": _digest(verdict)}


def _source_snapshot(root):
    from sciagent.evidence import sha256_file
    paths = sorted(p for package in ("sciagent", "verifier") for p in (root / package).rglob("*.py"))
    if root / "verifier/verifier.py" not in paths:
        raise ValueError(f"missing verifier in {root}")
    return _digest({p.relative_to(root).as_posix(): sha256_file(p) for p in paths})


def _archive_snapshot(attempt):
    from sciagent.evidence import artifact_hashes, inspect_manifest, sha256_file
    study = attempt.parent.parent
    manifest = study / "manifest.json"
    if manifest.exists():
        _, files, changed = inspect_manifest(study)
        if changed:
            raise ValueError(f"archive inventory differs on {study}: {sorted(changed)}")
    else:
        files = artifact_hashes(study)
    return {"study_inventory_sha256": _digest(files), "inventory_files": len(files),
            "manifest_sha256": sha256_file(manifest)}


def _orders(blocks, rng):
    # Counterbalance first position exactly for even block counts; randomize
    # which arm receives the extra first position for odd block counts.
    first = [0, 1] * (blocks // 2)
    if blocks % 2:
        first.append(rng.randrange(2))
    rng.shuffle(first)
    return [[ARMS[i], ARMS[1 - i]] for i in first]


def _measure_pair(attempt, roots, order, script):
    pair = {"order": order}
    for arm in order:
        env = {**os.environ, **THREAD_SETTINGS, "PYTHONPATH": str(roots[arm])}
        out = subprocess.check_output([sys.executable, str(script), "--worker", str(attempt)],
                                      cwd=roots[arm], env=env, text=True)
        pair[arm] = json.loads(out)
    if pair["baseline"]["verdict_sha256"] != pair["candidate"]["verdict_sha256"]:
        raise ValueError(f"complete verdict differs on {attempt}")
    return pair


def _paired_effect(rows, metric):
    from benchmarks.e1_analysis import _interval
    differences = [row["candidate"][metric] - row["baseline"][metric] for row in rows]
    return {"median_candidate_minus_baseline": median(differences), "paired_bootstrap_95pct": _interval(differences)}


def compare(baseline, candidate, attempts, blocks, seed, *, expect_verdict=None):
    if blocks < 2:
        raise ValueError("at least two paired blocks required")
    if sys.platform != "linux":
        raise ValueError("this benchmark uses Linux ru_maxrss (KiB) and sched_getaffinity")

    from sciagent.evidence import sha256_file
    script = Path(__file__).resolve()
    roots = {"baseline": baseline.resolve(strict=True), "candidate": candidate.resolve(strict=True)}
    source_hashes = {arm: _source_snapshot(root) for arm, root in roots.items()}
    report = {"schema_version": 2, "seed": seed, "blocks": blocks, "warmup_pairs_per_attempt": 1,
              "affinity": sorted(os.sched_getaffinity(0)), "thread_settings": THREAD_SETTINGS,
              "platform": platform.platform(), "python": platform.python_version(),
              "packages": {k: metadata.version(k) for k in ("numpy", "scipy", "pydantic")},
              "source_roots": {k: str(p) for k, p in roots.items()}, "source_tree_sha256": source_hashes,
              "harness_sha256": sha256_file(script), "expected_verdict": expect_verdict,
              "interval_method": "median paired difference, 4096 percentile block resamples, 95% marginal",
              "attempts": []}
    rng = random.Random(seed)
    for attempt in attempts:
        attempt = attempt.resolve(strict=True)
        inventory = _archive_snapshot(attempt)
        warmup = _measure_pair(attempt, roots, _orders(1, rng)[0], script)
        digest = warmup["baseline"]["verdict_sha256"]
        if expect_verdict is not None and warmup["baseline"]["verdict"] != expect_verdict:
            raise ValueError(f"expected {expect_verdict} verdict on {attempt}")
        rows = []
        for block, order in enumerate(_orders(blocks, rng)):
            pair = _measure_pair(attempt, roots, order, script)
            if pair["baseline"]["verdict_sha256"] != digest:
                raise ValueError(f"complete verdict changed between blocks on {attempt}")
            rows.append({"block": block, **pair})
        if inventory != _archive_snapshot(attempt):
            raise ValueError(f"retained archive changed during measurement on {attempt}")
        times = {arm: median(r[arm]["seconds"] for r in rows) for arm in ARMS}
        rss = {arm: median(r[arm]["process_peak_rss_kib"] for r in rows) for arm in ARMS}
        report["attempts"].append({"path": str(attempt), **inventory, "warmup": warmup, "samples": rows,
                                   "median_seconds": times, "median_process_peak_rss_kib": rss,
                                   "paired_seconds": _paired_effect(rows, "seconds"),
                                   "paired_peak_rss_kib": _paired_effect(rows, "process_peak_rss_kib"),
                                   "speedup_ratio_of_medians": times["baseline"] / times["candidate"]})
    if source_hashes != {arm: _source_snapshot(root) for arm, root in roots.items()}:
        raise ValueError("source files changed during measurement")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--attempt", type=Path, action="append")
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--expect-verdict", choices=("ACCEPTED", "REJECTED", "RETRY_ELIGIBLE"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker)))
    else:
        if not __package__:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        if not all((args.baseline, args.candidate, args.attempt, args.output)):
            parser.error("paired comparison requires --baseline, --candidate, --attempt and --output")
        result = compare(args.baseline, args.candidate, args.attempt, args.blocks, args.seed,
                         expect_verdict=args.expect_verdict)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
