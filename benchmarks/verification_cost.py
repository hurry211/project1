"""Compare the cost and complete verdict of two verifiers on identical retained fields."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
from statistics import median
import subprocess
import sys
import time


def worker(attempt):
    from verifier.verifier import verify_attempt
    start = time.monotonic()
    verdict = verify_attempt(attempt, require_run_records=True)
    duration = time.monotonic() - start
    encoded = json.dumps(verdict, sort_keys=True, allow_nan=False).encode()
    return {"seconds": duration, "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "verdict": verdict["verdict"], "verdict_sha256": hashlib.sha256(encoded).hexdigest()}


def compare(baseline, candidate, attempts, blocks, seed):
    if blocks < 2:
        raise ValueError("at least two paired blocks required")
    report = {"seed": seed, "blocks": blocks, "affinity": sorted(os.sched_getaffinity(0)), "attempts": []}
    rng = random.Random(seed)
    roots = {"baseline": baseline.resolve(), "candidate": candidate.resolve()}
    script = Path(__file__).resolve()
    for attempt in attempts:
        attempt = attempt.resolve(strict=True)
        rows = []
        for block in range(-1, blocks):
            order = list(roots)
            rng.shuffle(order)
            pair = {}
            for arm in order:
                env = {**os.environ, "PYTHONPATH": str(roots[arm]), "OPENBLAS_NUM_THREADS": "1",
                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                out = subprocess.check_output([sys.executable, str(script), "--worker", str(attempt)],
                                              cwd=roots[arm], env=env, text=True)
                pair[arm] = json.loads(out)
            if pair["baseline"]["verdict_sha256"] != pair["candidate"]["verdict_sha256"]:
                raise ValueError(f"complete verdict differs on {attempt}")
            if block >= 0:
                rows.append({"order": order, **pair})
        medians = {s: median(r[s]["seconds"] for r in rows) for s in roots}
        report["attempts"].append({"path": str(attempt), "samples": rows, "median_seconds": medians,
                                   "speedup_ratio_of_medians": medians["baseline"] / medians["candidate"]})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--attempt", type=Path, action="append")
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker)))
    else:
        if not all((args.baseline, args.candidate, args.attempt, args.output)):
            parser.error("paired comparison requires --baseline, --candidate, --attempt and --output")
        result = compare(args.baseline, args.candidate, args.attempt, args.blocks, args.seed)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
