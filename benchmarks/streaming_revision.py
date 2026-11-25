"""Compare full independent verification on the same pinned multi-level MFEM study."""
import argparse
import json
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
import time

from benchmarks.e1_analysis import _interval
from benchmarks.verification_cost import _orders, _source_snapshot
from sciagent.evidence import inspect_manifest, sha256_file


def measure(source, attempt):
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONPATH": str(source)}
    start = time.monotonic()
    worker = subprocess.run([sys.executable, "-m", "sciagent.verification_worker", str(attempt), "3072", "--profile"],
                            input="{}", cwd=source, env=env, text=True, capture_output=True, timeout=600, check=True)
    elapsed = time.monotonic() - start
    reply = json.loads(worker.stdout)
    verdict = reply["verdict"]
    if verdict["verdict"] != "ACCEPTED" or verdict["failure_codes"]:
        raise ValueError(f"unqualified verifier: {verdict['failure_codes']}")
    checks = [(lv["n"], [(c["name"], c["passed"], c["threshold"]) for c in lv["checks"]]) for lv in verdict["levels"]]
    study_checks = [(c["name"], c["passed"], c["threshold"]) for c in verdict["study_checks"]]
    signature = (checks, study_checks, verdict["accuracy_check"], verdict["input_sha256"])
    metrics = [{"n": l["n"], "dofs": l["num_dofs"], "l2": l["l2_error"], "h1": l["h1_semi_error"],
                "reference_difference": l["reference_rel_diff"]} for l in verdict["levels"]]
    return {"seconds": elapsed, "rss_kib": reply["peak_rss_kib"], "phases": reply["phase_seconds"], "levels": metrics}, signature


def compare(root, pin, before, after, blocks, seed):
    manifest, _, changed = inspect_manifest(root, pin)
    if changed or manifest["final_state"] != "ACCEPTED" or len(manifest["attempts"]) != 1:
        raise ValueError("requires a complete single-attempt MFEM archive")
    if blocks < 2 or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("pin to one CPU and supply at least two scored blocks")
    attempt = root / "attempts/attempt_01"
    sources = {"baseline": before.resolve(), "candidate": after.resolve()}
    hashes = {k: _source_snapshot(path) for k, path in sources.items()}
    rows = []
    rng = random.Random(seed)
    signature = target = None
    for block, order in enumerate([_orders(1, rng)[0], *_orders(blocks, rng)], -1):
        row = {"block": block, "order": order}
        for arm in order:
            row[arm], sig = measure(sources[arm], attempt)
            if signature is None:
                signature, target = sig, row[arm]["levels"]
            if sig != signature:
                raise ValueError("gate set, thresholds, accuracy decision or raw input hashes changed")
            for new, old in zip(row[arm]["levels"], target, strict=True):
                if new["n"] != old["n"] or new["dofs"] != old["dofs"] or new["l2"] != old["l2"] or new["h1"] != old["h1"]:
                    raise ValueError("error integration or mesh changed")
        rows.append(row)
    if inspect_manifest(root, pin)[2] or any(_source_snapshot(sources[a]) != hashes[a] for a in sources):
        raise ValueError("sources or archive changed during comparison")
    scored = rows[1:]
    delta = [r["candidate"]["seconds"] - r["baseline"]["seconds"] for r in scored]
    return {"manifest_sha256": pin, "source_sha256": hashes, "seed": seed, "cpu": min(os.sched_getaffinity(0)),
            "address_space_limit_mib": 3072, "all_verdicts": "ACCEPTED", "input_sha256": signature[-1],
            "samples": rows, "median_seconds": {a: median(r[a]["seconds"] for r in scored) for a in sources},
            "median_rss_kib": {a: median(r[a]["rss_kib"] for r in scored) for a in sources},
            "paired_delta_seconds": median(delta), "paired_95pct_seconds": _interval(delta)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "baseline", "candidate", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    report = compare(args.study.resolve(), args.manifest_sha256, args.baseline, args.candidate, args.blocks, args.seed)
    with args.output.open("x") as out:
        out.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
