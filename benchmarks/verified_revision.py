"""Measure submission to verified MFEM output in paired runs of two source trees."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
from statistics import median
import subprocess
import sys

from benchmarks import e1
from benchmarks.e1_analysis import _interval
from sciagent.evidence import sha256_file
from sciagent.mfem_runner import solver_binary
from sciagent.spec import sha256_of


def digest_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def numerical_digest(attempt):
    """Compare all checks and inputs across fresh runs. Normalize paths and native timers."""
    verdict = json.loads((attempt / "verdict.json").read_bytes())
    for level in verdict["levels"]:
        directory = Path("levels") / f"n{level['n']:04d}"
        if Path(level["level_dir"]) != attempt / directory:
            raise ValueError("verdict level points outside the study")
        level["level_dir"] = directory.as_posix()
        result = directory / "result.json"
        if sha256_file(attempt / result) != verdict["input_sha256"][result.as_posix()]:
            raise ValueError("native result digest differs from the verdict")
        raw = json.loads((attempt / result).read_bytes())
        for key in ("time_setup_ms", "time_solve_ms", "time_error_ms"):
            del raw[key]
        verdict["input_sha256"][result.as_posix()] = digest_json(raw)
    return digest_json(verdict)


def ordered_pairs(blocks, rng):
    first = ["baseline", "candidate"]
    firsts = (first * ((blocks + 1) // 2))[:blocks]
    rng.shuffle(firsts)
    for arm in firsts:
        yield [arm, first[1 - first.index(arm)]]


# The unmeasured coordinator imports the arm's E1 driver from that checkout.
# run_trial starts another fresh process and owns the actual endpoint clock.
TRIAL = """
import json, sys
from pathlib import Path
from benchmarks import e1
args = json.load(sys.stdin)
folder = Path(args.pop("folder"))
assert e1.ROOT == Path.cwd()
row = e1.run_trial(folder, **args)
e1.write_json(folder / "trial.json", row)
"""


def capture(args):
    roots = {"baseline": args.baseline.resolve(strict=True), "candidate": args.candidate.resolve(strict=True)}
    source_names = (*e1.TIMED_SOURCES, "benchmarks/e1.py")
    sources = {arm: e1.source_fingerprints(source_names, root) for arm, root in roots.items()}
    if any(value is None for files in sources.values() for value in files.values()):
        raise ValueError("incomplete source tree")
    if any(sources["baseline"][name] != sources["candidate"][name]
           for name in source_names if not name.startswith("verifier/")):
        raise ValueError("paired comparison requires the same controller and timing driver")
    binary = sha256_file(solver_binary())
    if not binary or args.cpu not in os.sched_getaffinity(0) or args.blocks < 2:
        raise ValueError("a real serial solver, an available CPU and at least two blocks are required")
    work = {k: e1.workload_definitions(args.scale)[k] for k in args.workloads}
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    orders = {name: [next(ordered_pairs(2, rng)), *ordered_pairs(args.blocks, rng)] for name in work}
    pins = {name: sha256_of(e1.make_request(w, "gs")) for name, w in work.items()}
    config = {"workloads": work, "request_sha256": pins, "binary_sha256": binary, "cpu": args.cpu,
              "threads": e1.THREADS, "kernel": platform.release(), "python": platform.python_version(),
              "source_sha256": sources, "seed": args.seed, "orders_including_warmup": orders,
              "capture_script_sha256": e1.source_fingerprints(("benchmarks/verified_revision.py",))}
    e1.write_json(directory / "design.json", config)
    records, cache, scores = [], {}, {}
    for name, schedule in orders.items():
        for block, order in enumerate(schedule, start=-1):
            for arm in order:
                folder = directory / f"{name}_{block:03d}_{arm}"
                job = {"folder": str(folder), "workload": work[name], "arm": "gs", "binary_sha256": binary,
                       "expected_request_sha256": pins[name], "cpu": args.cpu, "timeout": 900,
                       "min_free_bytes": 1024**3}
                subprocess.run([sys.executable, "-c", TRIAL], cwd=roots[arm], check=True,
                               env={**os.environ, **e1.THREADS, "PYTHONPATH": str(roots[arm])},
                               input=json.dumps(job), text=True)
                raw = json.loads((folder / "trial.json").read_bytes())
                row = {k: raw[k] for k in ("eligible", "state", "code", "elapsed_seconds", "request_sha256")}
                if raw["eligible"]:
                    row.update(manifest_sha256=raw["manifest_sha256"], archive=raw["archive"],
                               profile=raw["profile"], numerical_sha256=numerical_digest(
                                   folder / raw["archive"] / "attempts" / "attempt_01"))
                    e1._compact(folder, raw, cache)
                row.update(workload=name, revision=arm, block=block, directory=folder.name)
                records.append(row)
                with (directory / "rows.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(name, block, arm, row["code"], round(row["elapsed_seconds"], 3), flush=True)
        rows = [r for r in records if r["workload"] == name]
        if not all(r["eligible"] for r in rows) or len({r["numerical_sha256"] for r in rows}) != 1:
            raise ValueError(f"{name}: an unqualified study or unequal numerical evidence precludes scoring")
        timed = {a: [r["elapsed_seconds"] for r in rows if r["revision"] == a and r["block"] >= 0]
                 for a in roots}
        delta = [c - b for b, c in zip(timed["baseline"], timed["candidate"], strict=True)]
        scores[name] = {"median_seconds": {a: median(v) for a, v in timed.items()},
                        "median_paired_delta_seconds": median(delta), "paired_95pct_seconds": _interval(delta)}
    if any(e1.source_fingerprints(source_names, root) != sources[a] for a, root in roots.items()):
        raise ValueError("measured source changed during capture")
    e1.write_json(directory / "report.json", {"design_sha256": sha256_file(directory / "design.json"),
                  "journal_sha256": sha256_file(directory / "rows.jsonl"), "scores": scores, "records": records})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("baseline", "candidate", "directory"):
        p.add_argument("--" + option, required=True, type=Path)
    p.add_argument("--scale", choices=("memo", "compact"), default="memo")
    p.add_argument("--workloads", choices=("C1", "C2", "H1", "H2"), nargs="+", default=["H1", "H2"])
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--cpu", type=int, required=True)
    capture(p.parse_args())
