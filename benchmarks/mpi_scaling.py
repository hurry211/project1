"""Verified strong scaling on one host with compact MPI partitions and independent SA-AMG."""
import argparse
import hashlib
import json
import platform
import os
from pathlib import Path
import random
from statistics import median
import time

from benchmarks.e1_analysis import _interval
from sciagent import mpi_study
from sciagent.evidence import artifact_hashes, sha256_file
from sciagent.mfem_runner import run_level, solver_binary
from sciagent.spec import StudySpec
from verifier.mpi import verify_mpi_level, verify_mpi_orders
from verifier.verifier import verify_level

ARMS = {"serial": 0, "mpi1": 1, "mpi2": 2, "mpi4": 4, "mpi4_tsv": 4}
SIZES = {"small": (32, 64, 128), "large": (128, 256, 512)}


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def runtime_identity():
    """Pin the code and platform that determine the measured endpoint."""
    from importlib.metadata import version

    root = Path(__file__).resolve().parents[1]
    files = ["benchmarks/mpi_scaling.py", "sciagent/mpi_study.py", "sciagent/mfem_runner.py",
             "sciagent/evidence.py", "sciagent/spec.py", "verifier/fem.py", "verifier/verifier.py",
             "verifier/mpi.py", "verifier/expr.py", "verifier/reference_solver.py"]
    return {"kernel": platform.release(), "packages": {k: version(k) for k in ("numpy", "scipy", "pydantic")},
            "cpu_model": next(s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                              if s.startswith("model name")),
            "source_sha256": {f: hashlib.sha256((root/f).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                              for f in files}}


def spec(levels):
    raw = mpi_study.read(mpi_study.SERIAL_SPEC)
    raw["execution"]["mesh_levels"] = levels
    raw["validation"].update(reference_solve_method="amg_pcg", reference_solve_max_dofs=(levels[-1]+1)**2)
    return StudySpec.model_validate(raw)


def schedule(repeats, seed):
    rng = random.Random(seed)
    if not 3 <= repeats <= 30:
        raise ValueError("use 3 to 30 scored blocks")
    # Rotate a randomly permuted five-arm ordering to balance positions within each cycle.
    order = list(ARMS)
    for i in range(-1, repeats):
        if i <= 0 or i % len(order) == 0:
            rng.shuffle(order)
        else:
            order = order[1:] + order[:1]
        for name in order:
            yield i, name


def serial_verdict(directory, n, study):
    report = verify_level(directory, study.problem, study.execution, study.validation, n, require_run_record=True)
    if not report.ok:
        raise ValueError([(c.code, c.detail) for c in report.checks if not c.passed])
    return {"n": n, "ranks": 0, "l2_error": report.l2_error, "h1_semi_error": report.h1_semi_error,
            "reference_relative_field_difference": report.reference_rel_diff, "iterations": report.solver_iterations}


def trial(root, label, arm, study, build, *, replay=False, control=None):
    ranks = ARMS[arm]
    levels = study.execution.mesh_levels
    verdicts = []
    launch = []
    start = time.perf_counter()
    for n in levels:
        directory = root / label / f"n{n}"
        if ranks == 0:
            if not replay:
                rec = run_level(study.problem, study.execution, "sin_sin", n, directory, 300, 3072)
                if rec.exit_code != 0:
                    raise RuntimeError(f"serial level {n} failed")
            record = mpi_study.read(directory / "run_record.json")
            launch.append(record["wall_seconds"])
            verdicts.append(serial_verdict(directory, n, study))
        else:
            compact = arm != "mpi4_tsv"
            diag = "compact" if compact else "phases"
            if not replay:
                mpi_study._launch(directory, build, ranks, n, 300, diagnostic=diag)
            record = mpi_study.read(directory / "process.json")
            if record["exit_code"] != 0 or record["timed_out"] is not False:
                raise ValueError("unsuccessful MPI process")
            launch.append(record["launch_seconds"])
            verdicts.append(verify_mpi_level(directory, (control or root / "control") / f"n{n}", n=n, ranks=ranks,
                                             streaming=True, compact=compact))
    orders = verify_mpi_orders(verdicts, expected_levels=tuple(levels))
    if verdicts[-1]["l2_error"] > 2 / levels[-1]**2 or verdicts[-1]["h1_semi_error"] > 5 / levels[-1]:
        raise ValueError("independently integrated error above accuracy target")
    pins = artifact_hashes(root / label)
    wall = time.perf_counter() - start
    return {"verdict": "ACCEPTED", "orders": orders, "levels": verdicts, "launch_seconds": sum(launch),
            "verified_seconds": wall, "files": pins}


def scores(rows):
    timed = {arm: [x["verified_seconds"] for x in rows if x["block"] >= 0 and x["arm"] == arm] for arm in ARMS}
    diffs = {}
    for base, candidate in (("mpi1", "mpi2"), ("mpi1","mpi4"), ("mpi4_tsv","mpi4"), ("serial","mpi4")):
        delta = [c-b for b,c in zip(timed[base], timed[candidate], strict=True)]
        diffs[f"{candidate}_vs_{base}"] = {"ratio_of_medians": median(timed[base])/median(timed[candidate]),
                                           "median_paired_seconds": median(delta), "paired_95pct_seconds": _interval(delta)}
    return {"median_verified_seconds": {arm: median(values) for arm,values in timed.items()}, "paired": diffs}


def run(root, scale, repeats, seed):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    build = mpi_study._probe()
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) != 4 or any(os.getenv(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise ValueError("select four CPUs and one numerical thread")
    study = spec(SIZES[scale])
    design = {"scale": scale, "repeats": repeats, "seed": seed, "cpus": cpus, "request": study.model_dump(mode="json"),
              "serial_sha256": sha256_file(solver_binary()), "mpi_build": build, "runtime": runtime_identity()}
    save(root / "design.json", design)
    rows = []
    try:
        os.sched_setaffinity(0, cpus[:1])
        anchor = trial(root, "control", "serial", study, build)
        save(root / "control.json", anchor)
        for block, arm in schedule(repeats, seed):
            os.sched_setaffinity(0, cpus[:max(1, ARMS[arm])])
            label = f"block{block}_{arm}"
            sample = trial(root, label, arm, study, build)
            row = dict(block=block, arm=arm, label=label, **sample)
            rows.append(row)
            with (root / "rows.jsonl").open("a") as log:
                log.write(json.dumps(row, allow_nan=False) + "\n")
            print(label, round(row["verified_seconds"],3), flush=True)
    finally:
        os.sched_setaffinity(0, cpus)
    save(root / "report.json", {"design_sha256": sha256_file(root/"design.json"), "control_sha256": sha256_file(root/"control.json"),
                                "rows_sha256": sha256_file(root/"rows.jsonl"), "scores": scores(rows)})
    return sha256_file(root / "report.json")


def load_capture(root, pin):
    if sha256_file(root / "report.json") != pin:
        raise ValueError("report digest mismatch")
    report = mpi_study.read(root / "report.json")
    for filename in ("design", "control", "rows"):
        suffix = "jsonl" if filename == "rows" else "json"
        if sha256_file(root / f"{filename}.{suffix}") != report[filename + "_sha256"]:
            raise ValueError(f"{filename} digest mismatch")
    cfg = mpi_study.read(root / "design.json")
    study = spec(SIZES[cfg["scale"]])
    if cfg["request"] != study.model_dump(mode="json"):
        raise ValueError("unexpected serial control request")
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    order = list(schedule(cfg["repeats"], cfg["seed"]))
    if [(r["block"],r["arm"]) for r in rows] != order or report["scores"] != scores(rows):
        raise ValueError("schedule or score disagrees")
    return cfg, study, rows, order


def replay(root, pin):
    cfg, study, rows, order = load_capture(root, pin)
    samples = [("control", "serial", mpi_study.read(root/"control.json"))]
    for (block, arm), row in zip(order, rows, strict=True):
        label = f"block{block}_{arm}"
        if row["label"] != label:
            raise ValueError("unexpected sample path")
        samples.append((label, arm, row))
    for label, arm, row in samples:
        observed = trial(root, label, arm, study, cfg["mpi_build"], replay=True)
        for key in ("files","levels","orders","launch_seconds","verdict"):
            if observed[key] != row[key]:
                raise ValueError(f"{label}: {key} differs")
    return {"studies_replayed": len(samples), "manifest_sha256": pin}


def choose_arm(rows):
    """Consider a switch when the upper paired interval clears a 5% saving."""
    by_arm = {arm: [] for arm in ("serial", "mpi1", "mpi2", "mpi4")}
    for row in rows:
        if row["verdict"] != "ACCEPTED":
            raise ValueError("ineligible calibration study")
        if row["block"] >= 0 and row["arm"] in by_arm:
            by_arm[row["arm"]].append(row["verified_seconds"])
    baseline = by_arm["serial"]
    if len(baseline) < 5 or any(len(v) != len(baseline) for v in by_arm.values()):
        return "serial"
    eligible = ["serial"]
    for arm, times in by_arm.items():
        if arm != "serial":
            delta = [t - b for b, t in zip(baseline, times, strict=True)]
            if _interval(delta)[1] < -0.05 * median(baseline):
                eligible.append(arm)
    return min(eligible, key=lambda arm: median(by_arm[arm]))


def select(root, pin, destination):
    start = time.perf_counter()
    cfg, study, rows, _ = load_capture(root, pin)
    build = mpi_study._probe()
    if (cfg["mpi_build"] != build or cfg["runtime"] != runtime_identity()
            or cfg["serial_sha256"] != sha256_file(solver_binary())
            or cfg["cpus"] != sorted(os.sched_getaffinity(0))):
        raise ValueError("source, binary, CPU or thread environment differs from calibration")
    anchor = mpi_study.read(root / "control.json")
    if anchor["verdict"] != "ACCEPTED" or anchor["files"] != artifact_hashes(root / "control"):
        raise ValueError("serial anchor differs from calibration")
    choice = choose_arm(rows)
    destination.mkdir(parents=True, exist_ok=False)
    decision = {"arm": choice, "campaign_sha256": pin, "seconds_before_native_launch": time.perf_counter()-start}
    save(destination/"decision.json", decision)
    cpus = cfg["cpus"]
    try:
        os.sched_setaffinity(0, cpus[:max(1, ARMS[choice])])
        verified = trial(destination, "selected", choice, study, build, control=root/"control")
    finally:
        os.sched_setaffinity(0, cpus)
    receipt = {"decision_sha256": sha256_file(destination/"decision.json"), "verified": verified,
               "seconds_including_decision": time.perf_counter()-start}
    save(destination/"receipt.json", receipt)
    return {"arm": choice, "verdict": verified["verdict"], "receipt_sha256": sha256_file(destination/"receipt.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--scale", choices=SIZES, default="large")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--replay-sha256")
    parser.add_argument("--select-destination", type=Path, help="run a prospective choice using a pinned capture")
    args = parser.parse_args()
    if args.select_destination:
        if not args.replay_sha256:
            parser.error("--select-destination needs --replay-sha256")
        print(select(args.directory.resolve(), args.replay_sha256, args.select_destination.resolve()))
        raise SystemExit(0)
    print(replay(args.directory, args.replay_sha256) if args.replay_sha256 else
          run(args.directory, args.scale, args.repetitions, args.seed))
