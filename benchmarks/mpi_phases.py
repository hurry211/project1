"""Paired MPI launch controls and opt-in phase measurements for the Stage 3 P1 study."""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import platform
import random
from statistics import median
import time

from sciagent import controller, mpi_study as mpi
from sciagent.evidence import EvidenceStore, inspect_manifest, sha256_file
from verifier.mpi import verify_mpi_level, verify_mpi_orders


PHASES = ("mesh_partition_seconds", "assembly_solver_setup_seconds", "pre_solve_barrier_seconds",
          "cg_including_lazy_amg_seconds", "postprocess_collectives_seconds", "partition_export_seconds")
SIZES = (32, 64)
KIND = "mpi_phase_diagnostic_v1"


def schedule(repetitions, seed):
    if type(repetitions) is not int or not 3 <= repetitions <= 30:
        raise ValueError("use 3 to 30 timed blocks")
    rng = random.Random(seed)
    for block in range(-1, repetitions):
        pairs = [(r, n) for r in mpi.RANKS for n in (0, *SIZES)]
        rng.shuffle(pairs)
        for rank, n in pairs:
            yield block, rank, n


def cpu_set(mask):
    cpus = set()
    for part in mask.split(","):
        bounds = list(map(int, part.split("-")))
        if len(bounds) > 2 or min(bounds) < 0 or bounds[-1] < bounds[0]:
            raise ValueError("invalid affinity record")
        cpus.update(range(bounds[0], bounds[-1]+1))
    return cpus


def rank_measurements(directory, ranks, build):
    rows = []
    for rank in range(ranks):
        item = mpi.read(directory / f"phases.{rank}.json")
        if item["rank"] != rank or cpu_set(item["cpus_allowed_list"]) != set(build["cpu_affinity"]):
            raise ValueError("rank identity or CPU envelope differs from the study")
        if not all(math.isfinite(item[k]) and item[k] >= 0 for k in PHASES):
            raise ValueError("invalid rank duration")
        if not math.isclose(sum(item[k] for k in PHASES), item["measured_window_seconds"],
                            rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("phase accounting differs from measured rank window")
        rows.append(item)
    return {k: max(row[k] for row in rows) for k in (*PHASES, "measured_window_seconds")}


def experiment(root, build, serial, repetitions, seed, live):
    screens = {}
    for ranks in mpi.RANKS:
        levels = [mpi._checked_sample(root, f"screen/r{ranks}_n{n}", ranks, n, serial,
                                      build if live else None)[0] for n in mpi.LEVELS]
        screens[str(ranks)] = verify_mpi_orders(levels)
    rows = []
    med = defaultdict(list)
    for block, ranks, n in schedule(repetitions, seed):
        label = f"blocks/b{block}_r{ranks}_n{n}"
        directory = root / label
        mode = "phases" if n else "null"
        if live:
            started = time.perf_counter()
            proc = mpi._launch(directory, build, ranks, n, 120, diagnostic=mode)
        else:
            proc = mpi.read(directory / "process.json")
        if proc["exit_code"] != 0 or proc["timed_out"] or not (0 < proc["launch_seconds"] < 120):
            raise ValueError(f"invalid process record in {label}")
        argv = proc["argv"]
        if n:
            if len(argv) < 3 or not Path(argv[-2]).is_absolute() or not argv[-2].endswith("/" + label):
                raise ValueError(f"invalid study output directory in {label}")
            expect = mpi._mpi_argv(build, ranks, n, argv[-2]) + ["--phase-timing"]
        else:
            launch = mpi._mpi_argv(build, ranks, n, "")
            expect = launch[:launch.index(build["binary"]) + 1] + ["--null-launch"]
        if argv != expect:
            raise ValueError(f"unexpected MPI launcher arguments in {label}")
        times = {"launch_seconds": proc["launch_seconds"]}
        study = {}
        if n:
            verify_start = time.perf_counter()
            metrics = verify_mpi_level(directory, serial / f"n{n:04d}", n=n, ranks=ranks)
            if live:
                stop = time.perf_counter()
                timing = {"verified_workflow_seconds": stop-started,
                          "independent_verification_seconds": stop-verify_start}
                (directory / "measurement.json").write_text(json.dumps(timing) + "\n")
            else:
                timing = mpi.read(directory / "measurement.json")
            phase = rank_measurements(directory, ranks, build)
            if (timing["verified_workflow_seconds"] < times["launch_seconds"] + timing["independent_verification_seconds"]
                    or times["launch_seconds"] < phase["measured_window_seconds"]
                    or not math.isclose(phase["cg_including_lazy_amg_seconds"], metrics["solve_seconds_max"],
                                        rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError(f"invalid verified timing in {label}")
            times.update(phase, **timing)
            study = {k: metrics[k] for k in ("l2_error", "h1_semi_error", "relative_residual",
                                             "serial_relative_field_difference","reference_relative_field_difference")}
        rows.append({"block": block, "ranks": ranks, "n": n, "time": times, **study})
        if block >= 0:
            med[f"r{ranks}_n{n}"].append(times)
    return {"rank_clock_summary": "maximum per phase over ranks, maxima can occur on different ranks",
            "sizes": list(SIZES), "ranks": list(mpi.RANKS), "timed_blocks": repetitions, "warmup_blocks": 1,
            "screen_orders": screens, "samples": rows,
            "median_seconds": {key: {k: median(t[k] for t in arr) for k in arr[0]} for key, arr in med.items()}}


def run(root, repetitions, seed):
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise ValueError("run the Python process with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1")
    store = EvidenceStore(root)
    build = mpi._probe()
    store.write_json_once("mpi_environment.json", build)
    store.write_json_once("protocol.json", {"repetitions": repetitions, "seed": seed,
                                            "kernel": platform.release(), "machine": platform.machine(),
                                            "cpu_model": next((l.split(":", 1)[1].strip()
                                                for l in Path("/proc/cpuinfo").read_text().splitlines()
                                                if l.startswith("model name")), None),
                                            "source_sha256": {name: sha256_file(mpi.environment.PROJECT_ROOT / name)
                                                for name in ("benchmarks/mpi_phases.py", "cpp/mpi/poisson_mpi.cpp",
                                                             "sciagent/mpi_study.py", "verifier/mpi.py")}})
    baseline = controller.run_study(mpi.SERIAL_SPEC, root / "serial")
    serial_digest = sha256_file(Path(baseline["run_dir"]) / "manifest.json")
    serial = mpi._serial_levels(root, serial_digest)
    result = experiment(root, build, serial, repetitions, seed, True)
    store.write_json_once("results.json", result)
    store.finish_manifest({"kind": KIND, "state": "ACCEPTED", "serial_manifest_sha256": serial_digest})
    return sha256_file(root / "manifest.json")


def replay(root, digest):
    manifest, _, changed = inspect_manifest(root, digest)
    if changed or manifest["kind"] != KIND or manifest["state"] != "ACCEPTED":
        raise ValueError("MPI diagnostic archive changed or is incomplete")
    build = mpi.read(root / "mpi_environment.json")
    mpi._check_profile(build)
    serial = mpi._serial_levels(root, manifest["serial_manifest_sha256"])
    protocol = mpi.read(root / "protocol.json")
    result = experiment(root, build, serial, protocol["repetitions"], protocol["seed"], False)
    if not mpi._equivalent(result, mpi.read(root / "results.json")):
        raise ValueError("MPI diagnostic results differ on replay")
    return digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--repetitions", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261124)
    p.add_argument("--replay-sha256", help="verify an existing campaign using the externally retained digest")
    args = p.parse_args()
    root = args.directory.resolve()
    digest = replay(root, args.replay_sha256) if args.replay_sha256 is not None else run(root, args.repetitions, args.seed)
    print(json.dumps({"state": "ACCEPTED", "manifest_sha256": digest}))


if __name__ == "__main__":
    main()
