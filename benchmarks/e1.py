"""Paired E1 acquisition on a single Linux CPU, including full independent verification."""
import argparse
import json
import math
import os
import platform
import random
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from sciagent.evidence import sha256_file
from sciagent.spec import StudySpec, canonical_json, sha256_of

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"gs": "cg_gs", "jacobi": "cg_jacobi", "gs_repeat": "cg_gs"}
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def workload_definitions(scale="memo"):
    sizes = {"memo": (64, 128, 512, 128), "compact": (16, 32, 64, 32)}[scale]
    count = 4 if scale == "memo" else 3
    return {
        name: {"case": "cos_exp" if name == "H2" else "sin_sin",
               "levels": [n // 2**i for i in reversed(range(count))],
               "l2_target": 2.0 / n**2, "h1_target": 5.0 / n}
        for name, n in zip(("C1", "C2", "H1", "H2"), sizes, strict=True)
    }


def make_request(workload, arm):
    template = {"sin_sin": "poisson_sin_sin_p1_tri.json",
                "cos_exp": "poisson_cos_exp_mixed_p1_quad.json"}[workload["case"]]
    raw = json.loads((ROOT / "specs" / template).read_bytes())
    raw["execution"].update(mesh_type="tri", polynomial_order=1, assembly="full",
                            solver=ARMS[arm], mesh_levels=workload["levels"])
    raw["validation"]["max_l2_error_finest"] = workload["l2_target"]
    # A refinement would change the equal-work comparison. One attempt per arm.
    raw["policy"]["max_attempts"] = 1
    return StudySpec.model_validate(raw)


def write_json(path, data):
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def block_schedule(workloads, repetitions, warmups, seed):
    rng = random.Random(seed)
    for workload in workloads:
        for block in range(-warmups, repetitions):
            arms = list(ARMS)
            rng.shuffle(arms)
            for arm in arms:
                yield workload, block, arm


def _frozen(study):
    return {key: sha256_of(getattr(study, key)) for key in ("problem", "policy", "validation")}


def qualification(manifest, verdict, *, frozen, binary_sha256, request_matches, workload):
    if manifest.get("final_state") != "ACCEPTED":
        return False, manifest.get("final_code", "INCOMPLETE")
    if manifest.get("frozen_sha256") != frozen or not request_matches:
        return False, "SPEC_MISMATCH"
    if manifest.get("environment", {}).get("solver_binary_sha256") != binary_sha256:
        return False, "BINARY_MISMATCH"
    if len(manifest["attempts"]) != 1 or verdict.get("verdict") != "ACCEPTED" or verdict.get("failure_codes") != []:
        return False, "VERDICT_MISMATCH"
    levels = verdict["levels"]
    if [lv["n"] for lv in levels] != workload["levels"]:
        return False, "LEVEL_MISMATCH"
    finest = levels[-1]
    if not (0 <= finest["l2_error"] <= workload["l2_target"] and
            0 <= finest["h1_semi_error"] <= workload["h1_target"]):
        return False, "ACCURACY_TARGET"
    return True, "OK"


def worker(directory):
    # The environment's BLAS thread limits are set before the child imports SciPy.
    from sciagent.controller import run_study
    from sciagent.mfem_runner import solver_binary
    from sciagent.profiling import capture

    job = json.loads((directory / "job.json").read_bytes())
    os.sched_setaffinity(0, {job["cpu"]})
    if sha256_file(solver_binary()) != job["binary_sha256"]:
        write_json(directory / "worker.json", {"state": "FAILED", "code": "BINARY_MISMATCH", "eligible": False})
        return
    with capture() as profile:
        manifest = run_study(directory / "request.json", directory / "studies",
                             expected_spec_sha256=job["request_sha256"])
    archive = Path(manifest["run_dir"])
    attempt = archive / "attempts/attempt_01"
    verdict_path = attempt / "verdict.json"
    verdict = json.loads(verdict_path.read_bytes()) if verdict_path.exists() else {}
    pinned = sha256_file(archive / "manifest.json")
    expected = sha256_file(archive / "request/study_spec.json") == job["request_sha256"]
    eligible, code = qualification(manifest, verdict, frozen=job["frozen_sha256"],
                                   binary_sha256=job["binary_sha256"], request_matches=expected,
                                   workload=job["workload"])
    data = profile.summary()
    by_n = {v["n"]: v for v in data["native_levels"]}
    levels = []
    for level in verdict.get("levels", []):
        n = level["n"]
        native = attempt / "levels" / f"n{n:04d}" / "result.json"
        raw = json.loads(native.read_bytes()) if native.exists() else {}
        levels.append({"n": n, "iterations": raw.get("solver_iterations"),
                       "final_norm": raw.get("solver_final_norm"),
                       "relative_residual": raw.get("diagnostics", {}).get("algebraic_residual", {}).get("relative_l2"),
                       "solve_seconds": raw["time_solve_ms"] / 1000 if raw.get("time_solve_ms") is not None else None,
                       "native_wall_seconds": by_n.get(n, {}).get("process_seconds"),
                       "l2_error": level["l2_error"], "h1_semi_error": level["h1_semi_error"]})
    write_json(directory / "worker.json", {
        "eligible": eligible, "state": manifest["final_state"], "code": code,
        "verifier_verdict": verdict.get("verdict"), "failure_codes": verdict.get("failure_codes", []),
        "frozen_sha256": manifest.get("frozen_sha256"), "binary_sha256": manifest.get("environment", {}).get("solver_binary_sha256"),
        "manifest_sha256": pinned, "archive": str(archive.relative_to(directory)), "levels": levels, "profile": data})


def allocated_bytes(path):
    # Physical blocks once per inode, including hardlinked evidence.
    seen = set()
    total = 0
    for base, _, files in os.walk(path):
        for name in files:
            try:
                st = os.stat(os.path.join(base, name), follow_symlinks=False)
            except FileNotFoundError:
                continue
            key = (st.st_dev, st.st_ino)
            if key not in seen:
                total += 512 * st.st_blocks
                seen.add(key)
    return total


def run_trial(directory, workload, arm, *, binary_sha256, cpu, timeout, min_free_bytes=0,
              storage_limit_bytes=None, expected_request_sha256=None):
    started = time.monotonic()
    if cpu not in os.sched_getaffinity(0) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("unavailable CPU or invalid wall budget")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    study = make_request(workload, arm)
    request = directory / "request.json"
    request.write_text(canonical_json(study))
    digest = sha256_file(request)
    if expected_request_sha256 is not None and digest != expected_request_sha256:
        raise ValueError("request differs from preregistered acquisition protocol")
    job = {"cpu": cpu, "workload": workload, "arm": arm, "binary_sha256": binary_sha256,
           "request_sha256": digest, "frozen_sha256": _frozen(study)}
    write_json(directory / "job.json", job)
    def storage_exhausted():
        return (shutil.disk_usage(directory).free < min_free_bytes or
                (storage_limit_bytes is not None and allocated_bytes(directory) >= storage_limit_bytes))
    code = "STORAGE_LIMIT" if storage_exhausted() else None
    if not code:
        with (directory / "worker.log").open("wb") as log:
            process = subprocess.Popen([sys.executable, "-m", "benchmarks.e1", "--worker", str(directory)],
                                       env={**os.environ, **THREADS}, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                fd = os.pidfd_open(process.pid)
                try:
                    while True:
                        remaining = started + timeout - time.monotonic()
                        if remaining <= 0:
                            code = "TIMEOUT"
                            break
                        # A pidfd wakes at exit without adding a fixed polling delay to timing.
                        if select.select([fd], [], [], min(1.0, remaining))[0]:
                            break
                        if storage_exhausted():
                            code = "STORAGE_LIMIT"
                            break
                finally:
                    os.close(fd)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if process.returncode and not code:
            code = "WORKER_FAILED"
    elapsed = time.monotonic() - started
    if not code and elapsed > timeout:
        code = "TIMEOUT"
    if not code and storage_exhausted():
        code = "STORAGE_LIMIT"
    row = {"state": "FAILED", "code": code, "eligible": False, "levels": []} if code else json.loads(
        (directory / "worker.json").read_bytes())
    return {**row, "arm": arm, "elapsed_seconds": elapsed, "request_sha256": digest}


def _compact(directory, row, cache):
    """Retain every archive path, sharing identical finalized field bytes on disk."""
    if not row.get("eligible"):
        return
    archive = directory / row["archive"]
    manifest = json.loads((archive / "manifest.json").read_bytes())
    for rel, digest in manifest["artifact_sha256"].items():
        if "/levels/" not in rel:
            continue
        path = archive / rel
        if path.stat().st_size < 8192:
            continue
        if digest not in cache:
            cache[digest] = path
        else:
            previous = cache[digest]
            if sha256_file(previous) != digest or sha256_file(path) != digest:
                raise ValueError("archive bytes changed before compaction")
            temp = path.with_suffix(path.suffix + ".link")
            os.link(previous, temp)
            temp.replace(path)


def _host(directory, cpu):
    from sciagent.environment import probe
    env = probe()
    def output(command):
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"environment": env, "lscpu": output(["lscpu"]), "affinity_before_worker": sorted(os.sched_getaffinity(0)),
            "logical_cpu": cpu, "cpu_governor": output(["cat", f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"]),
            "kernel": platform.release(), "threads": THREADS, "filesystem": output(["stat", "-f", "-c", "%T", str(directory)]),
            "blas": output([sys.executable, "-c", "import numpy as np; np.show_config()"])}


def campaign(args):
    from sciagent.mfem_runner import solver_binary
    allowed = os.sched_getaffinity(0)
    cpu = min(allowed) if args.cpu is None else args.cpu
    if cpu not in allowed or args.repetitions < 1 or args.warmups < 0:
        raise ValueError("unavailable CPU or invalid repetition count")
    budgets = [args.study_timeout, args.wall_budget, args.max_output_gib]
    if any(not math.isfinite(x) or x <= 0 for x in budgets) or not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        raise ValueError("invalid campaign resource budget")
    if len(set(args.workloads)) != len(args.workloads):
        raise ValueError("duplicate workload")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    workloads = {k: workload_definitions(args.scale)[k] for k in args.workloads}
    binary = sha256_file(solver_binary())
    if binary is None:
        raise FileNotFoundError(solver_binary())
    pins = {name: {arm: sha256_of(make_request(w, arm)) for arm in ARMS} for name, w in workloads.items()}
    config = {"schema_version": 1, "scale": args.scale, "cpu": cpu, "seed": args.seed, "warmups": args.warmups,
              "repetitions": args.repetitions, "study_timeout": args.study_timeout, "wall_budget": args.wall_budget,
              "max_output_gib": args.max_output_gib, "min_free_gib": args.min_free_gib,
              "compact_archives": args.compact_archives, "workloads": workloads, "request_sha256": pins,
              "binary_sha256": binary, "host": _host(directory, cpu)}
    write_json(directory / "config.json", config)
    rows = []
    state = "COMPLETE"
    stopped = set()
    started = time.monotonic()
    cache = {}
    for name, block, arm in block_schedule(workloads, args.repetitions, args.warmups, args.seed):
        if (name, arm) in stopped:
            continue
        remaining = started + args.wall_budget - time.monotonic()
        budget = int(args.max_output_gib * 2**30) - allocated_bytes(directory)
        if remaining <= 0 or budget <= 0 or shutil.disk_usage(directory).free < args.min_free_gib * 2**30:
            state = "BUDGET_EXHAUSTED"
            break
        label = f"{name}_{'warm' if block < 0 else 'block'}{abs(block):03d}_{arm}"
        folder = directory / label
        row = run_trial(folder, workloads[name], arm, binary_sha256=binary, cpu=cpu,
                        timeout=min(remaining, args.study_timeout), min_free_bytes=int(args.min_free_gib*2**30),
                        storage_limit_bytes=budget, expected_request_sha256=pins[name][arm])
        row.update(workload=name, block=block, warmup=block < 0, directory=label)
        with (directory / "rows.jsonl").open("a") as out:
            out.write(json.dumps(row, allow_nan=False) + "\n")
        rows.append(row)
        print(f"{label} {row['code']} {row['elapsed_seconds']:.3f}s", flush=True)
        if args.compact_archives:
            _compact(folder, row, cache)
        if row["code"] == "STORAGE_LIMIT":
            state = "BUDGET_EXHAUSTED"
            break
        if row["code"] == "TIMEOUT":
            stopped.add((name, arm))
    if stopped and state == "COMPLETE":
        state = "PARTIAL_TIMEOUT"
    report = {"status": state, "config_sha256": sha256_file(directory / "config.json"), "workloads": workloads,
              "records": rows, "acquisition_seconds": time.monotonic()-started,
              "scheduled_trials": len(workloads)*(args.warmups+args.repetitions)*len(ARMS),
              "completed_trials": len(rows), "stopped_arms": sorted(stopped)}
    write_json(directory / "campaign.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    destination = p.add_mutually_exclusive_group(required=True)
    destination.add_argument("--directory", type=Path)
    destination.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--scale", choices=("compact", "memo"), default="memo")
    p.add_argument("--workloads", choices=("C1","C2","H1","H2"), nargs="+", default=("C1","C2","H2","H1"))
    p.add_argument("--seed", type=int, default=20261124)
    p.add_argument("--repetitions", type=int, default=30)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--cpu", type=int)
    p.add_argument("--study-timeout", type=float, default=900.0)
    p.add_argument("--wall-budget", type=float, default=14400.0)
    p.add_argument("--min-free-gib", type=float, default=1.0)
    p.add_argument("--max-output-gib", type=float, default=12.0)
    p.add_argument("--compact-archives", action="store_true")
    args = p.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        report = campaign(args)
        if report["status"] != "COMPLETE" or not all(r["eligible"] for r in report["records"]):
            raise SystemExit(2)


if __name__ == "__main__":
    main()

