"""Choose and run one reviewed E1 solver using only prior calibration and optional live pilots."""
import argparse
from functools import cache
import hashlib
import json
import math
import os
import platform
import resource
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.e1 import ARMS, THREADS, TIMED_SOURCES, make_request, run_trial, source_fingerprints, workload_definitions, write_json
from sciagent.evidence import sha256_file
from sciagent.spec import sha256_of


def _pinned(path, digest):
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != digest:
        raise ValueError(f"record differs from retained digest: {path.name}")
    return json.loads(contents)


def calibration(directory, digest):
    report = _pinned(directory / "campaign.json", digest)
    cfg = _pinned(directory / "config.json", report["config_sha256"])
    workloads = workload_definitions(cfg["scale"])
    if report["workloads"] != cfg["workloads"] or any(workloads[k] != w for k, w in cfg["workloads"].items()):
        raise ValueError("calibration workload definitions differ from E1")
    @cache
    def contract(name, arm):
        # The request is frozen for this arm across all repetitions of one capture.
        req = make_request(workloads[name], arm)
        return sha256_of(req), {k: sha256_of(getattr(req, k)) for k in ("problem", "validation", "policy")}

    seen = set()
    records = []
    for row in report["records"]:
        name = row["workload"]
        if name not in ("C1", "C2"):
            continue
        arm, block = row["arm"], row["block"]
        pin, frozen = contract(name, arm)
        key = (name, block, arm)
        if (key in seen or type(block) is not int or row["warmup"] is not (block < 0)
                or row["request_sha256"] != cfg["request_sha256"][name][arm] or row["request_sha256"] != pin):
            raise ValueError("uncertified calibration assignment or request")
        if row["eligible"]:
            if (row["state"] != "ACCEPTED" or row["code"] != "OK" or row["verifier_verdict"] != "ACCEPTED"
                    or row["failure_codes"] or row["binary_sha256"] != cfg["binary_sha256"]
                    or row["frozen_sha256"] != frozen or len(row["manifest_sha256"]) != 64
                    or [v["n"] for v in row["levels"]] != workloads[name]["levels"]):
                raise ValueError("inconsistent eligible calibration receipt")
        seen.add(key)
        records.append(row)
    expected = {(w, b, a) for w in ("C1", "C2") for b in range(-cfg["warmups"], cfg["repetitions"]) for a in ARMS}
    if not seen or seen != expected or cfg["repetitions"] < 1:
        raise ValueError("missing or unexpected calibration blocks")
    return cfg, records


def _cpu_model():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def _calibration_cpu(host):
    for line in host["lscpu"].splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return None


def _available_mib():
    memory = Path("/proc/meminfo").read_text().splitlines()
    kb = int(next(line for line in memory if line.startswith("MemAvailable:")).split()[1])
    available = kb // 1024
    limit = Path("/sys/fs/cgroup/memory.max")
    if limit.exists() and limit.read_text().strip() != "max":
        available = min(available, (int(limit.read_text()) - int(limit.with_name("memory.current").read_text())) // 2**20)
    return available


def _identities(source, environment):
    from sciagent.environment import MFEM_FLAGS

    def mapping(value):
        return value if isinstance(value, dict) else {}

    source, environment = mapping(source), mapping(environment)
    pkgs = mapping(environment.get("python_packages"))
    mfem = mapping(environment.get("mfem"))
    flags = mapping(mfem.get("config"))
    return {
        **{k: source.get(k) for k in TIMED_SOURCES},
        "python": environment.get("python"),
        **{k: pkgs.get(k) for k in ("numpy", "scipy", "pydantic")},
        **{f"mfem.{k}": mfem.get(k) for k in ("git", "version", "library_sha256")},
        **{f"mfem.{k}": flags.get(k) for k in MFEM_FLAGS},
    }


def performance_provenance(config, runtime_env, current_sources):
    expected = _identities(config.get("source_sha256") or {}, config["host"].get("environment") or {})
    observed = _identities(current_sources, runtime_env)
    differences = {}
    for name, wanted in expected.items():
        actual = observed[name]
        if wanted is None or wanted == "":
            reason = "missing_calibration_value"
        elif actual is None or actual == "":
            reason = "missing_runtime_value"
        elif type(actual) is not type(wanted) or actual != wanted:
            reason = "value_changed"
        else:
            continue
        differences[name] = {"reason": reason, "calibration": wanted, "runtime": actual}
    return {"matched": not differences, "differences": differences, "runtime_identity": observed}


def select_and_run(directory, job):
    from benchmarks.e1_analysis import choose_solver, fit_policy
    from sciagent import capability, environment, mfem_runner
    from sciagent.profiling import capture

    cfg, records = calibration(Path(job["calibration_directory"]), job["campaign_sha256"])
    deadline = job["deadline_monotonic"]
    cpu = cfg["cpu"]
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("calibrated logical CPU is unavailable")
    os.sched_setaffinity(0, {cpu})
    available = _available_mib()
    w = workload_definitions(cfg["scale"])[job["workload"]]
    name = job["workload"]
    if name not in ("H1", "H2") or any(os.environ.get(k) != v for k, v in THREADS.items()):
        raise ValueError("invalid holdout or thread environment")
    budget = job["memory_mib"]
    effective = min(budget, available)
    allowed = set(job["authorized_solvers"])
    if not allowed or allowed - {"cg_gs","cg_jacobi"}:
        raise ValueError("empty or unknown solver authorization")
    policy = fit_policy(records)
    template = make_request(w, "gs")
    binary = mfem_runner.solver_binary()
    if (not _cpu_model() or _calibration_cpu(cfg["host"]) != _cpu_model()
            or cfg["host"]["kernel"] != platform.release() or cfg["host"]["threads"] != THREADS
            or cfg["host"]["logical_cpu"] != cpu or sha256_file(binary) != cfg["binary_sha256"]):
        raise ValueError("CPU, kernel, threading or binary outside calibration envelope")
    if effective < template.policy.per_level_memory_mb:
        raise ValueError("available memory or requested cap below reviewed worker envelope")
    dofs = template.execution.estimated_dofs(w["levels"][-1])
    if dofs > min(template.validation.reference_solve_max_dofs, template.policy.max_dofs_per_level):
        raise ValueError("final workload exceeds its independent verification cap")
    # This fresh worker and its descendants share a per-process virtual-memory cap.
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = effective * 2**20
    if soft != resource.RLIM_INFINITY:
        cap = min(cap, soft)
    if hard != resource.RLIM_INFINITY:
        cap = min(cap, hard)
    if cap < template.policy.per_level_memory_mb * 2**20:
        raise ValueError("inherited worker memory limit is below the study envelope")
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    env = environment.probe(binary)
    provenance = performance_provenance(cfg, env, source_fingerprints())
    decision_policy = policy if provenance["matched"] else {
        **policy, "enabled": False, "reason": "calibration_provenance_mismatch"}
    catalog = mfem_runner.list_cases(binary)
    requests = {}
    arm_for = {"cg_gs":"gs", "cg_jacobi":"jacobi"}
    for solver in sorted(allowed):
        req = make_request(w, arm_for[solver])
        case = mfem_runner.bind_case(req.problem, catalog)
        decision = capability.evaluate(req.execution, env, case_id=case, catalog=catalog)
        if (case != w["case"] or decision.status != "tested" or req.execution.mesh_type != "tri"
                or req.execution.polynomial_order != 1 or req.execution.assembly != "full"
                or req.problem.equation != "-laplacian(u) = f" or not req.problem.dirichlet_attributes()):
            raise ValueError("operator or solver is outside the reviewed scalar SPD workload")
        requests[solver] = req
    pilot_start = time.monotonic()
    pilots = {}
    if len(allowed) > 1 and (decision_policy["enabled"] or job["measure_pilots"]):
        for solver in sorted(allowed):
            req = requests[solver]
            pilots[solver] = []
            for n in w["levels"][:2]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("wall budget exhausted before pilot")
                path = directory / "pilots" / solver / f"n{n:04d}"
                with capture() as profile:
                    run = mfem_runner.run_level(req.problem, req.execution, w["case"], n, path,
                                                min(remaining, req.policy.per_level_timeout_seconds),
                                                req.policy.per_level_memory_mb, binary=binary)
                if run.exit_code != 0:
                    raise ValueError(f"pilot failed at {solver}/{n}: {run.exit_meaning}")
                result = json.loads((path / "result.json").read_bytes())
                if result["solver"] != solver or result["n"] != n or result["solver_converged"] is not True:
                    raise ValueError("pilot result disagrees with the requested calculation")
                pilots[solver].append({"n": n, "iterations": result["solver_iterations"],
                                       "native_wall_seconds": profile.summary()["native_levels"][0]["process_seconds"],
                                       "solve_seconds": result["time_solve_ms"] / 1000,
                                       "result_sha256": sha256_file(path / "result.json")})
    pilot_cost = time.monotonic() - pilot_start if pilots else 0.
    selection = choose_solver(decision_policy, pilots, target_n=w["levels"][-1], authorized_solvers=allowed,
                              hardware_ok=True, pilot_elapsed_seconds=pilot_cost)
    choice = selection["solver"]
    if choice not in allowed:
        raise ValueError("policy did not select an authorized candidate")
    digest = sha256_of(requests[choice])
    decision = {"workload": name, "case": w, "calibration_sha256": job["campaign_sha256"],
                "calibration_rows": len(records), "config_binary_sha256": cfg["binary_sha256"],
                "cpu": cpu, "cpu_model": _cpu_model(), "available_memory_mib": available,
                "worker_address_space_limit_bytes": cap, "authorized_solvers": sorted(allowed),
                "dofs": dofs, "fit": policy, "performance_provenance": provenance,
                "automatic_switch_enabled": decision_policy["enabled"],
                "diagnostic_pilots_requested": job["measure_pilots"], "pilots": pilots, "selection": selection,
                "request_sha256": digest, "elapsed_before_execution_seconds": time.monotonic() - job["started_monotonic"]}
    path = directory / "decision.json"
    write_json(path, decision)
    decision_sha = sha256_file(path)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("no remaining time for full numerical verification")
    result = run_trial(directory / "final", w, arm_for[choice], binary_sha256=cfg["binary_sha256"], cpu=cpu,
                       expected_request_sha256=digest, timeout=min(remaining, requests[choice].policy.max_total_wall_seconds),
                       supervised_group=True)
    return {"eligible": result["eligible"], "code": result["code"], "solver": choice,
            "decision_sha256": decision_sha, "final": result, "pilot_seconds": pilot_cost}


def run_selection(directory, calibration_directory, campaign_sha256, workload, *, wall_budget=900.,
                  memory_mib=3072, authorized_solvers=("cg_gs","cg_jacobi"), measure_pilots=False):
    started = time.monotonic()
    if (not math.isfinite(wall_budget) or wall_budget <= 0 or type(memory_mib) is not int or memory_mib < 1
            or workload not in ("H1","H2") or not authorized_solvers
            or len(set(authorized_solvers)) != len(authorized_solvers)
            or set(authorized_solvers) - {"cg_gs","cg_jacobi"}):
        raise ValueError("invalid solver authorization, workload or resource limit")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    job = {"calibration_directory": str(Path(calibration_directory).resolve()), "campaign_sha256": campaign_sha256,
           "workload": workload, "authorized_solvers": list(authorized_solvers), "measure_pilots": measure_pilots,
           "memory_mib": memory_mib, "wall_budget": wall_budget,
           "started_monotonic": started, "deadline_monotonic": started+wall_budget}
    write_json(directory / "job.json", job)
    # Include the selector's cold Python imports, calibration, pilots, final worker, and shutdown.
    with (directory / "selection.log").open("wb") as log:
        proc = subprocess.Popen([sys.executable, "-m", "benchmarks.e1_select", "--worker", str(directory)],
                                env={**os.environ, **THREADS}, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            fd = os.pidfd_open(proc.pid)
            try:
                select.select([fd], [], [], max(0., started + wall_budget - time.monotonic()))
            finally:
                os.close(fd)
        finally:
            # Also covers a pilot or solver still running when its Python parent exited.
            # The group leader has not been reaped, so the group ID cannot be recycled.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    elapsed = time.monotonic()-started
    result_path = directory / "outcome.json"
    result = json.loads(result_path.read_bytes()) if result_path.is_file() else {"eligible":False,"code":"WORKER_FAILED"}
    if proc.returncode != 0 or elapsed > wall_budget:
        result.update(eligible=False, code="TIMEOUT" if elapsed > wall_budget else "WORKER_FAILED")
    result.update(elapsed_seconds=elapsed, job_sha256=sha256_file(directory / "job.json"))
    write_json(directory / "receipt.json", result)
    return result


def selection_arguments(p, *, required=True):
    p.add_argument("--directory", type=Path, required=required, help="fresh directory for the decision and verified study")
    p.add_argument("--calibration-directory", type=Path, required=required, help="directory containing the pinned E1 campaign")
    p.add_argument("--campaign-sha256", required=required, help="independently retained calibration report digest")
    p.add_argument("--workload", choices=("H1","H2"), required=required,
                   help="H1 for the larger sine Dirichlet case, H2 for the mixed cosine-exponential case")
    p.add_argument("--wall-budget", type=float, default=900.)
    p.add_argument("--memory-mib", type=int, default=3072)
    p.add_argument("--authorized-solver", choices=("cg_gs","cg_jacobi"), action="append")
    p.add_argument("--measure-pilots", action="store_true")


def submit(args):
    result = run_selection(args.directory, args.calibration_directory, args.campaign_sha256, args.workload,
                           wall_budget=args.wall_budget, memory_mib=args.memory_mib,
                           authorized_solvers=args.authorized_solver or ("cg_gs","cg_jacobi"),
                           measure_pilots=args.measure_pilots)
    print(json.dumps({k: v for k, v in result.items() if k != "final"}, allow_nan=False))
    return 0 if result["eligible"] else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    selection_arguments(p, required=False)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        job = json.loads((args.worker / "job.json").read_bytes())
        try:
            result = select_and_run(args.worker, job)
        except (ValueError, KeyError, OSError, RuntimeError, TypeError, MemoryError) as exc:
            result = {"eligible": False, "code": "SELECTION_FAILED", "reason": str(exc)}
        write_json(args.worker / "outcome.json", result)
        return
    if None in (args.directory, args.calibration_directory, args.campaign_sha256, args.workload):
        p.error("supply directory, calibration directory, campaign SHA-256 and workload")
    return submit(args)


if __name__ == "__main__":
    raise SystemExit(main())
