"""Experimental, independently checked FA/PA comparison; never promotes capabilities.

Linux only (POSIX resource limits and ru_maxrss in KiB). Run from repo root.
Every trial gets a new directory. No production acceptance or archive claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

from sciagent import mfem_runner
from sciagent.spec import StudySpec, canonical_json, ExecutionSpec

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("time_setup_ms", "time_assembly_system_ms", "time_preconditioner_setup_ms",
           "time_cg_iterate_ms", "time_solve_ms", "time_error_ms", "peak_rss_kib")


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def limits(mb):
    def apply():
        resource.setrlimit(resource.RLIMIT_AS, (mb * 1024**2,) * 2)
    return apply


def verify_worker(attempt):
    from verifier.verifier import verify_attempt
    t = time.monotonic()
    verdict = verify_attempt(Path(attempt), require_run_records=True)
    dump(Path(attempt) / "verdict.json", verdict)
    print(json.dumps({"verifier_seconds": time.monotonic() - t,
                      "verifier_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
    return 0 if verdict["verdict"] == "ACCEPTED" else 1


def one_trial(study, assembly, directory, binary, timeout, memory):
    start = time.monotonic()
    ex = ExecutionSpec.model_validate({**study.execution.model_dump(), "assembly": assembly,
                                       "solver": "cg_jacobi"})
    attempt = directory / "attempts" / "attempt_01"
    attempt.mkdir(parents=True, exist_ok=False)
    frozen = directory / "frozen"
    frozen.mkdir()
    for name in ("problem", "validation", "policy"):
        (frozen / (name + ".json")).write_text(canonical_json(getattr(study, name)))
    (attempt / "execution.json").write_text(canonical_json(ex))
    case = mfem_runner.bind_case(study.problem, mfem_runner.list_cases(binary))
    if case is None:
        raise RuntimeError("manufactured case not compiled")
    measurements = []
    for n in ex.mesh_levels:
        level = attempt / "levels" / f"n{n:04d}"
        run = mfem_runner.run_level(study.problem, ex, case, n, level,
                                    timeout_s=timeout, mem_mb=memory, binary=binary)
        if run.exit_code != 0:
            raise RuntimeError(f"native failure at {level}: {run.exit_meaning}")
        raw = json.loads((level / "result.json").read_text())
        metrics = {k: raw[k] for k in METRICS}
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
               for v in metrics.values()) or metrics["peak_rss_kib"] <= 0:
            raise RuntimeError("missing or invalid native performance measurement")
        measurements.append({"n": n, "dofs": raw["num_dofs"],
                             "iterations": raw["solver_iterations"],
                             "native_wall_seconds": run.wall_seconds, **metrics})
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    worker = subprocess.run([sys.executable, "-m", "benchmarks.performance",
                             "--verify-attempt", str(attempt.resolve())],
                            capture_output=True, text=True, timeout=timeout,
                            preexec_fn=limits(memory), env=env, cwd=ROOT)
    (attempt / "verify.stdout.txt").write_text(worker.stdout)
    (attempt / "verify.stderr.txt").write_text(worker.stderr)
    if worker.returncode != 0:
        raise RuntimeError(f"verification failed at {attempt}: exit {worker.returncode}")
    v = json.loads((attempt / "verdict.json").read_text())
    if v["verdict"] != "ACCEPTED":
        raise RuntimeError("independent verifier did not accept")
    for row, check in zip(measurements, v["levels"], strict=True):
        if row["n"] != check["n"]:
            raise RuntimeError("level mismatch")
        row.update(l2=check["l2_error"], h1=check["h1_semi_error"],
                   reference_rel_diff=check["reference_rel_diff"])
    return {"assembly": assembly, "levels": measurements, "verdict": v["verdict"],
            "input_sha256": v["input_sha256"], **json.loads(worker.stdout),
            "end_to_end_seconds": time.monotonic() - start}


def compare_pair(full, partial):
    """Reject mismatched numerical work, before reporting any timing comparison."""
    if full["assembly"] != "full" or partial["assembly"] != "partial":
        raise ValueError("invalid pair")
    if full["verdict"] != "ACCEPTED" or partial["verdict"] != "ACCEPTED":
        raise ValueError("unverified pair")
    for a, b in zip(full["levels"], partial["levels"], strict=True):
        if (a["n"], a["dofs"]) != (b["n"], b["dofs"]):
            raise ValueError("different mesh or DOFs")
        for key in ("l2", "h1"):
            if not math.isclose(a[key], b[key], rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError(f"different verified {key} errors")
    if not all(math.isfinite(x["end_to_end_seconds"]) and x["end_to_end_seconds"] > 0
               for x in (full, partial)):
        raise ValueError("invalid trial time")
    return full["end_to_end_seconds"] / partial["end_to_end_seconds"]


def run(output, repeats, order, levels, timeout, memory):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = mfem_runner.solver_binary().resolve()
    source = ROOT / "specs" / f"poisson_cos_exp_mixed_p{order}_quad_pa.json"
    raw = json.loads(source.read_text())
    raw["execution"]["mesh_levels"] = levels
    study = StudySpec.model_validate(raw)
    if len(levels) < 3 or study.execution.estimated_dofs(levels[-1]) > 100_000:
        raise ValueError("need >=3 doubling levels and <=100000 DOFs")
    report = {"status": "INCOMPLETE", "scope": "experimental serial CPU; production capability unchanged",
              "platform": os.uname()._asdict() if hasattr(os.uname(), "_asdict") else list(os.uname()),
              "binary_sha256": digest(binary), "source_sha256": digest(source),
              "study": study.model_dump(mode="json"), "repeats": repeats,
              "timeout_seconds_per_process": timeout, "address_space_limit_mb_per_process": memory,
              "pairs": []}
    report_path = output / "report.json"
    dump(report_path, report)
    for i in range(repeats):
        pair = {}
        # Alternate ordering to reduce systematic warm-cache bias.
        for asm in (("full", "partial") if i % 2 == 0 else ("partial", "full")):
            pair[asm] = one_trial(study, asm, output / f"rep{i+1:02d}_{asm}", binary, timeout, memory)
        pair["end_to_end_speedup_full_over_partial"] = compare_pair(pair["full"], pair["partial"])
        report["pairs"].append(pair)
        dump(report_path, report)
    summary = {}
    for asm in ("full", "partial"):
        trials = [pair[asm] for pair in report["pairs"]]
        summary[asm] = {
            "median_end_to_end_seconds": statistics.median(t["end_to_end_seconds"] for t in trials),
            "median_verifier_seconds": statistics.median(t["verifier_seconds"] for t in trials),
            "median_verifier_peak_rss_kib": statistics.median(t["verifier_peak_rss_kib"] for t in trials),
            "median_by_level": [
                {"n": n, **{key: statistics.median(t["levels"][j][key] for t in trials)
                             for key in (*METRICS, "native_wall_seconds", "iterations", "l2", "h1")}}
                for j, n in enumerate(levels)]}
    ratios = [p["end_to_end_speedup_full_over_partial"] for p in report["pairs"]]
    report.update(status="COMPLETE", summary=summary,
                  median_paired_speedup=statistics.median(ratios),
                  paired_speedup_range=[min(ratios), max(ratios)])
    dump(report_path, report)
    print(report_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-attempt", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--output", type=Path)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--order", type=int, choices=[1, 2], default=1)
    p.add_argument("--levels", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--memory-mb", type=int, default=3072)
    a = p.parse_args()
    if a.verify_attempt:
        return verify_worker(a.verify_attempt)
    if not a.output or not 1 <= a.repeats <= 30 or not 0 < a.timeout <= 1800 or not 256 <= a.memory_mb <= 16384:
        p.error("require --output, 1..30 repeats, 0<timeout<=1800, 256..16384 MiB")
    run(a.output, a.repeats, a.order, a.levels, a.timeout, a.memory_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
