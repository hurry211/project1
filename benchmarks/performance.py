"""Repeated full/partial CG+Jacobi and retry ablation with independent verification."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from sciagent.controller import run_study
from sciagent.environment import PROJECT_ROOT, probe
from sciagent.evidence import sha256_file
from sciagent.spec import load_study

NATIVE_TIMINGS = ("time_rhs_ms", "time_assembly_ms", "time_preconditioner_ms", "time_solve_ms",
                  "time_error_ms", "time_export_ms")


def trial(spec: Path, output: Path, reuse: bool) -> dict:
    started = time.perf_counter()
    manifest = run_study(spec, output / "runs", reuse_levels=reuse)
    wall = time.perf_counter() - started
    run = Path(manifest["run_dir"])
    native = []
    verification = []
    reused = 0
    for attempt in manifest["attempts"]:
        if "verification" in attempt:
            verification.append(attempt["verification"])
        for level in attempt["levels"]:
            if "reused_from" in level:
                reused += 1
                continue
            path = run / "attempts" / f"attempt_{attempt['attempt']:02d}" / "levels" / f"n{level['n']:04d}"
            if (path / "result.json").is_file():
                result = json.loads((path / "result.json").read_text())
                native.append({"attempt": attempt["attempt"], "n": level["n"],
                               **{key: result[key] for key in (*NATIVE_TIMINGS, "peak_rss_kib", "num_dofs",
                                                               "solver_iterations")}})
    attempts = manifest["attempts"]
    finest = attempts[-1].get("errors") if attempts else None
    metrics = {"wall_seconds": wall, "verification_seconds": sum(v["wall_seconds"] for v in verification),
               "native_peak_rss_kib": max((x["peak_rss_kib"] for x in native
                                           if x["peak_rss_kib"] is not None), default=None),
               "verifier_peak_rss_kib": max((v["peak_rss_kib"] for v in verification
                                            if v["peak_rss_kib"] is not None), default=None),
               "native_levels": len(native), "reused_levels": reused,
               **{key: sum(x[key] for x in native) for key in NATIVE_TIMINGS}}
    if finest:
        metrics.update(l2=finest[-1]["l2"], h1_semi=finest[-1]["h1_semi"],
                       reference_rel_diff=finest[-1]["reference_rel_diff"])
    if native:
        metrics.update(finest_iterations=native[-1]["solver_iterations"],
                       finest_dofs=native[-1]["num_dofs"])
    return {"state": manifest["final_state"], "code": manifest["final_code"],
            "archive": run.relative_to(output).as_posix(),
            "manifest_sha256": sha256_file(run / "manifest.json"),
            "frozen_sha256": manifest.get("frozen_sha256"),
            "attempt_decisions": [a["verdict"] for a in attempts],
            "attempt_errors": [a.get("errors", []) for a in attempts],
            "metrics": metrics, "native": native}


def describe(values):
    median = statistics.median(values)
    return {"n": len(values), "median": median, "min": min(values), "max": max(values),
            "mad": statistics.median(abs(v - median) for v in values)}


def summarize(trials):
    rows = [row["metrics"] for row in trials if row["state"] == "ACCEPTED"]
    summary = {}
    for key in sorted({k for row in rows for k in row}):
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            summary[key] = describe(values)
    return summary


def equivalent(a, b):
    if (a["state"] != "ACCEPTED" or b["state"] != "ACCEPTED"
            or a["attempt_decisions"] != b["attempt_decisions"]
            or a["frozen_sha256"] != b["frozen_sha256"]
            or a["metrics"]["finest_dofs"] != b["metrics"]["finest_dofs"]):
        return False
    for left, right in zip(a["attempt_errors"], b["attempt_errors"]):
        if [x["n"] for x in left] != [y["n"] for y in right]:
            return False
        if any(not math.isclose(x[key], y[key], rel_tol=1e-7)
               for x, y in zip(left, right) for key in ("l2", "h1_semi")):
            return False
    return True


def compare(arms, output, repeats):
    warmup = {}
    samples = {label: [] for label, _, _ in arms}
    paired = []
    for index in range(-1, repeats):
        order = arms if index % 2 == 0 else list(reversed(arms))
        current = {label: trial(spec, output, reuse) for label, spec, reuse in order}
        if index < 0:
            warmup = current
        else:
            for label, row in current.items():
                samples[label].append(row)
            a, b = (current[label] for label, _, _ in arms)
            passed = equivalent(a, b)
            paired.append({"equivalent": passed, "wall_ratio_first_over_second":
                           a["metrics"]["wall_seconds"] / b["metrics"]["wall_seconds"] if passed else None})
    ratios = [row["wall_ratio_first_over_second"] for row in paired if row["equivalent"]]
    return {"warmup": warmup, "samples": samples, "paired": paired,
            "paired_wall_ratio": describe(ratios) if ratios else None,
            "statistics": {label: summarize(rows) for label, rows in samples.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new experiment directory")
    parser.add_argument("--repeats", type=int, default=5)
    loads = parser.add_mutually_exclusive_group()
    loads.add_argument("--mixed-boundary", action="store_true",
                        help="compare only the mixed-boundary Poisson case on meshes 16, 32, 64")
    loads.add_argument("--gaussian", action="store_true",
                       help="compare Q2 localized Gaussian loads at finest n=128 and n=256")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two measured repetitions are required")
    args.output.mkdir(parents=True, exist_ok=False)
    specs = PROJECT_ROOT / "specs"
    base_name = "poisson_cos_exp_mixed_p1_quad.json" if args.mixed_boundary else "poisson_sin_sin_p1_tri.json"
    if args.gaussian:
        base_name = "experiments/gaussian_offset_q2.json"
    base = load_study(specs / base_name)
    level_sets = ((16, 32, 64),) if args.mixed_boundary else ((8, 16, 32), (16, 32, 64))
    if args.gaussian:
        level_sets = ((32, 64, 128), (64, 128, 256))
    report = {"environment": probe(), "measured_repeats": args.repeats, "base_spec": base_name, "groups": {}}
    for order in ((2,) if args.gaussian else (1, 2)):
        for levels in level_sets:
            group = f"q{order}-n{levels[-1]}"
            arms = []
            for assembly in ("full", "partial"):
                execution = base.execution.model_copy(update={
                    "mesh_type": "quad", "polynomial_order": order, "mesh_levels": levels,
                    "solver": "cg_jacobi", "assembly": assembly})
                spec = args.output / f"{group}-{assembly}.json"
                spec.write_text(base.model_copy(update={"execution": execution}).model_dump_json(indent=2))
                arms.append((assembly, spec, True))
            report["groups"][group] = compare(arms, args.output, args.repeats)
    if not args.mixed_boundary and not args.gaussian:
        retry = specs / "retry_demo_sin_sin_p1_tri.json"
        report["groups"]["retry"] = compare([("recompute", retry, False), ("reuse", retry, True)],
                                           args.output, args.repeats)
    path = args.output / "performance.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    passed = all(pair["equivalent"] for g in report["groups"].values() for pair in g["paired"])
    print(json.dumps({"passed": passed, "report": str(path)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
