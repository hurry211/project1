"""Measure six predeclared adaptive policies on real, independently verified MFEM tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks.adaptive_measure import measured
from sciagent.evidence import artifact_hashes, sha256_file
from sciagent import adaptive, environment, mfem_runner
from sciagent.spec import ProblemSpec, RetryPolicy, ValidationSpec


STRATEGIES = ("fixed_baseline", "expert_rules", "predictive_model", "pilot_autotuner", "advisor_rules", "ai_assisted")


def _prob(cid: str, eq: str, u: str, f: str, kappa: str = "1", neumann: tuple[str, ...] = ()) -> ProblemSpec:
    sides = [(1, "bottom"), (2, "right"), (3, "top"), (4, "left")]
    payload = {
        "id": cid,
        "equation": eq,
        "domain": "unit_square_2d",
        "exact_solution": u,
        "source": f,
        "boundaries": [
            {"attribute": a, "label": lbl, "kind": "neumann" if lbl in neumann else "dirichlet", "value": "from_exact_solution"}
            for a, lbl in sides
        ],
    }
    if eq != "-laplacian(u) = f":
        payload["kappa"] = kappa
    return ProblemSpec.model_validate(payload)


def build_calibration_tasks() -> list[adaptive.AdaptiveTask]:
    pol = RetryPolicy(max_dofs_per_level=90_000, max_total_wall_seconds=120.0, per_level_memory_mb=2048)
    return [
        adaptive.AdaptiveTask(
            task_id="cal_sin11_med", split="calibration", case_id="sin_sin",
            problem=_prob("sin_sin", "-laplacian(u) = f", "sin(pi*x)*sin(pi*y)", "2*pi**2*sin(pi*x)*sin(pi*y)"),
            validation=ValidationSpec(max_l2_error_finest=2e-4), policy=pol,
        ),
        adaptive.AdaptiveTask(
            task_id="cal_sin22_tight", split="calibration", case_id="sin_2x_2y",
            problem=_prob("sin_2x_2y", "-laplacian(u) = f", "sin(2*pi*x)*sin(2*pi*y)", "8*pi**2*sin(2*pi*x)*sin(2*pi*y)"),
            validation=ValidationSpec(max_l2_error_finest=3e-5), policy=pol,
        ),
        adaptive.AdaptiveTask(
            task_id="cal_cosexpy_mix", split="calibration", case_id="cos_exp",
            problem=_prob("cos_exp", "-laplacian(u) = f", "cos(pi*x)*exp(y)", "(pi**2 - 1)*cos(pi*x)*exp(y)", neumann=("top",)),
            validation=ValidationSpec(max_l2_error_finest=2e-5), policy=pol,
        ),
        adaptive.AdaptiveTask(
            task_id="cal_gauss_mid", split="calibration", case_id="gaussian_mid",
            problem=_prob("gaussian_mid", "-laplacian(u) = f", "exp(-40*((x-0.45)**2+(y-0.55)**2))",
                          "(160-6400*((x-0.45)**2+(y-0.55)**2))*exp(-40*((x-0.45)**2+(y-0.55)**2))"),
            validation=ValidationSpec(max_l2_error_finest=5e-5), policy=pol,
        ),
    ]


def build_held_out_tasks() -> list[adaptive.AdaptiveTask]:
    pol = RetryPolicy(max_dofs_per_level=90_000, max_total_wall_seconds=120.0, per_level_memory_mb=2048)
    return [
        adaptive.AdaptiveTask(
            task_id="heldout_sin23_highacc", split="held_out", case_id="sin_2x_3y",
            problem=_prob("sin_2x_3y", "-laplacian(u) = f", "sin(2*pi*x)*sin(3*pi*y)", "13*pi**2*sin(2*pi*x)*sin(3*pi*y)"),
            validation=ValidationSpec(max_l2_error_finest=3e-6), policy=pol,
        ),
        adaptive.AdaptiveTask(
            task_id="heldout_gaussian_steep", split="held_out", case_id="gaussian_steep",
            problem=_prob("gaussian_steep", "-laplacian(u) = f", "exp(-120*((x-0.375)**2+(y-0.625)**2))",
                          "(480-57600*((x-0.375)**2+(y-0.625)**2))*exp(-120*((x-0.375)**2+(y-0.625)**2))"),
            validation=ValidationSpec(max_l2_error_finest=4e-5), policy=pol,
        ),
        adaptive.AdaptiveTask(
            task_id="heldout_var_diff", split="held_out", case_id="sin_sin_variable",
            problem=_prob("sin_sin_variable", "-div(kappa*grad(u)) = f",
                          "sin(pi*x)*sin(2*pi*y)",
                          "5*pi**2*(1+x)*sin(pi*x)*sin(2*pi*y)-pi*cos(pi*x)*sin(2*pi*y)",
                          kappa="1+x", neumann=("top",)),
            validation=ValidationSpec(max_l2_error_finest=8e-6), policy=pol,
        ),
        adaptive.AdaptiveTask(
            task_id="heldout_unstruct_var", split="held_out", case_id="sin_sin_variable",
            problem=_prob("sin_sin_variable", "-div(kappa*grad(u)) = f",
                          "sin(pi*x)*sin(2*pi*y)",
                          "5*pi**2*(1+x)*sin(pi*x)*sin(2*pi*y)-pi*cos(pi*x)*sin(2*pi*y)",
                          kappa="1+x", neumann=("top",)),
            validation=ValidationSpec(max_l2_error_finest=4e-5), policy=pol,
            base_mesh_type="tri", mesh_topology="unstructured", unstructured_nx=8, unstructured_seed=19,
        ),
        adaptive.AdaptiveTask(
            task_id="heldout_tight_mixed", split="held_out", case_id="cos_exp",
            problem=_prob("cos_exp", "-laplacian(u) = f", "cos(pi*x)*exp(y)", "(pi**2 - 1)*cos(pi*x)*exp(y)",
                          neumann=("right", "top")),
            validation=ValidationSpec(max_l2_error_finest=1e-7), policy=pol,
        ),
        adaptive.AdaptiveTask(
            task_id="transfer_unstructured_quad", split="held_out", case_id="sin_sin_variable",
            problem=_prob("sin_sin_variable", "-div(kappa*grad(u)) = f", "sin(pi*x)*sin(2*pi*y)",
                          "5*pi**2*(1+x)*sin(pi*x)*sin(2*pi*y)-pi*cos(pi*x)*sin(2*pi*y)",
                          kappa="1+x", neumann=("top",)),
            validation=ValidationSpec(max_l2_error_finest=4e-5), policy=pol,
            base_mesh_type="quad", mesh_topology="unstructured", unstructured_nx=8, unstructured_seed=27,
        ),
        adaptive.AdaptiveTask(
            task_id="transfer_sine_mixed", split="held_out", case_id="sin_2x_2y",
            problem=_prob("sin_2x_2y", "-laplacian(u) = f", "sin(2*pi*x)*sin(2*pi*y)",
                          "8*pi**2*sin(2*pi*x)*sin(2*pi*y)", neumann=("top",)),
            validation=ValidationSpec(max_l2_error_finest=3e-5), policy=pol,
        ),
    ]


def collect_calibration_observations(work_dir: Path, mms_bin: Path, amr_bin: Path) -> tuple[adaptive.PerformancePredictor, list[dict]]:
    cal_tasks = build_calibration_tasks()
    env = environment.probe()
    records = []
    probe_archetypes = [
        ("quad", "cartesian", 1, "full", "cg_gs", (8, 16, 32)),
        ("quad", "cartesian", 1, "full", "cg_gs", (16, 32, 64)),
        ("quad", "cartesian", 2, "full", "cg_gs", (16, 32, 64)),
        ("quad", "cartesian", 2, "partial", "cg_jacobi", (16, 32, 64)),
        ("quad", "cartesian", 2, "full", "cg_jacobi", (16, 32, 64)),
        ("quad", "cartesian", 2, "full", "cg_none", (8, 16, 32)),
        ("quad", "cartesian", 2, "full", "cg_gs", (8, 16, 32)),
        ("quad", "cartesian", 2, "full", "cg_jacobi", (8, 16, 32)),
        ("quad", "cartesian", 2, "partial", "cg_jacobi", (8, 16, 32)),
        ("tri", "cartesian", 2, "full", "cg_gs", (16, 32, 64)),
        ("quad", "cartesian", 3, "full", "cg_gs", (8, 16, 32)),
        ("quad", "cartesian", 4, "partial", "cg_jacobi", (8, 16, 32)),
        ("quad", "cartesian", 4, "full", "cg_gs", (8, 16, 32)),
        ("quad", "amr", 1, "full", "cg_gs", (8,)),
        ("quad", "amr", 2, "full", "cg_gs", (8,)),
    ]
    for task in cal_tasks:
        hw = adaptive.extract_hardware_features(task.policy, env)
        pf = adaptive.extract_problem_features(task.problem, task.validation, task.mesh_topology, case_id=task.case_id)
        # Run calibration probes without rejecting on tight target so predictor observes true L2 errors
        probe_task = adaptive.AdaptiveTask(
            task_id=task.task_id, split="calibration", case_id=task.case_id,
            problem=task.problem, validation=task.validation.model_copy(update={"max_l2_error_finest": None}),
            policy=task.policy, base_mesh_type=task.base_mesh_type, mesh_topology=task.mesh_topology,
        )
        for mt, topo, p, asm, sol, sched in probe_archetypes:
            if topo == "amr" and task.case_id not in ("gaussian_mid", "sin_2x_2y"):
                continue
            cand = adaptive.StrategyCandidate(
                candidate_id=f"{topo}-{mt}-p{p}-{asm}-{sol}-n{sched[-1]}",
                mesh_type=mt, mesh_topology=topo, polynomial_order=p,
                assembly=asm, solver=sol, mesh_levels=sched,
                estimated_finest_dofs=(4000 if topo == "amr" else (sched[-1] * p + 1) ** 2),
                amr_max_steps=10, amr_max_dofs=4000,
            )
            dest = work_dir / task.task_id / cand.candidate_id
            res = measured(dest.with_suffix(".json"), task.policy, adaptive.execute_candidate,
                           probe_task, cand, dest, mms_bin=mms_bin, amr_bin=amr_bin)
            res["wall_seconds"] = res["external_wall_seconds"]
            records.append({"pf": pf, "hw": hw, "cand": cand, **res})
            print(json.dumps({"stage": "calibrate", "task": task.task_id, "cand": cand.candidate_id,
                              "ok": res["ok"], "l2": res["l2_error"], "s": res["wall_seconds"]}), flush=True)
    predictor = adaptive.PerformancePredictor.fit(records)
    return predictor, [
        {k: v for k, v in r.items() if k not in ("pf", "hw", "cand")} | {"task_id": r["pf"].case_id, "cand_id": r["cand"].candidate_id}
        for r in records
    ]


def _bootstrap_ci(values: list[float], n_boot: int = 4096, seed: int = 123) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    if arr.size < 2:
        return med, med, med
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.median(arr[idx], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(med, 4), round(float(lo), 4), round(float(hi), 4)


def paired_results(trials):
    indexed = {(t["task_id"], t["repeat"], t["strategy"]): t for t in trials}
    if len(indexed) != len(trials):
        raise ValueError("duplicate task/repeat/strategy")
    pairs = {}
    for strategy in ("expert_rules", "predictive_model", "pilot_autotuner", "advisor_rules", "ai_assisted"):
        for base in ("fixed_baseline", "advisor_rules") if strategy == "ai_assisted" else ("fixed_baseline",):
            ratios = {}
            excluded = []
            for (task, rep, arm), row in indexed.items():
                if arm != strategy:
                    continue
                control = indexed[task, rep, base]
                if row["target_l2"] != control["target_l2"]:
                    raise ValueError("paired accuracy targets differ")
                if row["accepted"] and control["accepted"]:
                    ratios.setdefault(task, []).append(control["total_end_to_end_seconds"] / row["total_end_to_end_seconds"])
                else:
                    excluded.append({"task": task, "repeat": rep})
            # Resample independent problem clusters, keeping each task's paired repeats together.
            per_task = [float(np.median(values)) for values in ratios.values()]
            stats = _bootstrap_ci(per_task) if per_task else (None, None, None)
            pairs[f"{strategy}_vs_{base}"] = {"matched_accepted": sum(map(len, ratios.values())), "matched_tasks": len(ratios), "excluded": excluded,
                "median_ratio": stats[0], "task_cluster_ci95": list(stats[1:])}
    return pairs


def run_adaptive_benchmark(
    repeats: int = 3,
    out_dir: Path = ROOT / "runs/adaptive-report",
    mms_bin: Path | None = None,
    amr_bin: Path | None = None,
    *, work_dir: Path | None = None, ollama_model=None, ollama_digest=None,
) -> dict:
    if repeats < 1 or bool(ollama_model) != bool(ollama_digest):
        raise ValueError("supply at least one repeat and a complete model pin")
    mms_bin = (mms_bin or mfem_runner.solver_binary()).resolve(strict=True)
    amr_bin = (amr_bin or mfem_runner.amr_binary()).resolve(strict=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "adaptive_comparison.json").exists():
        raise FileExistsError(out_dir / "adaptive_comparison.json")
    work = (work_dir or out_dir / "raw").resolve()
    work.mkdir(parents=True, exist_ok=False)
    design = {"schema": "adaptive_v2", "seed": 2026, "repeats": repeats,
              "strategies": STRATEGIES, "model": ollama_model, "model_digest": ollama_digest,
              "serial_sha256": sha256_file(mms_bin), "amr_sha256": sha256_file(amr_bin),
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "sources_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p)
                  for folder in ("benchmarks", "sciagent", "verifier", "cpp")
                  for p in sorted((ROOT / folder).glob("*")) if p.suffix in (".py", ".cpp", ".hpp")},
              "tasks": [t.task_id for t in build_calibration_tasks()+build_held_out_tasks()]}
    (work / "design.json").write_text(json.dumps(design, indent=2))
    predictor, cal_summary = collect_calibration_observations(work / "cal", mms_bin, amr_bin)
    held_out = build_held_out_tasks()
    rng = np.random.default_rng(2026)
    trials = []
    for rep in range(repeats):
        for task in held_out:
            order = list(STRATEGIES)
            rng.shuffle(order)
            for pos, strat in enumerate(order):
                dest = work / f"rep{rep}_{task.task_id}_{strat}"
                res = measured(dest.with_suffix(".json"), task.policy, adaptive.run_strategy_on_task,
                               strat, task, dest, predictor=predictor, mms_bin=mms_bin, amr_bin=amr_bin,
                               ollama_model=ollama_model, ollama_digest=ollama_digest)
                res.update(task_id=task.task_id, strategy=strat, target_l2=task.validation.max_l2_error_finest,
                           repeat=rep, order_position=pos, directory=dest.name,
                           total_end_to_end_seconds=res["external_wall_seconds"])
                trials.append(res)
                print(rep, task.task_id, strat, res["accepted"], round(res["external_wall_seconds"], 3), flush=True)
    # Aggregate statistics cleanly per strategy across all randomized held-out trials
    strategy_summary = {}
    campaign_totals_by_rep = {s: [] for s in STRATEGIES}
    for s in STRATEGIES:
        rows = [t for t in trials if t["strategy"] == s]
        times = [t["total_end_to_end_seconds"] for t in rows]
        pilots = [t["pilot_seconds"] for t in rows]
        decs = [t["decision_seconds"] for t in rows]
        retry_waste = [
            sum(a["wall_seconds"] for a in t["attempts"] if not a["ok"]) if t["attempts"] else t["external_wall_seconds"]
            for t in rows
        ]
        rss = [t["peak_rss_mb"] for t in rows]
        for rep in range(repeats):
            campaign_totals_by_rep[s].append(sum(t["total_end_to_end_seconds"] for t in rows if t["repeat"] == rep))
        med, lo, hi = _bootstrap_ci(times)
        camp_med, camp_lo, camp_hi = _bootstrap_ci(campaign_totals_by_rep[s])
        strategy_summary[s] = {
            "trials": len(rows),
            "accepted_count": sum(1 for t in rows if t["accepted"]),
            "acceptance_rate": round(sum(1 for t in rows if t["accepted"]) / max(len(rows), 1), 4),
            "median_task_seconds": med,
            "median_task_ci95": [lo, hi],
            "mean_task_seconds": round(float(np.mean(times)), 4),
            "campaign_total_median_seconds": camp_med,
            "campaign_total_ci95": [camp_lo, camp_hi],
            "total_pilot_seconds": round(sum(pilots), 4),
            "total_decision_seconds": round(sum(decs), 4),
            "total_failed_retry_seconds": round(sum(retry_waste), 4),
            "median_peak_rss_mb": round(float(np.median(rss)), 2),
            "max_peak_rss_mb": round(float(max(rss)), 2),
        }

    paired_speedups = paired_results(trials)
    inventory = artifact_hashes(work)
    (work / "inventory.json").write_text(json.dumps(inventory, sort_keys=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "adaptive_v2", "design": design, "design_sha256": sha256_file(work / "design.json"),
        "inventory_sha256": sha256_file(work / "inventory.json"), "raw_files": len(inventory),
        "memory_definition": "Maximum of Python worker and child high-water marks and 10-ms sampled sum of simultaneous RSS; shared pages may be counted twice. The external Ollama daemon is excluded.",
        "repeats": repeats,
        "calibration_task_ids": [t.task_id for t in build_calibration_tasks()],
        "held_out_task_ids": [t.task_id for t in build_held_out_tasks()],
        "predictor": predictor.to_dict(),
        "calibration_observations": cal_summary,
        "strategy_summary": strategy_summary,
        "paired_speedups": paired_speedups,
        "trials": trials,
    }
    (out_dir / "adaptive_comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/adaptive-report")
    parser.add_argument("--mms-bin", type=Path, default=None)
    parser.add_argument("--amr-bin", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--ollama-model")
    parser.add_argument("--ollama-digest")
    args = parser.parse_args(argv)
    run_adaptive_benchmark(args.repeats, args.out_dir, args.mms_bin, args.amr_bin,
                           work_dir=args.work_dir, ollama_model=args.ollama_model, ollama_digest=args.ollama_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
