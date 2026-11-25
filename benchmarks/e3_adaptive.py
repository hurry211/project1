"""E3 Task 3 & Task 4: Separated calibration/held-out adaptive strategy benchmark.

Runs randomized paired experiments comparing 5 strategies on disjoint held-out tasks:
  1. fixed_baseline     (E2 Q1 full CG+GS with retry escalation)
  2. expert_rules       (domain heuristic rules)
  3. predictive_model   (Ridge log-linear surrogate on problem+hardware features)
  4. pilot_autotuner    (real coarse probes + asymptotic extrapolation, pilots charged)
  5. ai_assisted        (local Ollama hook if active, else guardrail-checked structured advisor)

Publishes raw paired trial archives, median + 95% bootstrap CIs, peak RSS,
failure/pilot overhead accounting, and SVG charts under evidence/e3/.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks.e3_plots import render_bar_comparison
from sciagent import adaptive, environment, mfem_runner
from sciagent.spec import ProblemSpec, RetryPolicy, ValidationSpec


STRATEGIES = ("fixed_baseline", "expert_rules", "predictive_model", "pilot_autotuner", "ai_assisted")


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
            res = adaptive.execute_candidate(
                probe_task, cand, work_dir / task.task_id / cand.candidate_id, mms_bin=mms_bin, amr_bin=amr_bin
            )
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


def run_adaptive_benchmark(
    repeats: int = 3,
    out_dir: Path = ROOT / "evidence/e3",
    mms_bin: Path = ROOT / "build/cpp/poisson_mms",
    amr_bin: Path = ROOT / "build/cpp/poisson_amr",
) -> dict:
    work = Path(tempfile.mkdtemp(prefix="e3_adaptive_"))
    try:
        predictor, cal_summary = collect_calibration_observations(work / "cal", mms_bin, amr_bin)
        held_out = build_held_out_tasks()
        rng = np.random.default_rng(2026)

        trials = []
        for rep in range(repeats):
            for task in held_out:
                order = list(STRATEGIES)
                rng.shuffle(order)
                for strat in order:
                    res = adaptive.run_strategy_on_task(
                        strat, task, work / f"rep{rep}_{task.task_id}_{strat}",
                        predictor=predictor, mms_bin=mms_bin, amr_bin=amr_bin,
                    )
                    res["repeat"] = rep
                    res["order_position"] = order.index(strat)
                    trials.append(res)
                    print(json.dumps({
                        "stage": "trial", "rep": rep, "task": task.task_id, "strat": strat,
                        "ok": res["accepted"], "init": res["selected_initial_candidate"],
                        "total_s": res["total_end_to_end_seconds"], "pilot_s": res["pilot_seconds"],
                        "rss_mb": res["peak_rss_mb"], "l2": res["finest_l2_error"],
                    }), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # Aggregate statistics cleanly per strategy across all randomized held-out trials
    strategy_summary = {}
    campaign_totals_by_rep = {s: [] for s in STRATEGIES}
    for s in STRATEGIES:
        rows = [t for t in trials if t["strategy"] == s]
        times = [t["total_end_to_end_seconds"] for t in rows]
        pilots = [t["pilot_seconds"] for t in rows]
        decs = [t["decision_seconds"] for t in rows]
        retry_waste = [
            sum(a["wall_seconds"] for a in t["attempts"][:-1]) + (0.0 if t["accepted"] else t["attempts"][-1]["wall_seconds"])
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

    # Pairwise paired speedup of predictive_model & ai_assisted over fixed_baseline & pilot_autotuner
    paired_speedups = {}
    for target_s in ("predictive_model", "ai_assisted"):
        for base_s in ("fixed_baseline", "pilot_autotuner"):
            ratios = []
            for rep in range(repeats):
                for task in held_out:
                    t_base = next(t for t in trials if t["repeat"] == rep and t["task_id"] == task.task_id and t["strategy"] == base_s)
                    t_targ = next(t for t in trials if t["repeat"] == rep and t["task_id"] == task.task_id and t["strategy"] == target_s)
                    ratios.append(t_base["total_end_to_end_seconds"] / max(t_targ["total_end_to_end_seconds"], 1e-6))
            med_r, lo_r, hi_r = _bootstrap_ci(ratios)
            paired_speedups[f"{target_s}_vs_{base_s}"] = {"median_ratio": med_r, "ci95": [lo_r, hi_r]}

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "e3_adaptive_v1",
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

    bars = [
        {
            "label": s.replace("_", " ").title(),
            "value": strategy_summary[s]["campaign_total_median_seconds"],
            "lo": strategy_summary[s]["campaign_total_ci95"][0],
            "hi": strategy_summary[s]["campaign_total_ci95"][1],
            "sublabel": f"acc={strategy_summary[s]['accepted_count']}/{strategy_summary[s]['trials']}",
        }
        for s in STRATEGIES
    ]
    render_bar_comparison(
        out_dir / "fig_adaptive_campaign_cost.svg",
        title="Held-Out Campaign End-to-End Trusted Cost (Median & 95% Bootstrap CI)",
        ylabel="Campaign Wall Time across 5 Held-Out Tasks (s)",
        bars=bars,
    )

    per_task_bars = []
    for task in build_held_out_tasks():
        for strat, short in (("fixed_baseline", "Fix"), ("expert_rules", "Exp"),
                             ("predictive_model", "Pred"), ("pilot_autotuner", "Tune"), ("ai_assisted", "AI")):
            vals = [t["total_end_to_end_seconds"] for t in trials if t["task_id"] == task.task_id and t["strategy"] == strat]
            per_task_bars.append({
                "label": f"{task.task_id.replace('heldout_', '')}:{short}",
                "value": statistics.median(vals),
            })
    render_bar_comparison(
        out_dir / "fig_adaptive_per_task.svg",
        title="Per-Task Median Trusted Solve Time on Disjoint Held-Out Corpus",
        ylabel="End-to-End Seconds (incl. decision/pilot/retry)",
        bars=per_task_bars,
        width=1120,
        height=440,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evidence/e3")
    parser.add_argument("--mms-bin", type=Path, default=ROOT / "build/cpp/poisson_mms")
    parser.add_argument("--amr-bin", type=Path, default=ROOT / "build/cpp/poisson_amr")
    args = parser.parse_args(argv)
    run_adaptive_benchmark(args.repeats, args.out_dir, args.mms_bin, args.amr_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
