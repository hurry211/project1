"""E3.1 Task 4: Separated calibration/held-out 5-way risk-aware adaptive benchmark.

Enforces:
- Strict split hygiene: calibration (6 tasks) and held-out (6 tasks) use completely
  disjoint problem IDs and exact-solution formulas, with zero interior exact-solution
  feature leakage.
- Strong baselines (`fixed_baseline` Q2 full CG+GS, `expert_rules`, `predictive_model`,
  `pilot_autotuner`) vs `hybrid_guardrail_predictor` (zero fake LLM arm).
- Full resource & risk accounting across >= 5 randomized paired repeats: completion rates,
  separate failure cost, solver/verifier/decision/pilot/retry cost breakdowns, peak RSS,
  and mutually-accepted paired speedups with bootstrap 95% CIs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks.e3_plots import render_bar_comparison
from sciagent import adaptive, environment
from sciagent.spec import ProblemSpec, RetryPolicy, ValidationSpec


def _bc_all_dirichlet(domain: str = "unit_square_2d") -> list[dict]:
    if domain == "lshape_2d":
        return [{"attribute": 1, "label": "lshape_outer", "kind": "dirichlet", "value": "from_exact_solution"}]
    return [
        {"attribute": a, "label": lbl, "kind": "dirichlet", "value": "from_exact_solution"}
        for a, lbl in ((1, "bottom"), (2, "right"), (3, "top"), (4, "left"))
    ]


def _bc_mixed(neumann: tuple[str, ...] = ("top",)) -> list[dict]:
    return [
        {"attribute": a, "label": lbl, "kind": "neumann" if lbl in neumann else "dirichlet", "value": "from_exact_solution"}
        for a, lbl in ((1, "bottom"), (2, "right"), (3, "top"), (4, "left"))
    ]


def _load_lshape_bytes() -> bytes:
    for p in (
        ROOT / "third_party/mfem-src/data/l-shape.mesh",
        Path("/home/fy/project1-upstream-6b5761c/third_party/mfem-src/data/l-shape.mesh"),
    ):
        if p.is_file():
            return p.read_bytes()
    raise FileNotFoundError("cannot locate third_party/mfem-src/data/l-shape.mesh")


def build_calibration_tasks() -> list[adaptive.AdaptiveTask]:
    """Six calibration tasks cleanly disjoint from all held-out problem IDs and formulas."""
    pol = RetryPolicy(max_attempts=3, max_dofs_per_level=60_000, max_total_wall_seconds=300.0, per_level_memory_mb=2048)
    return [
        adaptive.AdaptiveTask(
            name="cal_sin_sin",
            case_id="sin_sin",
            problem=ProblemSpec.model_validate({
                "id": "sin_sin", "equation": "-laplacian(u) = f", "domain": "unit_square_2d",
                "exact_solution": "sin(pi*x)*sin(pi*y)",
                "source": "2*pi**2*sin(pi*x)*sin(pi*y)",
                "boundaries": _bc_all_dirichlet(),
            }),
            validation=ValidationSpec(max_l2_error_finest=5e-4),
            policy=pol,
        ),
        adaptive.AdaptiveTask(
            name="cal_cos_exp_mixed",
            case_id="cos_exp",
            problem=ProblemSpec.model_validate({
                "id": "cos_exp", "equation": "-laplacian(u) = f", "domain": "unit_square_2d",
                "exact_solution": "cos(pi*x)*exp(y)",
                "source": "(pi**2 - 1)*cos(pi*x)*exp(y)",
                "boundaries": _bc_mixed(("right", "top")),
            }),
            validation=ValidationSpec(max_l2_error_finest=5e-4),
            policy=pol,
        ),
        adaptive.AdaptiveTask(
            name="cal_sin_2x_2y",
            case_id="sin_2x_2y",
            problem=ProblemSpec.model_validate({
                "id": "sin_2x_2y", "equation": "-laplacian(u) = f", "domain": "unit_square_2d",
                "exact_solution": "sin(2*pi*x)*sin(2*pi*y)",
                "source": "8*pi**2*sin(2*pi*x)*sin(2*pi*y)",
                "boundaries": _bc_all_dirichlet(),
            }),
            validation=ValidationSpec(max_l2_error_finest=3e-4),
            policy=pol,
        ),
        adaptive.AdaptiveTask(
            name="cal_var_kappa_mixed",
            case_id="sin_sin_variable",
            problem=ProblemSpec.model_validate({
                "id": "sin_sin_variable", "equation": "-div(kappa*grad(u)) = f", "kappa": "1+x",
                "domain": "unit_square_2d",
                "exact_solution": "sin(pi*x)*sin(2*pi*y)",
                "source": "5*pi**2*(1+x)*sin(pi*x)*sin(2*pi*y)-pi*cos(pi*x)*sin(2*pi*y)",
                "boundaries": _bc_mixed(("top",)),
            }),
            validation=ValidationSpec(max_l2_error_finest=3e-4),
            policy=pol,
        ),
        adaptive.AdaptiveTask(
            name="cal_gaussian_mid",
            case_id="gaussian_mid",
            problem=ProblemSpec.model_validate({
                "id": "gaussian_mid", "equation": "-laplacian(u) = f", "domain": "unit_square_2d",
                "exact_solution": "exp(-40*((x-0.45)**2+(y-0.55)**2))",
                "source": "(160-6400*((x-0.45)**2+(y-0.55)**2))*exp(-40*((x-0.45)**2+(y-0.55)**2))",
                "boundaries": _bc_all_dirichlet(),
            }),
            validation=ValidationSpec(max_l2_error_finest=2e-3),
            policy=pol,
        ),
        adaptive.AdaptiveTask(
            name="cal_sin_cos",
            case_id="sin_cos",
            problem=ProblemSpec.model_validate({
                "id": "sin_cos", "equation": "-laplacian(u) = f", "domain": "unit_square_2d",
                "exact_solution": "sin(pi*x)*cos(pi*y)",
                "source": "2*pi**2*sin(pi*x)*cos(pi*y)",
                "boundaries": _bc_mixed(("top", "bottom")),
            }),
            validation=ValidationSpec(max_l2_error_finest=4e-4),
            policy=pol,
        ),
    ]


def build_held_out_tasks() -> list[adaptive.AdaptiveTask]:
    """Six completely unseen held-out tasks across smooth, variable-kappa, sharp/steep, unstructured, and L-shape."""
    pol = RetryPolicy(max_attempts=3, max_dofs_per_level=60_000, max_total_wall_seconds=300.0, per_level_memory_mb=2048)
    lshape_bytes = _load_lshape_bytes()
    return [
        adaptive.AdaptiveTask(
            name="held_sin_3x_2y_tight",
            case_id="sin_3x_2y",
            problem=ProblemSpec.model_validate({
                "id": "sin_3x_2y", "equation": "-laplacian(u) = f", "domain": "unit_square_2d",
                "exact_solution": "sin(3*pi*x)*sin(2*pi*y)",
                "source": "13*pi**2*sin(3*pi*x)*sin(2*pi*y)",
                "boundaries": _bc_all_dirichlet(),
            }),
            validation=ValidationSpec(max_l2_error_finest=1e-4),
            policy=pol,
        ),
        adaptive.AdaptiveTask(
            name="held_cos_2x_sin_y_var_mixed",
            case_id="cos_2x_sin_y_var",
            problem=ProblemSpec.model_validate({
                "id": "cos_2x_sin_y_var", "equation": "-div(kappa*grad(u)) = f", "kappa": "1+x",
                "domain": "unit_square_2d",
                "exact_solution": "cos(2*pi*x)*sin(pi*y)",
                "source": "5*pi**2*(1+x)*cos(2*pi*x)*sin(pi*y)+2*pi*sin(2*pi*x)*sin(pi*y)",
                "boundaries": _bc_mixed(("top",)),
            }),
            validation=ValidationSpec(max_l2_error_finest=2e-4),
            policy=pol,
        ),
        adaptive.AdaptiveTask(
            name="held_gaussian_sharp",
            case_id="gaussian_sharp",
            problem=ProblemSpec.model_validate({
                "id": "gaussian_sharp", "equation": "-laplacian(u) = f", "domain": "unit_square_2d",
                "exact_solution": "exp(-80*((x-0.625)**2+(y-0.375)**2))",
                "source": "(320-25600*((x-0.625)**2+(y-0.375)**2))*exp(-80*((x-0.625)**2+(y-0.375)**2))",
                "boundaries": _bc_all_dirichlet(),
            }),
            validation=ValidationSpec(max_l2_error_finest=1e-4),
            policy=pol,
        ),
        adaptive.AdaptiveTask(
            name="held_gaussian_steep_amr",
            case_id="gaussian_steep",
            problem=ProblemSpec.model_validate({
                "id": "gaussian_steep", "equation": "-laplacian(u) = f", "domain": "unit_square_2d",
                "exact_solution": "exp(-120*((x-0.375)**2+(y-0.625)**2))",
                "source": "(480-57600*((x-0.375)**2+(y-0.625)**2))*exp(-120*((x-0.375)**2+(y-0.625)**2))",
                "boundaries": _bc_all_dirichlet(),
            }),
            validation=ValidationSpec(max_l2_error_finest=5e-4),
            policy=pol,
        ),
        adaptive.AdaptiveTask(
            name="held_lshape_singularity",
            case_id="lshape_singularity",
            problem=ProblemSpec.model_validate({
                "id": "lshape_singularity", "equation": "-laplacian(u) = f", "domain": "lshape_2d",
                "exact_solution": "((x-0.5)**2+(y-0.5)**2)**(1/3)*sin((2/3)*theta_lshape)",
                "source": "0",
                "boundaries": _bc_all_dirichlet("lshape_2d"),
            }),
            validation=ValidationSpec(max_l2_error_finest=2e-4),
            policy=pol,
            mesh_topology="amr",
            lshape_mesh_bytes=lshape_bytes,
        ),
        adaptive.AdaptiveTask(
            name="held_unstructured_sin_2x_3y",
            case_id="sin_2x_3y",
            problem=ProblemSpec.model_validate({
                "id": "sin_2x_3y", "equation": "-laplacian(u) = f", "domain": "unit_square_2d",
                "exact_solution": "sin(2*pi*x)*sin(3*pi*y)",
                "source": "13*pi**2*sin(2*pi*x)*sin(3*pi*y)",
                "boundaries": _bc_mixed(("top",)),
            }),
            validation=ValidationSpec(max_l2_error_finest=5e-5),
            policy=pol,
            mesh_topology="unstructured",
            unstructured_seed=29,
        ),
    ]


def collect_calibration_observations(
    work_dir: Path, mms_bin: Path, amr_bin: Path
) -> tuple[adaptive.PerformancePredictor, list[dict]]:
    """Observe real MFEM+verifier trajectories across the 6 calibration tasks only."""
    env = environment.probe(mms_bin)
    cal_tasks = build_calibration_tasks()
    records = []
    for task in cal_tasks:
        hw = adaptive.extract_hardware_features(task.policy, env)
        pf = adaptive.extract_problem_features(task.problem, task.validation, task.mesh_topology, case_id=task.case_id)
        if task.case_id == "gaussian_mid":
            probes = [
                ("quad", "cartesian", 2, 8, 8, 20_000),
                ("quad", "cartesian", 3, 4, 8, 20_000),
                ("quad", "cartesian", 3, 8, 8, 20_000),
                ("quad", "cartesian", 4, 8, 8, 20_000),
                ("quad", "amr", 2, 4, 8, 12_000),
                ("quad", "amr", 3, 4, 8, 12_000),
            ]
        else:
            probes = [
                ("tri", "cartesian", 1, 8, 8, 20_000),
                ("quad", "cartesian", 2, 8, 8, 20_000),
                ("quad", "cartesian", 3, 4, 8, 20_000),
                ("quad", "cartesian", 3, 8, 8, 20_000),
                ("tri", "cartesian", 3, 8, 8, 20_000),
                ("quad", "cartesian", 4, 8, 8, 20_000),
                ("quad", "amr", 2, 4, 6, 8_000),
                ("quad", "amr", 3, 4, 6, 8_000),
            ]
        for mt, top, p, n0, steps, mdofs in probes:
            cand = adaptive.StrategyCandidate(mt, top, p, "full", "cg_gs", n0, amr_max_steps=steps, amr_max_dofs=mdofs)
            res = adaptive.execute_candidate(task, cand, work_dir, mms_bin, amr_bin, tag=f"cal_{task.name}")
            records.append({
                "task": task.name,
                "case_id": task.case_id,
                "target_l2": task.validation.max_l2_error_finest,
                "candidate": vars(cand),
                "problem_features": vars(pf),
                "hw_features": vars(hw),
                "verdict": res["verdict"],
                "l2_error": res["l2_error"],
                "wall_seconds": res["wall_seconds"],
                "solver_seconds": res["solver_seconds"],
                "verifier_seconds": res["verifier_seconds"],
                "peak_rss_mb": res["peak_rss_mb"],
            })
            print(json.dumps({"stage": "calibration", "task": task.name, "cand": cand.tag(),
                              "verdict": res["verdict"], "l2": res["l2_error"], "wall_s": res["wall_seconds"]}), flush=True)

    predictor = adaptive.PerformancePredictor.fit(records)
    return predictor, records


def _bootstrap_ci95(values: list[float], *, seed: int = 20261125, n_boot: int = 2000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    meds = sorted(statistics.median(rng.choices(values, k=n)) for _ in range(n_boot))
    return (meds[int(0.025 * (n_boot - 1))], meds[int(0.975 * (n_boot - 1))])


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    s = sorted(values)
    return float(np.percentile(s, 75) - np.percentile(s, 25))


def run_adaptive_benchmark(
    repeats: int = 5,
    out_dir: Path = ROOT / "evidence/e3",
    mms_bin: Path = ROOT / "build/cpp/poisson_mms",
    amr_bin: Path = ROOT / "build/cpp/poisson_amr",
) -> dict:
    cal_tasks = build_calibration_tasks()
    held_tasks = build_held_out_tasks()
    assert not (set(t.case_id for t in cal_tasks) & set(t.case_id for t in held_tasks)), "Split hygiene violated!"
    assert not (set(t.problem.exact_solution for t in cal_tasks) & set(t.problem.exact_solution for t in held_tasks)), "Formula leakage!"

    work = Path(tempfile.mkdtemp(prefix="e3_adaptive_"))
    try:
        predictor, cal_records = collect_calibration_observations(work / "cal", mms_bin, amr_bin)
        env = environment.probe(mms_bin)
        hw = adaptive.extract_hardware_features(held_tasks[0].policy, env)

        runs = []
        rng = random.Random(20261125)
        for rep in range(repeats):
            for task in held_tasks:
                order = list(adaptive.STRATEGY_NAMES)
                rng.shuffle(order)
                for order_pos, strat in enumerate(order):
                    tdir = work / f"r{rep}_{task.name}_{strat}"
                    res = adaptive.run_strategy_on_task(
                        strat, task, predictor, hw, env, tdir, mms_bin, amr_bin,
                    )
                    res["repeat"] = rep
                    res["order_in_block"] = order_pos
                    runs.append(res)
                    print(json.dumps({
                        "stage": "held_out", "rep": rep, "task": task.name, "strategy": strat,
                        "chosen": res["chosen_final"], "accepted": res["accepted"],
                        "p_fail": res["predicted_failure_prob"], "wall_s": res["end_to_end_seconds"],
                    }), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # Aggregate strictly separating completion rate, accepted cost, and failed cost
    by_key = {(r["repeat"], r["task"], r["strategy"]): r for r in runs}
    strategy_summary = {}
    for strat in adaptive.STRATEGY_NAMES:
        s_runs = [r for r in runs if r["strategy"] == strat]
        acc_runs = [r for r in s_runs if r["accepted"]]
        fail_runs = [r for r in s_runs if not r["accepted"]]

        acc_walls = [r["end_to_end_seconds"] for r in acc_runs]
        all_walls = [r["end_to_end_seconds"] for r in s_runs]
        rss_vals = [r["peak_rss_mb"] for r in s_runs if r["peak_rss_mb"] > 0]

        # Campaign total per repeat over all 6 held-out tasks
        camp_all = [
            sum(r["end_to_end_seconds"] for r in s_runs if r["repeat"] == rep)
            for rep in range(repeats)
        ]
        # Strictly mutually-accepted paired speedups (ONLY pairs where BOTH baseline and strat ACCEPTED)
        paired_vs_fixed = [
            by_key[(r["repeat"], r["task"], "fixed_baseline")]["end_to_end_seconds"] / r["end_to_end_seconds"]
            for r in acc_runs
            if by_key[(r["repeat"], r["task"], "fixed_baseline")]["accepted"]
        ]
        paired_vs_expert = [
            by_key[(r["repeat"], r["task"], "expert_rules")]["end_to_end_seconds"] / r["end_to_end_seconds"]
            for r in acc_runs
            if by_key[(r["repeat"], r["task"], "expert_rules")]["accepted"]
        ]
        paired_vs_pilot = [
            by_key[(r["repeat"], r["task"], "pilot_autotuner")]["end_to_end_seconds"] / r["end_to_end_seconds"]
            for r in acc_runs
            if by_key[(r["repeat"], r["task"], "pilot_autotuner")]["accepted"]
        ]

        w_ci = _bootstrap_ci95(acc_walls, seed=hash(strat) % 1000)
        f_ci = _bootstrap_ci95(paired_vs_fixed, seed=101)
        e_ci = _bootstrap_ci95(paired_vs_expert, seed=202)
        p_ci = _bootstrap_ci95(paired_vs_pilot, seed=303)

        strategy_summary[strat] = {
            "total_runs": len(s_runs),
            "accepted_runs": len(acc_runs),
            "failed_runs": len(fail_runs),
            "completion_rate": round(len(acc_runs) / max(len(s_runs), 1), 4),
            "accepted_wall_median_s": round(statistics.median(acc_walls), 4) if acc_walls else None,
            "accepted_wall_iqr_s": round(_iqr(acc_walls), 4) if acc_walls else None,
            "accepted_wall_ci95_s": [round(w_ci[0], 4), round(w_ci[1], 4)] if acc_walls else None,
            "failed_total_wall_s": round(sum(r["end_to_end_seconds"] for r in fail_runs), 4),
            "failed_wall_median_s": round(statistics.median([r["end_to_end_seconds"] for r in fail_runs]), 4) if fail_runs else 0.0,
            "total_retries": sum(r["retries"] for r in s_runs),
            "total_retry_overhead_s": round(sum(r["retry_overhead_seconds"] for r in s_runs), 4),
            "total_solver_s": round(sum(r["solver_seconds"] for r in s_runs), 4),
            "total_verifier_s": round(sum(r["verifier_seconds"] for r in s_runs), 4),
            "total_decision_s": round(sum(r["decision_seconds"] for r in s_runs), 4),
            "total_pilot_s": round(sum(r["pilot_seconds"] for r in s_runs), 4),
            "total_campaign_all_s": round(sum(all_walls), 4),
            "campaign_median_s": round(statistics.median(camp_all), 4),
            "peak_rss_median_mb": round(statistics.median(rss_vals), 2) if rss_vals else None,
            "peak_rss_max_mb": round(max(rss_vals), 2) if rss_vals else None,
            "mutually_accepted_pairs_vs_fixed": len(paired_vs_fixed),
            "accepted_paired_speedup_vs_fixed": round(statistics.median(paired_vs_fixed), 3) if paired_vs_fixed else None,
            "accepted_paired_speedup_vs_fixed_ci95": [round(f_ci[0], 3), round(f_ci[1], 3)] if paired_vs_fixed else None,
            "mutually_accepted_pairs_vs_expert": len(paired_vs_expert),
            "accepted_paired_speedup_vs_expert": round(statistics.median(paired_vs_expert), 3) if paired_vs_expert else None,
            "accepted_paired_speedup_vs_expert_ci95": [round(e_ci[0], 3), round(e_ci[1], 3)] if paired_vs_expert else None,
            "mutually_accepted_pairs_vs_pilot": len(paired_vs_pilot),
            "accepted_paired_speedup_vs_pilot": round(statistics.median(paired_vs_pilot), 3) if paired_vs_pilot else None,
            "accepted_paired_speedup_vs_pilot_ci95": [round(p_ci[0], 3), round(p_ci[1], 3)] if paired_vs_pilot else None,
        }

    # Per-task breakdown using median across repeats
    per_task = {}
    for task in held_tasks:
        per_task[task.name] = {}
        for strat in adaptive.STRATEGY_NAMES:
            tr = [r for r in runs if r["task"] == task.name and r["strategy"] == strat]
            acc = [r for r in tr if r["accepted"]]
            per_task[task.name][strat] = {
                "accepted_count": len(acc),
                "repeats": repeats,
                "chosen_final": tr[-1]["chosen_final"],
                "accepted_median_s": round(statistics.median([r["end_to_end_seconds"] for r in acc]), 4) if acc else None,
                "failed_median_s": round(statistics.median([r["end_to_end_seconds"] for r in tr if not r["accepted"]]), 4) if len(acc) < len(tr) else None,
                "predicted_failure_prob": tr[-1]["predicted_failure_prob"],
                "l2_error": acc[-1]["final_l2_error"] if acc else tr[-1]["final_l2_error"],
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "e3_adaptive_v2",
        "repeats": repeats,
        "calibration_cases": [t.case_id for t in cal_tasks],
        "held_out_cases": [t.case_id for t in held_tasks],
        "calibration_observations": len(cal_records),
        "predictor_l2_rmse": round(predictor.l2_rmse, 4),
        "risk_weights": [round(float(x), 4) for x in predictor.w_risk],
        "strategy_summary": strategy_summary,
        "per_task_summary": per_task,
        "held_out_runs": runs,
    }
    (out_dir / "adaptive_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    render_adaptive_plots(report, out_dir)
    return report


def render_adaptive_plots(report: dict, out_dir: Path) -> None:
    """Render adaptive comparison SVGs directly from adaptive_comparison.json."""
    summary = report["strategy_summary"]
    label_map = {
        "fixed_baseline": "Fixed Q2",
        "expert_rules": "Expert Rules",
        "predictive_model": "Predictive",
        "pilot_autotuner": "Pilot Autotuner",
        "hybrid_guardrail_predictor": "Hybrid Guardrail",
    }
    bars_camp = [
        {
            "label": label_map[s],
            "sublabel": f"acc={summary[s]['accepted_runs']}/{summary[s]['total_runs']}, fail={summary[s]['failed_total_wall_s']:.1f}s",
            "value": summary[s]["campaign_median_s"],
        }
        for s in adaptive.STRATEGY_NAMES
    ]
    render_bar_comparison(
        out_dir / "fig_adaptive_campaign_cost.svg",
        title="Held-Out End-to-End Campaign Cost (6 Disjoint PDE Tasks, Incl. Retries/Pilots/Failures)",
        ylabel="Median Campaign Time per 6-Task Pass (s)",
        bars=bars_camp,
    )

    bars_rate = [
        {
            "label": label_map[s],
            "sublabel": (
                f"acc_med={summary[s]['accepted_wall_median_s']}s"
                if summary[s]["accepted_wall_median_s"] is not None
                else "0 accepted"
            ),
            "value": round(summary[s]["completion_rate"] * 100.0, 1),
        }
        for s in adaptive.STRATEGY_NAMES
    ]
    render_bar_comparison(
        out_dir / "fig_adaptive_completion_rate.svg",
        title="Held-Out Task Completion Rate (% Accepted Under Strict Accuracy & Memory Gates)",
        ylabel="Acceptance Rate (%)",
        bars=bars_rate,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="E3.1 Task 4 risk-aware adaptive strategy benchmark.")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evidence/e3")
    ap.add_argument("--mms-bin", type=Path, default=ROOT / "build/cpp/poisson_mms")
    ap.add_argument("--amr-bin", type=Path, default=ROOT / "build/cpp/poisson_amr")
    args = ap.parse_args()
    run_adaptive_benchmark(args.repeats, args.out_dir, args.mms_bin, args.amr_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
