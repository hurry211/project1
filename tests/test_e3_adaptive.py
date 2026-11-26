"""Unit, split-hygiene, zero-leakage, and risk-guardrail tests for E3.1 adaptive decision engine."""

from __future__ import annotations

import pytest

from benchmarks.e31_adaptive import build_calibration_tasks, build_held_out_tasks
from sciagent import adaptive_e31 as adaptive


def _synthetic_records(n_repeat: int = 3) -> list[dict]:
    hw = adaptive.HardwareFeatures(16, 8000.0, 6000.0, False, False, 60_000, 2048)
    cal = build_calibration_tasks()
    recs = []
    for task in cal:
        pf = adaptive.extract_problem_features(task.problem, task.validation, task.mesh_topology, case_id=task.case_id)
        for p in (1, 2, 3):
            for n0 in (4, 8, 16):
                cand = adaptive.StrategyCandidate("quad", "cartesian", p, "full", "cg_gs", n0)
                dofs = cand.max_dofs_estimate()
                base = 0.25 * ((3.14159 / (4.0 * n0 * p)) ** (p + 1)) * (1.0 + pf.source_localization * 0.2)
                wall = 0.05 + (dofs / 12000.0) * (0.4 + 0.1 * p)
                recs.append({
                    "task": task.name,
                    "target_l2": task.validation.max_l2_error_finest,
                    "verdict": "ACCEPTED" if base <= (task.validation.max_l2_error_finest or 1.0) else "RETRY_ELIGIBLE",
                    "candidate": vars(cand),
                    "problem_features": vars(pf),
                    "hw_features": vars(hw),
                    "l2_error": base,
                    "wall_seconds": wall,
                    "peak_rss_mb": 64.0 + 0.002 * dofs,
                })
    return recs


def test_calibration_and_held_out_strict_split_hygiene_and_names():
    cal = build_calibration_tasks()
    held = build_held_out_tasks()
    assert len(cal) >= 6 and len(held) >= 6
    assert not (set(t.case_id for t in cal) & set(t.case_id for t in held))
    assert not (set(t.problem.exact_solution for t in cal) & set(t.problem.exact_solution for t in held))
    assert "hybrid_guardrail_predictor" in adaptive.STRATEGY_NAMES
    assert "ai_assisted" not in adaptive.STRATEGY_NAMES


def test_zero_interior_exact_solution_leakage():
    """Altering exact_solution ONLY inside Omega (zero bubble x*(1-x)*y*(1-y) on dOmega) must NOT change features."""
    held = build_held_out_tasks()
    var_task = next(t for t in held if t.case_id == "cos_2x_sin_y_var")
    perturbed = var_task.problem.model_copy(
        update={"exact_solution": f"({var_task.problem.exact_solution})+1e6*x*(1-x)*y*(1-y)"}
    )
    pf_orig = adaptive.extract_problem_features(var_task.problem, var_task.validation)
    pf_pert = adaptive.extract_problem_features(perturbed, var_task.validation)
    assert pf_orig == pf_pert


def test_risk_constrained_selection_avoids_high_failure_risk():
    hw = adaptive.HardwareFeatures(16, 8000.0, 6000.0, False, False, 60_000, 2048)
    pred = adaptive.PerformancePredictor.fit(_synthetic_records())
    held_tight = next(t for t in build_held_out_tasks() if t.name == "held_sin_3x_2y_tight")
    pf = adaptive.extract_problem_features(held_tight.problem, held_tight.validation)

    cands, _ = adaptive.enumerate_candidates(held_tight, hw, env={
        "mfem": {
            "version": "4.10.0",
            "git": "d964264cdb9a13e94a201b6c236c7721e0c8765f",
            "config": {k: (k == "MFEM_USE_DOUBLE") for k in (
                "MFEM_USE_MPI", "MFEM_USE_METIS", "MFEM_USE_METIS_5", "MFEM_USE_SUITESPARSE",
                "MFEM_USE_LAPACK", "MFEM_USE_OPENMP", "MFEM_USE_CUDA", "MFEM_USE_HIP",
                "MFEM_USE_RAJA", "MFEM_USE_OCCA", "MFEM_USE_PETSC", "MFEM_USE_ZLIB",
                "MFEM_USE_DOUBLE", "MFEM_USE_SINGLE",
            )},
        }
    })
    assert cands
    chosen_pred, info_pred = adaptive.choose_predictive_candidate(pf, hw, cands, pred, max_risk=0.25)
    chosen_hyb, info_hyb = adaptive.choose_hybrid_guardrail_candidate(held_tight, pf, hw, cands, pred, max_risk=0.15)
    assert chosen_pred.polynomial_order >= 2
    assert chosen_hyb.polynomial_order >= 2
    assert 0.0 <= info_pred["pred_failure_prob"] <= 1.0
    assert 0.0 <= info_hyb["pred_failure_prob"] <= 1.0
    # Verify logistic classifier assigns strictly lower risk to fine high-order than coarse low-order
    coarse_q1 = adaptive.StrategyCandidate("quad", "cartesian", 1, "full", "cg_gs", 8)
    fine_q3 = adaptive.StrategyCandidate("quad", "cartesian", 3, "full", "cg_gs", 16)
    assert pred.predict(pf, hw, fine_q3)["pred_failure_prob"] < pred.predict(pf, hw, coarse_q1)["pred_failure_prob"]


def test_predictor_rejects_insufficient_calibration_data():
    with pytest.raises(ValueError, match="at least|>= 5"):
        adaptive.PerformancePredictor.fit([])
