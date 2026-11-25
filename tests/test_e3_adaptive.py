"""Unit, split-hygiene, and accuracy-constraint tests for E3 Task 3 adaptive decision engine."""

from __future__ import annotations

import pytest

from benchmarks.e3_adaptive import build_calibration_tasks, build_held_out_tasks
from sciagent import adaptive
from sciagent.spec import ProblemSpec, RetryPolicy, ValidationSpec


def _sample_records():
    pol = RetryPolicy()
    hw = adaptive.HardwareFeatures(16, 8000.0, 6000.0, False, False, 100000, 2048)
    cal_tasks = build_calibration_tasks()
    recs = []
    # Physically consistent synthetic observations for pure unit test (no binary needed)
    for t in cal_tasks:
        pf = adaptive.extract_problem_features(t.problem, t.validation, t.mesh_topology, case_id=t.case_id)
        for p in (1, 2, 3, 4):
            for n in (8, 16, 32):
                dofs = (n * p + 1) ** 2
                err = 0.05 * pf.hessian_mean / ((n * p) ** (p + 1))
                sec = 0.01 * (dofs / 500.0) ** 1.05 * (1.0 + 0.08 * p)
                rss = 18.0 + dofs * 0.0005
                cand = adaptive.StrategyCandidate(
                    candidate_id=f"cart-quad-p{p}-full-cg_gs-n{n}",
                    mesh_type="quad", mesh_topology="cartesian",
                    polynomial_order=p, assembly="full", solver="cg_gs",
                    mesh_levels=(8, n) if n > 8 else (8, 16),
                    estimated_finest_dofs=dofs,
                )
                recs.append({
                    "pf": pf, "hw": hw, "cand": cand,
                    "ok": True, "l2_error": err, "wall_seconds": sec, "peak_rss_mb": rss,
                })
    return hw, adaptive.PerformancePredictor.fit(recs)


def test_calibration_and_held_out_splits_are_disjoint():
    cal = build_calibration_tasks()
    held = build_held_out_tasks()
    cal_sigs = {(t.problem.exact_solution, t.problem.kappa, t.mesh_topology, tuple(b.kind for b in t.problem.boundaries)) for t in cal}
    held_sigs = {(t.problem.exact_solution, t.problem.kappa, t.mesh_topology, tuple(b.kind for b in t.problem.boundaries)) for t in held}
    assert not (cal_sigs & held_sigs), f"data leakage between calibration and held-out splits: {cal_sigs & held_sigs}"
    assert len({t.task_id for t in cal} & {t.task_id for t in held}) == 0


def test_problem_feature_extractor_distinguishes_steep_vs_smooth_fields():
    cal = build_calibration_tasks()
    held = build_held_out_tasks()
    smooth = next(t for t in cal if t.case_id == "sin_sin")
    steep = next(t for t in held if t.case_id == "gaussian_steep")
    pf_sm = adaptive.extract_problem_features(smooth.problem, smooth.validation)
    pf_st = adaptive.extract_problem_features(steep.problem, steep.validation)
    assert pf_st.localization_ratio > 3.0 * pf_sm.localization_ratio
    assert pf_st.hessian_max > 10.0 * pf_sm.hessian_max


def test_accuracy_constraint_vetoes_cheap_inaccurate_q1_and_ai_fallback_safe():
    hw, predictor = _sample_records()
    held_tight = next(t for t in build_held_out_tasks() if t.task_id == "heldout_tight_mixed")
    pf = adaptive.extract_problem_features(held_tight.problem, held_tight.validation)
    from sciagent import capability
    env = {
        "mfem": {
            "found": True, "version": capability.MFEM_VERSION,
            "git": capability.MFEM_COMMIT,
            "config": capability.REVIEWED_FLAGS.copy(),
        }
    }
    cands, excluded = adaptive.enumerate_candidates(held_tight, hw, env)
    # Must explicitly audit and exclude MPI when not built with MPI
    assert any(e["code"] == "UNSUPPORTED_BACKEND" for e in excluded)
    chosen, _ = adaptive.choose_predictive_candidate(pf, hw, cands, predictor)
    # Cheap Q1 must NOT be chosen on an L2<=1e-7 accuracy target
    assert chosen.polynomial_order >= 3
    pred = predictor.predict(pf, chosen, hw)
    assert pred["pred_l2_error"] <= held_tight.validation.max_l2_error_finest


def test_negative_predictor_rejects_underdetermined_calibration():
    with pytest.raises(ValueError, match="insufficient"):
        adaptive.PerformancePredictor.fit([])
