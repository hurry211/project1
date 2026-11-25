import json
import time

import pytest

from benchmarks.adaptive_audit import verify_field
from benchmarks.adaptive_measure import measured
from benchmarks.e3_adaptive import build_held_out_tasks, paired_results
from conftest import requires_mfem
from sciagent.adaptive import StrategyCandidate, execute_candidate
from sciagent.spec import RetryPolicy
from verifier.verifier import verify_attempt


def test_failed_pairs_cannot_create_a_speedup():
    def trial(task, rep, strategy, ok, seconds):
        return dict(task_id=task, repeat=rep, strategy=strategy, accepted=ok,
                    total_end_to_end_seconds=seconds, target_l2=1e-4)

    rows = [trial("a", 0, "fixed_baseline", False, 30), trial("a", 0, "expert_rules", True, 1)]
    score = paired_results(rows)["expert_rules_vs_fixed_baseline"]
    assert score["matched_accepted"] == 0 and score["median_ratio"] is None
    rows += [trial("b", 0, "fixed_baseline", True, 4), trial("b", 0, "expert_rules", True, 2)]
    score = paired_results(rows)["expert_rules_vs_fixed_baseline"]
    assert score["median_ratio"] == 2 and score["matched_tasks"] == score["matched_accepted"] == 1


def test_parent_records_a_timed_out_worker(tmp_path):
    receipt = tmp_path / "timeout.json"
    result = measured(receipt, RetryPolicy(max_total_wall_seconds=0.1), time.sleep, 10)
    assert result["failure_codes"] == ["WALL_BUDGET_EXHAUSTED"]
    assert not result["accepted"] and json.loads(receipt.read_bytes()) == result


@pytest.mark.mfem
@requires_mfem
def test_coarse_gaussian_integrates_independently_and_checks_order(tmp_path):
    task = next(t for t in build_held_out_tasks() if t.case_id == "gaussian_steep")
    candidate = StrategyCandidate("q3", "quad", "cartesian", 3, "full", "cg_gs", (8, 16, 32), 9409)
    run = tmp_path / "regular"
    trial = execute_candidate(task, candidate, run)
    assert trial["ok"] and trial["l2_error"] <= 4e-5
    assert verify_field(run) == "ACCEPTED"
    attempt = run / "run/attempts/attempt_01"
    result = attempt / "levels/n0008/result.json"
    native = json.loads(result.read_bytes())
    assert native["error_quadrature_order"] == 20
    native["error_quadrature_order"] = 10
    result.write_text(json.dumps(native))
    assert "QUADRATURE_CLAIM_MISMATCH" in verify_attempt(attempt, require_run_records=True)["failure_codes"]


@pytest.mark.mfem
@requires_mfem
def test_sharp_amr_with_ten_steps_and_original_dof_budget(tmp_path):
    task = next(t for t in build_held_out_tasks() if t.case_id == "gaussian_steep")
    candidate = StrategyCandidate("AMR", "quad", "amr", 2, "full", "cg_gs", (8,), 4000,
                                  amr_max_steps=10, amr_max_dofs=4000)
    folder = tmp_path / "refined"
    result = execute_candidate(task, candidate, folder)
    assert result["ok"] and result["l2_error"] <= task.validation.max_l2_error_finest
    study = json.loads((folder / "verdict.json").read_bytes())
    assert study["step_count"] <= 10 and all(step["num_true_dofs_inferred"] <= 4000 for step in study["steps"])


@pytest.mark.mfem
@requires_mfem
def test_extended_catalog_is_opt_in_and_binds_frozen_formulation():
    from sciagent import mfem_runner
    from benchmarks.e3_adaptive import build_calibration_tasks

    task = next(t for t in build_calibration_tasks() if t.case_id == "sin_2x_2y")
    assert mfem_runner.bind_case(task.problem, mfem_runner.list_cases()) is None
    assert mfem_runner.bind_case(task.problem, mfem_runner.list_cases(include_e3=True)) == task.case_id


def test_search_rejects_duplicate_solvers_before_probing(specs_dir, tmp_path):
    from sciagent.spec import load_study
    from sciagent.strategy import search

    study = load_study(specs_dir / "experiments/strategy_poisson.json")
    with pytest.raises(ValueError, match="distinct"):
        search(study, tmp_path / "results", 1, solvers=("cg_gs", "cg_gs"))
    assert not (tmp_path / "results").exists()
