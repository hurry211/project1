"""Independent scoring of the paired planning panel."""

import json

import pytest

from benchmarks import planning
from sciagent import planner


def test_preregistered_panel_and_postproposal_controls():
    rows, controls = planning.protocol()
    assert len(rows) == len({r["id"] for r in rows}) == 19
    assert [r["group"] for r in rows].count("preference") == 9
    assert len(controls) == 4 and all(c["passed"] for c in controls)
    for row in rows:
        task = planner.PlanningTask.model_validate(row["task"])
        assert set(planner.eligible_ids(task)) == set(row["permitted_ids"])
        if row["expected_state"] == "ACCEPTED":
            assert row["expected_template_id"] in row["permitted_ids"]
        else:
            assert row["expected_state"] == "REVIEW_REQUIRED" and not row["permitted_ids"]


def trial(row, status, template, *, calls=1, code="OK", tokens=8):
    return {"id": row["id"], "state": status, "template_id": template, "code": code,
            "proposal_calls": calls, "model_calls": calls, "proposal_retries": 0, "numerical_retries": 0,
            "usage": {"prompt_tokens": tokens, "completion_tokens": tokens},
            "wall_seconds": 1.0, "external_replay_seconds": 0.25}


def test_score_separates_scope_preference_runtime_failure_and_abstention():
    rows, _ = planning.protocol()
    exact, preference, missing, abstain, hold, unsafe = [rows[i] for i in (0, 5, 1, 2, 14, 15)]
    group = [exact, preference, missing, abstain, hold, unsafe]
    other = next(t for t in preference["permitted_ids"] if t != preference["expected_template_id"])
    records = [trial(exact, "ACCEPTED", exact["expected_template_id"]),
               trial(preference, "ACCEPTED", other),
               trial(missing, "FAILED", None, code="PROVIDER_ERROR", tokens=None),
               trial(abstain, "REJECTED", None, code="ABSTAINED"),
               trial(hold, "REVIEW_REQUIRED", None, calls=0),
               trial(unsafe, "ACCEPTED", "unapproved")]
    result = planning.score(group, records)
    assert result["eligible_tasks"] == 4 and result["review_controls"] == 2
    assert result["authorized_verified_completions"] == 2 and result["intended_verified_completions"] == 1
    assert result["correct_actions"] == 2 and result["authorized_wrong_preference"] == 1
    assert result["preference_hits"] == 0 and result["eligible_without_verified_completion"] == 2
    assert result["false_rejects_on_eligible_tasks"] == 1 and result["false_accepts_on_review_controls"] == 1
    assert result["unauthorized_acceptances_all_tasks"] == 1 and result["provider_errors"] == 1
    assert result["model_responses_rejected_by_code"] == {"ABSTAINED": 1}
    assert result["tokens"]["prompt_tokens"] == {"total": None, "known_sum": 32, "unreported_trials": 1}
    assert result["workflow_seconds_total"] == 6.0 and result["external_replay_seconds_total"] == 1.5


def test_subset_or_duplicate_does_not_reduce_denominator():
    rows, _ = planning.protocol()
    row = rows[0]
    r = trial(row, "ACCEPTED", row["expected_template_id"])
    for records in ([r], [r, r]):
        with pytest.raises(ValueError, match="incomplete trial panel"):
            planning.score(rows[:2], records)


def test_report_digest_is_checked_before_replay(tmp_path):
    f = tmp_path / "result.json"
    f.write_text(json.dumps({"trials": []}))
    with pytest.raises(ValueError, match="digest mismatch"):
        planning.pinned_json(f, "0" * 64)


def test_missing_model_does_not_qualify_as_a_live_comparison():
    rows, _ = planning.protocol()
    row = rows[0]
    failed = planning.score([row], [trial(row, "FAILED", None, code="PROVIDER_ERROR", tokens=None)])
    rejected = planning.score([row], [trial(row, "REJECTED", None, code="INVALID_PROPOSAL")])
    result = {"metrics": {"llm": failed}, "gate_controls": [{"passed": True}]}
    with pytest.raises(ValueError, match="no completed live"):
        planning.validate_comparison(result)
    result["metrics"]["llm"] = rejected
    planning.validate_comparison(result)


@pytest.mark.parametrize("field,bad", [("wall_seconds", float("nan")), ("external_replay_seconds", -0.01)])
def test_invalid_runtime_is_rejected(field, bad):
    rows, _ = planning.protocol()
    row = rows[0]
    record = trial(row, "ACCEPTED", row["expected_template_id"])
    record[field] = bad
    with pytest.raises(ValueError, match="invalid measured"):
        planning.score([row], [record])
