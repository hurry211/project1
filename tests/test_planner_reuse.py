import json

import pytest

from conftest import requires_mfem
from sciagent import adaptive_advisor, controller, local_llm, planner, planning_run

MODEL = {"model": "model:local", "model_digest": "b" * 64}


def generation(**kwargs):
    return {"provider": "ollama_local", **MODEL, "options": local_llm.OPTIONS,
            "ollama_version": "test", "weight_blob_sha256": "c" * 64, "done_reason": "stop",
            "usage": {"prompt_tokens": 3, "completion_tokens": 7},
            "text": json.dumps({"template_id": "poisson_sin_sin_p1_tri_jacobi"})}


def test_bounded_reuses_successful_exact_live_preview_and_rechecks_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(local_llm, "generate", generation)
    task = planner.PlanningTask(case="sin_sin", description="prefer Jacobi")
    source = tmp_path / "preview"
    seed = planning_run.run(task, source, mode="llm", **MODEL, execute=False)
    pin = seed["manifest_sha256"]

    monkeypatch.setattr(local_llm, "generate", lambda **kw: pytest.fail("unexpected model call"))
    kwargs = dict(mode="bounded", proposal_directory=source, proposal_manifest_sha256=pin, execute=False, **MODEL)
    reused = planning_run.run(task, tmp_path / "repeat", **kwargs)
    assert reused["model_calls"] == 0 and reused["proposal_route"] == "pinned_preview"
    assert reused["template_id"] == seed["template_id"]
    assert reused["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}
    with pytest.raises(ValueError, match="settings, task or prompt"):
        planning_run.run(task.model_copy(update={"description": "prefer GS"}), tmp_path / "changed", **kwargs)
    with pytest.raises(ValueError, match="settings, task or prompt"):
        planning_run.run(task, tmp_path / "model-change", **{**kwargs, "model_digest": "d" * 64})
    (source / "response.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        planning_run.run(task, tmp_path / "edited-source", **kwargs)
    source.rename(tmp_path / "moved-source")
    assert planning_run.replay(tmp_path / "repeat", reused["manifest_sha256"])["integrity"]


def test_bounded_singleton_and_review_skip_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(local_llm, "generate", lambda **kw: pytest.fail("unnecessary model call"))
    t = planner.PlanningTask(case="sin_sin", polynomial_order=2)
    single = planning_run.run(t, tmp_path / "one", mode="bounded", **MODEL, execute=False)
    assert single["proposal_route"] == "single_eligible_template" and single["model_calls"] == 0
    review = planning_run.run(t.model_copy(update={"change_request": "modify tolerance"}), tmp_path / "review",
                              mode="bounded", **MODEL, execute=False)
    assert review["state"] == "REVIEW_REQUIRED" and review["proposal_route"] == "preflight"


@pytest.mark.parametrize("text,done,status,choice", [
    ('{"candidate_id":"a"}', "stop", "nominated", "a"),
    ('{"candidate_id":"a"}', "length", "incomplete", None),
    ('{"candidate_id":null}', "stop", "abstained", None),
    ('{"candidate_id":"other"}', "stop", "outside_shortlist", None),
    ('{"candidate_id":"a", "target":1}', "stop", "invalid_schema", None),
    ('{"candidate_id":"b","candidate_id":"a"}', "stop", "invalid_json", None),
])
def test_adaptive_model_nomination_preserves_raw_and_requires_shortlist(monkeypatch, text, done, status, choice):
    response = {"text": text, "done_reason": done}
    monkeypatch.setattr(local_llm, "generate", lambda *a, **kw: response)
    result, audit = adaptive_advisor.choose({"target_l2": 1e-6}, {}, [{"candidate_id": "a"}],
                                            model=MODEL["model"], digest=MODEL["model_digest"])
    assert result == choice and audit["response"] == response and audit["status"] == status
    assert audit["provider_completed"] and audit["used_live_llm"] is (choice is not None)
    assert json.loads(audit["prompt"])["candidates"] == [{"candidate_id": "a"}]


def test_adaptive_fallback_distinguishes_provider_error(monkeypatch):
    def unavailable(*a, **kw):
        raise local_llm.LocalProviderError("unavailable")
    monkeypatch.setattr(local_llm, "generate", unavailable)
    chosen, audit = adaptive_advisor.choose({}, {}, [{"candidate_id": "a"}],
                                            model=MODEL["model"], digest=MODEL["model_digest"])
    assert chosen is None and not audit["used_live_llm"] and not audit["provider_completed"]
    assert audit["status"] == "provider_error" and "response" not in audit



def test_adaptive_ablation_rejects_misattributed_model_choice(monkeypatch):
    from benchmarks.advisor_ablation import audit
    response = {**generation(), "text": '{"candidate_id":"alternate"}'}
    monkeypatch.setattr(local_llm, "generate", lambda *a, **kw: response)
    choice, record = adaptive_advisor.choose({"target_l2": 1e-6}, {}, [{"candidate_id": "first"},
                                            {"candidate_id": "alternate"}], model=MODEL["model"],
                                            digest=MODEL["model_digest"])
    live = {"task_id": "work", "repeat": 0, "strategy": "ai_assisted", "accepted": True, "target_l2": 1e-6,
            "meta": {"advisor": record, "used_live_llm": True}, "selected_initial_candidate": choice,
            "total_end_to_end_seconds": 2.2}
    control = {**live, "strategy": "advisor_rules", "meta": {"used_live_llm": False},
               "selected_initial_candidate": "first", "total_end_to_end_seconds": 1.1}
    result = {"design": MODEL, "held_out_task_ids": ["work"], "repeats": 1, "trials": [live, control]}
    assert audit(result)["changed_own_fallback"] == 1
    assert audit(result)["median_matched_live_minus_rules_seconds"] == pytest.approx(1.1)
    live["selected_initial_candidate"] = "first"
    with pytest.raises(ValueError, match="differs from live nomination"):
        audit(result)


@pytest.mark.mfem
@requires_mfem
def test_reused_model_proposal_cannot_accept_wrong_native_field(tmp_path, monkeypatch):
    monkeypatch.setattr(local_llm, "generate", generation)
    task = planner.PlanningTask(case="sin_sin", description="prefer Jacobi")
    source = tmp_path / "preview"
    seed = planning_run.run(task, source, mode="llm", **MODEL, execute=False)
    monkeypatch.setattr(local_llm, "generate", lambda **kwargs: pytest.fail("unexpected inference"))
    original = controller.mfem_runner.run_level

    def wrong_source(problem, execution, *args, **kwargs):
        return original(problem, execution.model_copy(update={"inject_fault": "wrong_source"}), *args, **kwargs)

    monkeypatch.setattr(controller.mfem_runner, "run_level", wrong_source)
    reused = planning_run.run(task, tmp_path / "actual", mode="bounded", **MODEL, execute=True,
                              proposal_directory=source, proposal_manifest_sha256=seed["manifest_sha256"])
    assert reused["state"] == "FAILED" and reused["template_id"] == seed["template_id"]
    assert reused["proposal_route"] == "pinned_preview" and reused["model_calls"] == 0
    replay = planning_run.replay(tmp_path / "actual", reused["manifest_sha256"])
    assert replay["integrity"] and replay["state"] == "FAILED"
