import json
import shutil
from pathlib import Path

import pytest

from conftest import requires_mfem
from sciagent import controller, local_llm, planner, planning_run
from sciagent.__main__ import main
from sciagent.evidence import sha256_file

DIGEST = "a" * 64


def _reply(text, *, done_reason="stop", **kwargs):
    return {"provider": "ollama_local", "model": kwargs["model"], "model_digest": kwargs["model_digest"],
            "options": local_llm.OPTIONS, "ollama_version": "test", "weight_blob_sha256": "b" * 64,
            "text": text, "done_reason": done_reason, "usage": {"prompt_tokens": 42, "completion_tokens": 8}}


def test_preview_persists_pinned_candidate_and_replays_without_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "run_study", lambda *a, **k: pytest.fail("a preview launched MFEM"))
    directory = tmp_path / "preview"
    task = planner.PlanningTask(case="sin_sin", description="精" * 2000)
    receipt = planning_run.run(task, directory, execute=False)
    assert receipt["state"] == "PLANNED"
    assert sha256_file(directory / "template/study.json") == planner.template_sha256(receipt["template_id"])
    assert planning_run.replay(directory, receipt["manifest_sha256"])["state"] == "PLANNED"
    with pytest.raises(FileExistsError):
        planning_run.run(task, directory)


@pytest.mark.parametrize("reply,reason,code", [
    ('{"template_id":"poisson_cos_exp_mixed_p1_quad"}', "stop", "UNAUTHORIZED_TEMPLATE"),
    ('{"template_id":"poisson_sin_sin_p1_tri","validation":{}}', "stop", "INVALID_PROPOSAL"),
    ('{"template_id":"poisson_sin_sin_p1_tri"}', "length", "INCOMPLETE_PROPOSAL"),
])
def test_candidate_cannot_authorize_execution(tmp_path, monkeypatch, reply, reason, code):
    monkeypatch.setattr(local_llm, "generate", lambda **kw: _reply(reply, done_reason=reason, **kw))
    monkeypatch.setattr(controller, "run_study", lambda *a, **kw: pytest.fail("numerical run was not authorized"))
    directory = tmp_path / "request"
    receipt = planning_run.run(planner.PlanningTask(case="sin_sin"), directory, mode="llm",
                               model="example:local", model_digest=DIGEST)
    assert (receipt["state"], receipt["code"], receipt["model_calls"]) == ("REJECTED", code, 1)
    monkeypatch.setattr(local_llm, "generate", lambda **kw: pytest.fail("replay called the provider"))
    assert planning_run.replay(directory, receipt["manifest_sha256"])["state"] == "REJECTED"
    assert not (directory / "studies").exists()


def test_review_and_model_failure_never_fall_back_to_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "run_study", lambda *a, **kw: pytest.fail("unexpected execution"))
    monkeypatch.setattr(local_llm, "generate", lambda **kw: pytest.fail("review called the provider"))
    config = {"mode": "llm", "model": "example:local", "model_digest": DIGEST}
    review = planning_run.run(planner.PlanningTask(case="sin_sin", change_request="change physics"),
                              tmp_path / "review", **config)
    assert (review["state"], review["model_calls"]) == ("REVIEW_REQUIRED", 0)
    assert planning_run.replay(tmp_path / "review", review["manifest_sha256"])["integrity"]

    def unavailable(**kwargs):
        raise ValueError("local service unavailable")

    monkeypatch.setattr(local_llm, "generate", unavailable)
    failed = planning_run.run(planner.PlanningTask(case="sin_sin"), tmp_path / "offline", **config)
    assert failed["code"] == "PROVIDER_ERROR" and failed["usage"]["prompt_tokens"] is None
    assert planning_run.replay(tmp_path / "offline", failed["manifest_sha256"])["state"] == "FAILED"


@pytest.mark.parametrize("change", ["validation", "line_endings"])
def test_controller_checks_the_template_it_actually_reads(tmp_path, specs_dir, monkeypatch, change):
    path = tmp_path / "study.json"
    shutil.copyfile(specs_dir / "poisson_sin_sin_p1_tri.json", path)
    expected = planner.template_sha256("poisson_sin_sin_p1_tri")
    if change == "line_endings":
        path.write_bytes(path.read_text().replace("\n", "\r\n").encode())
    else:
        changed = json.loads(path.read_text())
        changed["validation"]["require_reference_solve"] = False
        path.write_text(json.dumps(changed))
    monkeypatch.setattr(controller.environment, "probe", lambda: pytest.fail("digest gate reached environment probe"))
    record = controller.run_study(path, tmp_path / "runs", expected_spec_sha256=expected)
    assert (record["final_state"], record["final_code"], record["attempts"]) == ("FAILED", "SPEC_TAMPERED", [])
    assert (Path(record["run_dir"]) / "request/study_spec.json").read_bytes() == path.read_bytes()


@pytest.mark.mfem
@requires_mfem
def test_planner_cli_executes_and_replays_relocated_named_mesh(tmp_path, specs_dir, capsys, monkeypatch):
    directory = tmp_path / "run"
    assert main(["plan", str(specs_dir / "stage4/choose_diffusion.json"),
                 "--directory", str(directory), "--execute"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "ACCEPTED" and result["template_id"] == "diffusion_gmsh_p2_quad"
    moved = directory.rename(tmp_path / "moved")
    monkeypatch.setattr(controller, "run_study", lambda *a, **kw: pytest.fail("replay launched solver"))
    assert main(["verify-plan", str(moved), "--manifest-sha256", result["manifest_sha256"]]) == 0
    assert json.loads(capsys.readouterr().out)["integrity"]
    (moved / "template/study.json").write_text("{}")
    assert main(["verify-plan", str(moved), "--manifest-sha256", result["manifest_sha256"]]) == 1


@pytest.mark.mfem
@requires_mfem
def test_authorized_model_choice_cannot_accept_a_wrong_native_solution(tmp_path, monkeypatch):
    selected = "poisson_sin_sin_p1_tri"
    monkeypatch.setattr(local_llm, "generate", lambda **kw: _reply(json.dumps({"template_id": selected}), **kw))
    native_level = controller.mfem_runner.run_level

    def wrong_source(problem, execution, *args, **kwargs):
        return native_level(problem, execution.model_copy(update={"inject_fault": "wrong_source"}), *args, **kwargs)

    monkeypatch.setattr(controller.mfem_runner, "run_level", wrong_source)
    directory = tmp_path / "wrong-equation"
    result = planning_run.run(planner.PlanningTask(case="sin_sin"), directory, mode="llm",
                              model="example:local", model_digest=DIGEST)
    assert result["template_id"] == selected and result["state"] == "FAILED"
    child, = (directory / "studies").glob("*/manifest.json")
    attempt, = json.loads(child.read_text())["attempts"]
    assert all(level["exit_code"] == 0 for level in attempt["levels"])
    assert "REFERENCE_MISMATCH" in attempt["failure_codes"]
    assert planning_run.replay(directory, result["manifest_sha256"])["state"] == "FAILED"
