"""Controller tests that do not need MFEM: rejection before any execution."""

import json

import pytest

from sciagent import controller


@pytest.mark.parametrize("spec,code", [
    ("negative/missing_boundary.json", "INVALID_SPEC"),
    ("negative/duplicate_attribute.json", "INVALID_SPEC"),
    ("negative/unsupported_bc_value.json", "INVALID_SPEC"),
    ("negative/unknown_field.json", "INVALID_SPEC"),
    ("negative/non_doubling_levels.json", "INVALID_SPEC"),
    ("negative/inconsistent_mms_source.json", "MMS_INCONSISTENT"),
    ("negative/malicious_expression.json", "INVALID_EXPRESSION"),
])
def test_invalid_specs_stop_before_execution(specs_dir, tmp_path, spec, code):
    m = controller.run_study(specs_dir / spec, tmp_path / "runs")
    assert m["final_state"] == "INVALID_SPEC" and m["final_code"] == code
    run_dir = next((tmp_path / "runs").iterdir())
    assert not (run_dir / "attempts").exists()
    assert (run_dir / "request" / "study_spec.json").read_text() == (specs_dir / spec).read_text()
    states = [json.loads(l)["state"] for l in (run_dir / "events.jsonl").read_text().splitlines()]
    assert states == ["RECEIVED", "INVALID_SPEC"]


def test_evidence_store_is_append_only(tmp_path):
    from sciagent.evidence import EvidenceStore
    s = EvidenceStore(tmp_path / "r")
    s.write_once("a.txt", "x")
    with pytest.raises(FileExistsError):
        s.write_once("a.txt", "y")
    with pytest.raises(FileExistsError):
        EvidenceStore(tmp_path / "r")


def test_initial_dof_budget_stops_before_any_solver_call(specs_dir, tmp_path, monkeypatch):
    path = tmp_path / "limited.json"
    spec = json.loads((specs_dir / "poisson_sin_sin_p1_tri.json").read_text())
    spec["policy"]["max_dofs_per_level"] = 100
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(controller.environment, "probe", lambda: {
        "mfem": {"found": True}, "solver_binary_sha256": "test", "cxx": "test",
    })
    monkeypatch.setattr(controller.mfem_runner, "list_cases", lambda: {
        "cases": [{**spec["problem"], "id": "sin_sin"}],
    })
    monkeypatch.setattr(controller.mfem_runner, "run_level", lambda *a, **kw: pytest.fail("solver was launched"))
    m = controller.run_study(path, tmp_path / "runs")
    assert (m["final_state"], m["final_code"], m["attempts"]) == ("BUDGET_EXHAUSTED", "RESOURCE_LIMIT", [])


def test_quickstart_reports_missing_solver(tmp_path, monkeypatch, capsys):
    from sciagent.__main__ import main

    monkeypatch.setenv("THIRD_PARTY", str(tmp_path / "empty"))
    monkeypatch.delenv("MFEM_INSTALL", raising=False)
    monkeypatch.setenv("POISSON_MMS_BIN", str(tmp_path / "missing"))
    assert main(["quickstart", "--runs-dir", str(tmp_path / "runs")]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["state"] == "BLOCKED" and report["code"] == "MFEM_NOT_FOUND"
    assert report["tools"]["mfem"] is None and report["tools"]["solver_available"] is False


def test_unusable_solver_reports_build_error(tmp_path):
    from sciagent.mfem_runner import BuildError, list_cases

    binary = tmp_path / "poisson_mms"
    binary.write_text("incomplete binary")
    with pytest.raises(BuildError, match="cannot execute"):
        list_cases(binary)
