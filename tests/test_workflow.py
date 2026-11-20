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
