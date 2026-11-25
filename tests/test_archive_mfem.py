"""Stored-evidence checks against a real MFEM run, including relocation."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT, requires_mfem
from sciagent import controller
from sciagent.__main__ import main
from verifier.verifier import verify_attempt

pytestmark = [pytest.mark.mfem, requires_mfem]


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    base = tmp_path_factory.mktemp("archive")
    spec = json.loads((ROOT / "specs/poisson_sin_sin_p1_tri.json").read_text())
    spec["execution"]["mesh_levels"] = [4, 8, 16]
    path = base / "study.json"
    path.write_text(json.dumps(spec))
    manifest = controller.run_study(path, base)
    assert manifest["final_state"] == "ACCEPTED"
    return Path(manifest["run_dir"])


@pytest.fixture
def archive(original, tmp_path):
    return shutil.copytree(original, tmp_path / "relocated")


@pytest.mark.parametrize("field,value,gate", [
    ("rhs_quadrature_order", 1, "quadrature_orders"),
    ("error_quadrature_order", 1, "quadrature_orders"),
    ("solver_iterations", 20001, "cg_stopping_record"),
    ("solver_iterations", True, "cg_stopping_record"),
    ("solver_final_norm", 1.0, "cg_stopping_record"),
    ("solver_initial_norm", 10**400, "cg_stopping_record"),
    ("abs_tol", 1.0, "cg_stopping_record"),
])
def test_native_record_cannot_describe_an_incompatible_solve(archive, field, value, gate):
    attempt = archive / "attempts/attempt_01"
    path = attempt / "levels/n0004/result.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    report = verify_attempt(attempt, require_run_records=True)
    assert report["verdict"] == "REJECTED"
    assert [c["name"] for c in report["levels"][0]["checks"] if not c["passed"]] == [gate]


def test_relocated_archive_with_trusted_digest(archive, original):
    digest = hashlib.sha256((original / "manifest.json").read_bytes()).hexdigest()
    result = controller.reverify(archive / "attempts/attempt_01", digest)
    assert result["verdict"] == "ACCEPTED"
    assert result["integrity"] == {"passed": True, "mode": "sha256-inventory"}
    manifest = json.loads((archive / "manifest.json").read_text())
    snapshot = (archive / "case_registry.json").read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == manifest["artifact_sha256"]["case_registry.json"]
    catalog = json.loads(snapshot)
    selected, = [case for case in catalog["cases"] if case["id"] == manifest["bound_case"]]
    problem = json.loads((archive / "frozen/problem.json").read_text())
    assert catalog["formulation"]["equation"] == problem["equation"]
    assert selected["source"] == problem["source"]


@pytest.mark.parametrize("file,operation", [
    ("attempts/attempt_01/verdict.json", "verdict"),
    ("attempts/attempt_01/verdict.json", "remove"),
    ("attempts/attempt_01/levels/n0008/stdout.txt", "append"),
    ("attempts/attempt_01/levels/n0004/solution.gf", "remove"),
    ("attempts/attempt_01/levels/n0004/extra.txt", "append"),
    ("attempts/attempt_01/levels/n0004/result.json", "swap_levels"),
    ("attempts/attempt_01/execution.json", "remove"),
    ("events.jsonl", "reverse"),
    ("frozen/validation.json", "relax"),
    ("frozen/policy.json", "remove"),
    ("manifest.json", "summary"),
    ("manifest.sha256", "remove"),
])
def test_archive_changes_return_typed_rejection(archive, file, operation):
    path = archive / file
    if operation == "remove":
        path.unlink()
    elif operation == "reverse":
        path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    elif operation == "swap_levels":
        path.write_bytes((path.parent.parent / "n0008" / path.name).read_bytes())
    elif operation == "append":
        with path.open("a") as f:
            f.write("extra line\n")
    else:
        obj = json.loads(path.read_text())
        if operation == "verdict":
            obj["verdict"] = "REJECTED"
        elif operation == "relax":
            obj["require_reference_solve"] = False
        else:
            obj["final_state"] = "FAILED"
        path.write_text(json.dumps(obj))

    result = controller.reverify(archive / "attempts/attempt_01")
    assert result["verdict"] == "REJECTED" and "EVIDENCE_INTEGRITY" in result["failure_codes"]
    assert not result["integrity"]["passed"]


def test_external_manifest_digest_checked_even_with_matching_local_checksum(archive, original):
    path = archive / "manifest.json"
    trusted = hashlib.sha256((original / path.name).read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes() + b"\n")
    changed = hashlib.sha256(path.read_bytes()).hexdigest()
    (archive / "manifest.sha256").write_text(f"{changed}  manifest.json\n")
    assert main(["verify", str(archive / "attempts/attempt_01"), "--manifest-sha256", trusted]) == 1


def test_report_checks_evidence_before_writing(original, tmp_path):
    run = shutil.copytree(original, tmp_path / original.name)
    output = tmp_path / "report.md"
    cmd = [sys.executable, str(ROOT / "scripts/summarize_runs.py"), str(tmp_path), "--output", str(output)]
    subprocess.run(cmd, check=True, capture_output=True)
    report = output.read_text()
    assert "ACCEPTED" in report and "| 16 | 1/16 | 289 |" in report
    native = json.loads((run / "attempts/attempt_01/levels/n0016/result.json").read_text())["diagnostics"]
    residual = native["algebraic_residual"]
    flux = native["boundary_flux"]
    assert "Native solver diagnostics (informational)" in report
    assert (f"| 16 | {residual['absolute_l2']:.4e} | {residual['rhs_l2']:.4e} | "
            f"{residual['relative_l2']:.4e} | {flux['outward_total']:.4e} | "
            f"{flux['source_integral']:.4e} | {flux['signed_imbalance']:.4e} |") in report
    (run / "events.jsonl").write_text("")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0 and "events.jsonl" in proc.stderr
    assert output.read_text() == report


def test_nonzero_process_exit_rejects_otherwise_correct_data(archive):
    attempt = archive / "attempts/attempt_01"
    path = attempt / "levels/n0004/run_record.json"
    record = json.loads(path.read_text())
    record.update(exit_code=5, exit_meaning="SOLVER_NOT_CONVERGED")
    path.write_text(json.dumps(record))
    verdict = verify_attempt(attempt, require_run_records=True)
    assert verdict["verdict"] == "REJECTED" and verdict["failure_codes"] == ["SOLVER_EXIT"]


def test_legacy_archive_checks_original_request_and_stored_hashes(archive):
    attempt = archive / "attempts/attempt_01"
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["evidence_version"], manifest["artifact_sha256"]
    manifest_path.write_text(json.dumps(manifest))
    (archive / "manifest.sha256").unlink()
    path = attempt / "verdict.json"
    verdict = json.loads(path.read_text())
    verdict["verifier_version"] = "0.1.0"
    path.write_text(json.dumps(verdict))
    assert controller.reverify(attempt)["integrity"] == {"mode": "legacy-input-hashes", "passed": True}

    path = archive / "frozen/policy.json"
    data = json.loads(path.read_text())
    data["max_attempts"] += 1
    path.write_text(json.dumps(data))
    manifest["frozen_sha256"]["policy"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    assert controller.reverify(attempt)["verdict"] == "REJECTED"


def test_frozen_validation_checked_after_solver(specs_dir, tmp_path, monkeypatch):
    run_level = controller.mfem_runner.run_level

    def change_validation(*args, **kwargs):
        record = run_level(*args, **kwargs)
        path = next(tmp_path.glob("*/frozen/validation.json"))
        path.write_text("{}")
        return record

    monkeypatch.setattr(controller.mfem_runner, "run_level", change_validation)
    m = controller.run_study(specs_dir / "poisson_sin_sin_p1_tri.json", tmp_path)
    assert (m["final_state"], m["final_code"]) == ("FAILED", "SPEC_TAMPERED")
