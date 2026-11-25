"""Numerical and evidence regressions for reviewed performance paths."""

import json
import shutil
from pathlib import Path

import pytest

from conftest import requires_mfem
from sciagent import controller
from sciagent.evidence import artifact_hashes, sha256_file
from sciagent.spec import load_study
from sciagent.strategy import search

pytestmark = [pytest.mark.mfem, requires_mfem]


def write_variant(source, target, section, **values):
    spec = json.loads(source.read_text())
    spec.setdefault(section, {}).update(values)
    target.write_text(json.dumps(spec))
    return target


@pytest.mark.parametrize("order", [1, 2])
def test_matched_assembly(specs_dir, tmp_path, order):
    errors = []
    for assembly in ("full", "partial"):
        spec = write_variant(specs_dir / "poisson_sin_sin_p1_tri.json", tmp_path / f"{assembly}.json",
                             "execution", mesh_type="quad", polynomial_order=order,
                             mesh_levels=[4, 8, 16], assembly=assembly, solver="cg_jacobi")
        run = controller.run_study(spec, tmp_path)
        assert run["final_state"] == "ACCEPTED", run["final_reason"]
        attempt = run["attempts"][-1]
        assert attempt["verification"]["peak_rss_kib"] > 0
        errors.append(attempt["errors"][-1]["l2"])
        result = json.loads((Path(run["run_dir"]) / "attempts/attempt_01/levels/n0016/result.json").read_text())
        assert result["peak_rss_kib"] > 0
        assert all(result[f"time_{phase}_ms"] >= 0 for phase in ("assembly", "preconditioner", "solve"))
    assert errors[0] == pytest.approx(errors[1], rel=1e-7)


def test_retry_reuse_is_portable_and_replayable(specs_dir, tmp_path):
    spec = specs_dir / "retry_demo_sin_sin_p1_tri.json"
    run = controller.run_study(spec, tmp_path)
    assert run["final_state"] == "ACCEPTED", run["final_reason"]
    root = Path(run["run_dir"])
    total_reused = 0
    for i, summary in enumerate(run["attempts"], 1):
        attempt = root / "attempts" / f"attempt_{i:02d}"
        for level in summary["levels"]:
            if "reused_from" in level:
                total_reused += 1
                assert artifact_hashes(attempt / "levels" / f"n{level['n']:04d}") == artifact_hashes(
                    root / level["reused_from"])
        replay = controller.reverify(attempt, sha256_file(root / "manifest.json"))
        assert replay["verdict"] == summary["verdict"]
    n_unique = len(run["attempts"][-1]["levels"])
    assert total_reused == sum(len(a["levels"]) for a in run["attempts"]) - n_unique > 0


def test_retry_detects_corruption_before_reusing_verdict(specs_dir, tmp_path, monkeypatch):
    copytree = controller.shutil.copytree

    def corrupt(source, dest, **kwargs):
        copytree(source, dest, **kwargs)
        (dest / "stdout.txt").write_text("corrupted")

    monkeypatch.setattr(controller.shutil, "copytree", corrupt)
    run = controller.run_study(specs_dir / "retry_demo_sin_sin_p1_tri.json", tmp_path)
    assert (run["final_state"], run["final_code"]) == ("FAILED", "EVIDENCE_INTEGRITY")


def test_gmsh_diffusion_retry_replays_from_pinned_mesh(specs_dir, tmp_path):
    source = specs_dir / "stage2"
    raw = json.loads((source / "diffusion_gmsh_p2_quad.json").read_bytes())
    raw["execution"]["mesh_levels"] = [4, 8, 16]
    raw["validation"]["max_l2_error_finest"] = 1e-4
    raw["policy"]["max_attempts"] = 3
    mesh = tmp_path / raw["external_mesh"]["file"]
    shutil.copyfile(source / mesh.name, mesh)
    spec = tmp_path / "study.json"
    spec.write_text(json.dumps(raw))
    run = controller.run_study(spec, tmp_path / "runs")
    assert run["final_state"] == "ACCEPTED", run["final_reason"]
    assert len(run["attempts"]) >= 2
    last = run["attempts"][-1]
    assert all("reused_from" in row for row in last["levels"][:-1])
    assert "reused_from" not in last["levels"][-1]
    root = Path(run["run_dir"])
    pin = sha256_file(root / "manifest.json")
    attempt = root / "attempts" / f"attempt_{last['attempt']:02d}"
    mesh.unlink()  # Replay depends only on the archive.
    replay = controller.reverify(attempt, pin)
    assert replay["verdict"] == "ACCEPTED" and replay["integrity"]["passed"]
    (root / controller.ARCHIVED_MESH).write_bytes(b"changed mesh")
    assert "EVIDENCE_INTEGRITY" in controller.reverify(attempt, pin)["failure_codes"]


def test_verification_timeout_fails_closed(specs_dir, tmp_path):
    spec = write_variant(specs_dir / "retry_demo_sin_sin_p1_tri.json", tmp_path / "timeout.json",
                         "policy", verifier_timeout_seconds=0.001, verifier_memory_mb=256)
    run = controller.run_study(spec, tmp_path)
    assert (run["final_state"], run["final_code"]) == ("FAILED", "VERIFIER_TIMEOUT")
    assert len(run["attempts"]) == 1
    assert {r["exit_meaning"] for r in run["attempts"][0]["levels"]} == {"OK"}


def test_anisotropic_mms_and_wrong_rhs(specs_dir, tmp_path):
    spec = specs_dir / "extensions/anisotropic_1000_q1.json"
    run = controller.run_study(spec, tmp_path)
    assert run["final_state"] == "ACCEPTED", run["final_reason"]
    summary = run["attempts"][-1]
    assert summary["orders"][-1]["l2_order"] == pytest.approx(2, abs=0.02)
    assert summary["orders"][-1]["h1_semi_order"] == pytest.approx(1, abs=0.02)
    assert summary["errors"][-1]["reference_rel_diff"] < 1e-7
    wrong = write_variant(spec, tmp_path / "wrong.json", "execution", inject_fault="wrong_source",
                          mesh_levels=[4, 8, 16])
    rejected = controller.run_study(wrong, tmp_path)
    assert rejected["final_state"] == "FAILED"
    assert "REFERENCE_MISMATCH" in rejected["attempts"][-1]["failure_codes"]
    untested = write_variant(spec, tmp_path / "partial.json", "execution", assembly="partial")
    blocked = controller.run_study(untested, tmp_path)
    assert blocked["final_code"] == "NOT_VERIFIED_COMBINATION" and not blocked["attempts"]


def test_search_obeys_frozen_accuracy_and_acceptance(specs_dir, tmp_path):
    study = load_study(specs_dir / "experiments/strategy_poisson.json")
    report = search(study, tmp_path / "search", 1)
    assert report["winner"] in {"q2-full-cg-jacobi", "q2-partial-cg-jacobi"}
    assert [row["feasible"] for row in report["candidates"]] == [False, False, True, True]
    assert all(row["trials"][0]["finest_errors"]["l2"] <= report["l2_target"]
               for row in report["candidates"] if row["feasible"])


def test_layered_archive_and_incorrect_material(specs_dir, tmp_path):
    source = specs_dir / "extensions/layered_unit_drop_10000.json"
    small = write_variant(source, tmp_path / "small.json", "execution", mesh_levels=[4, 8, 16])
    for fault in ("none", "wrong_diffusion", "wrong_source"):
        spec = write_variant(small, tmp_path / f"{fault}.json", "execution", inject_fault=fault)
        run = controller.run_study(spec, tmp_path)
        root = Path(run["run_dir"])
        attempt = root / "attempts/attempt_01"
        out = controller.reverify(attempt, sha256_file(root / "manifest.json"))
        assert out["integrity"]["passed"] and out["convergence"]["mode"] == "exact_in_q1"
        assert out["convergence"]["pairs"] == []
        assert all(lv["exit_code"] == 0 for lv in run["attempts"][0]["levels"])
        if fault == "none":
            assert out["verdict"] == "ACCEPTED"
            assert {c["name"] for c in out["levels"][-1]["checks"] if c["name"].startswith("layered_")} == {
                "layered_nodal_abs", "layered_flux_rel", "layered_jump_rel", "layered_energy_rel"}
        elif fault == "wrong_diffusion":
            assert out["verdict"] == "REJECTED"
            assert {"REFERENCE_MISMATCH", "LAYERED_JUMP_REL", "LAYERED_ENERGY_REL"} <= set(out["failure_codes"])
        else:
            assert out["verdict"] == "REJECTED"
            assert {"REFERENCE_MISMATCH", "LAYERED_FLUX_REL"} <= set(out["failure_codes"])
    unaligned = write_variant(source, tmp_path / "unaligned.json", "execution", mesh_levels=[5, 10, 20])
    blocked = controller.run_study(unaligned, tmp_path)
    assert blocked["final_code"] == "NOT_VERIFIED_COMBINATION" and not blocked["attempts"]


@pytest.mark.parametrize("assembly", ["full", "partial"])
def test_localized_gaussian_q2(specs_dir, tmp_path, assembly):
    source = specs_dir / "experiments/gaussian_offset_q2.json"
    spec = write_variant(source, tmp_path / "gaussian.json", "execution", assembly=assembly)
    run = controller.run_study(spec, tmp_path)
    assert run["final_state"] == "ACCEPTED", run["final_reason"]
    order = run["attempts"][-1]["orders"][-1]
    assert order["l2_order"] == pytest.approx(3, abs=0.05)
    assert order["h1_semi_order"] == pytest.approx(2, abs=0.05)


def test_swapped_neumann_label_preserves_reference_diagnostic(specs_dir, tmp_path):
    run = controller.run_study(specs_dir / "negative/swapped_boundary_labels.json", tmp_path)
    codes = set(run["attempts"][0]["failure_codes"])
    assert {"BOUNDARY_COVERAGE", "BOUNDARY_LABEL_MISMATCH", "DIRICHLET_VIOLATION", "REFERENCE_MISMATCH"} <= codes
