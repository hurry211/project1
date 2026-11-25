import json
from pathlib import Path
import subprocess

import pytest

from conftest import requires_mfem
from sciagent import capability, controller
from sciagent.environment import mfem_install_dir


def test_original_coverage_cohort_remains_replayable():
    saved = json.loads((Path(__file__).resolve().parents[1] / capability.REPORT).read_bytes())
    original = {(r["mesh_type"], r["order"], r["assembly"], r["solver"]) for r in saved["rows"].values()}
    assert capability.STAGE2_CONFIGURATIONS == original
    assert capability.TESTED - original == {("quad", 1, "full", "cg_jacobi"), ("quad", 2, "full", "cg_jacobi")}


@pytest.mark.mfem
@requires_mfem
@pytest.mark.parametrize("order", (1, 2))
def test_new_full_jacobi_control_rejects_wrong_source(specs_dir, tmp_path, order):
    raw = json.loads((specs_dir / "poisson_cos_exp_mixed_p1_quad.json").read_bytes())
    raw["execution"].update(solver="cg_jacobi", polynomial_order=order, mesh_levels=[8, 16, 32])
    path = tmp_path / "study.json"
    path.write_text(json.dumps(raw))
    valid = controller.run_study(path, tmp_path / "studies")
    assert valid["final_state"] == "ACCEPTED"
    raw["execution"]["inject_fault"] = "wrong_source"
    path.write_text(json.dumps(raw))
    rejected = controller.run_study(path, tmp_path / "studies")
    assert rejected["final_state"] == "FAILED"
    assert "REFERENCE_MISMATCH" in rejected["attempts"][0]["failure_codes"]
    assert all(level["exit_code"] == 0 for level in rejected["attempts"][0]["levels"])


@pytest.mark.mfem
@requires_mfem
def test_full_and_partial_jacobi_action_match_assembled_diagonal(tmp_path):
    source = Path(__file__).parent / "native"
    subprocess.run(["cmake", "-S", str(source), "-B", str(tmp_path),
                    f"-DMFEM_DIR={mfem_install_dir() / 'lib/cmake/mfem'}"], check=True)
    subprocess.run(["cmake", "--build", str(tmp_path), "--target", "assembly_diagonal"], check=True)
    probe = subprocess.run([str(tmp_path / "assembly_diagonal")], check=True, capture_output=True, text=True)
    assert len(probe.stdout.splitlines()) == 8
