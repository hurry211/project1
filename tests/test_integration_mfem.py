"""Integration tests against the real MFEM executable (skipped if not built).

Each test runs the full controller (spec -> freeze -> env -> capability ->
MFEM runs -> independent verification -> manifest) into a temporary runs/
directory and asserts the final state and failure codes.
"""

import json
import subprocess

import pytest

from conftest import requires_mfem
from sciagent import controller
from sciagent.mfem_runner import list_cases, solver_binary

pytestmark = [pytest.mark.mfem, requires_mfem]


def run(specs_dir, rel, tmp_path):
    return controller.run_study(specs_dir / rel, tmp_path / "runs")


@pytest.mark.parametrize("spec", [
    "poisson_sin_sin_p1_tri.json",
    "poisson_sin_sin_p2_tri.json",
    "poisson_sin_sin_p1_tri_jacobi.json",
    "poisson_cos_exp_mixed_p1_quad.json",
    "poisson_cos_exp_mixed_p2_quad.json",
    "poisson_cos_exp_mixed_p1_quad_pa.json",
    "poisson_cos_exp_mixed_p2_quad_pa.json",
])
def test_positive_studies_accepted(specs_dir, tmp_path, spec):
    m = run(specs_dir, spec, tmp_path)
    assert m["final_state"] == "ACCEPTED", (m["final_code"], m["final_reason"])
    att = m["attempts"][-1]
    assert att["verdict"] == "ACCEPTED"
    for e in att["errors"]:
        assert e["reference_rel_diff"] is not None and e["reference_rel_diff"] < 1e-7


@pytest.mark.parametrize("spec,state,code", [
    ("negative/unsupported_backend_mpi.json", "BLOCKED", "UNSUPPORTED_BACKEND"),
    ("negative/unsupported_backend_cuda.json", "BLOCKED", "UNSUPPORTED_BACKEND"),
    ("negative/unsupported_solver_umfpack.json", "FAILED", "UNSUPPORTED_SOLVER"),
    ("negative/unsupported_pa_with_gs.json", "FAILED", "UNSUPPORTED_COMBINATION"),
    ("negative/not_verified_order3.json", "FAILED", "NOT_VERIFIED_COMBINATION"),
    ("negative/unbound_case.json", "FAILED", "UNSUPPORTED_PROBLEM"),
])
def test_rejected_before_running(specs_dir, tmp_path, spec, state, code):
    m = run(specs_dir, spec, tmp_path)
    assert (m["final_state"], m["final_code"]) == (state, code)
    assert m["attempts"] == []


@pytest.mark.parametrize("spec,expected_codes", [
    ("negative/swapped_boundary_labels.json", {"BOUNDARY_LABEL_MISMATCH"}),
    ("negative/unassigned_mesh_attribute.json", {"MISSING_ARTIFACT"}),
    ("negative/fault_wrong_source.json", {"REFERENCE_MISMATCH"}),
    ("negative/fault_drop_dirichlet_attr.json", {"DIRICHLET_VIOLATION", "REFERENCE_MISMATCH"}),
    ("negative/fault_nan_value.json", {"NON_FINITE_SOLUTION"}),
    ("negative/fault_early_stop_lie.json", {"REFERENCE_MISMATCH"}),
    ("negative/fault_self_report_lie.json", {"SELF_REPORT_MISMATCH"}),
    ("negative/preasymptotic_strict_band.json", {"CONVERGENCE_ORDER_OUT_OF_BAND", "SELF_REPORT_MISMATCH"}),
])
def test_wrong_results_rejected_by_verifier(specs_dir, tmp_path, spec, expected_codes):
    m = run(specs_dir, spec, tmp_path)
    assert m["final_state"] == "FAILED", m["final_reason"]
    codes = set(m["attempts"][-1]["failure_codes"])
    assert expected_codes <= codes, codes


def test_order_band_gate_alone(specs_dir, tmp_path):
    m = run(specs_dir, "negative/strict_band_resolved.json", tmp_path)
    assert m["final_state"] == "FAILED"
    assert m["attempts"][-1]["failure_codes"] == ["CONVERGENCE_ORDER_OUT_OF_BAND"]


def test_coarsest_level_quadrature_finding(specs_dir, tmp_path):
    """n=1 triangles: all nodes are Dirichlet, u_h == 0, so the exact L2 error
    is ||sin(pi x) sin(pi y)||_L2 = 1/2. The verifier must be close to 1/2."""
    run(specs_dir, "negative/preasymptotic_strict_band.json", tmp_path)
    att = next((tmp_path / "runs").glob("*/attempts/attempt_01"))
    lv = json.loads((att / "verdict.json").read_text())["levels"][0]
    assert lv["n"] == 1
    assert abs(lv["l2_error"] - 0.5) < 1e-4


def test_fault_early_stop_claims_success(specs_dir, tmp_path):
    """The lying solver exits 0 and claims convergence; the verifier still rejects."""
    m = run(specs_dir, "negative/fault_early_stop_lie.json", tmp_path)
    assert all(lv["exit_meaning"] == "OK" for lv in m["attempts"][0]["levels"])
    lvl = next(iter((tmp_path / "runs").glob("*/attempts/attempt_01/levels/n0008/result.json")))
    assert json.loads(lvl.read_text())["solver_converged"] is True
    assert m["final_state"] == "FAILED"


def test_bounded_retry_reaches_target(specs_dir, tmp_path):
    m = run(specs_dir, "retry_demo_sin_sin_p1_tri.json", tmp_path)
    assert m["final_state"] == "ACCEPTED"
    assert len(m["attempts"]) >= 2
    assert all(a["verdict"] == "RETRY_ELIGIBLE" for a in m["attempts"][:-1])
    for a, b in zip(m["attempts"], m["attempts"][1:]):
        assert b["mesh_levels"] == a["mesh_levels"] + [2 * a["mesh_levels"][-1]]
    run_dir = next((tmp_path / "runs").iterdir())
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    retry = [e for e in events if e["state"] == "RETRY_ALLOWED"]
    assert len(retry) == len(m["attempts"]) - 1
    assert all(e["unchanged"] == ["problem", "validation", "policy"] for e in retry)
    # every failed attempt preserved with its own verdict
    assert len(list(run_dir.glob("attempts/attempt_*/verdict.json"))) == len(m["attempts"])


def test_budget_exhausted(specs_dir, tmp_path):
    m = run(specs_dir, "budget_demo_sin_sin_p1_tri.json", tmp_path)
    assert (m["final_state"], m["final_code"]) == ("BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED")
    assert len(m["attempts"]) == 2


def test_reverify_from_artifacts_is_deterministic(specs_dir, tmp_path):
    m = run(specs_dir, "poisson_sin_sin_p1_tri.json", tmp_path)
    att = next((tmp_path / "runs").glob("*/attempts/attempt_01"))
    stored = json.loads((att / "verdict.json").read_text())
    again = controller.reverify(att)
    assert again["verdict"] == stored["verdict"] == "ACCEPTED"
    assert [lv["l2_error"] for lv in again["levels"]] == [lv["l2_error"] for lv in stored["levels"]]


def test_tampered_artifact_detected_on_reverify(specs_dir, tmp_path):
    run(specs_dir, "poisson_sin_sin_p1_tri.json", tmp_path)
    att = next((tmp_path / "runs").glob("*/attempts/attempt_01"))
    f = att / "levels" / "n0032" / "dof_values.txt"
    lines = f.read_text().splitlines()
    i, v = lines[500].split()
    lines[500] = f"{i} {float(v) + 1e-6!r}"
    f.write_text("\n".join(lines) + "\n")
    again = controller.reverify(att)
    assert again["verdict"] == "REJECTED" and "REFERENCE_MISMATCH" in again["failure_codes"]


# ---- direct binary contract tests (defense in depth inside the C++) --------

def _bin(tmp_path, *args):
    return subprocess.run([str(solver_binary()), *args, "--out", str(tmp_path)], capture_output=True, text=True)


def test_binary_rejects_incomplete_boundary_partition(tmp_path):
    p = _bin(tmp_path, "--dirichlet", "1,2,3", "--neumann", "none")
    assert p.returncode == 3 and "BOUNDARY_COVERAGE" in p.stderr


def test_binary_rejects_overlapping_boundary_sets(tmp_path):
    p = _bin(tmp_path, "--dirichlet", "1,2,3,4", "--neumann", "4")
    assert p.returncode == 3


def test_binary_rejects_pure_neumann(tmp_path):
    p = _bin(tmp_path, "--dirichlet", "none", "--neumann", "1,2,3,4")
    assert p.returncode == 3 and "nullspace" in p.stderr


def test_binary_rejects_pa_with_gs(tmp_path):
    p = _bin(tmp_path, "--assembly", "partial", "--solver", "cg_gs")
    assert p.returncode == 4 and "UNSUPPORTED_COMBINATION" in p.stderr


def test_binary_reports_nonconvergence(tmp_path):
    p = _bin(tmp_path, "--n", "32", "--max-iter", "2")
    assert p.returncode == 5
    assert json.loads((tmp_path / "result.json").read_text())["solver_converged"] is False


@pytest.mark.parametrize("obstruction", ["missing_directory", "result.json"])
def test_binary_reports_output_failure(tmp_path, obstruction):
    if obstruction == "result.json":
        (tmp_path / obstruction).mkdir()
        output = tmp_path
    else:
        output = tmp_path / obstruction
    p = _bin(output, "--n", "4")
    assert p.returncode == 7 and "cannot write" in p.stderr


def test_list_cases_matches_header():
    ids = {c["id"] for c in list_cases()["cases"]}
    assert ids == {"sin_sin", "cos_exp"}
