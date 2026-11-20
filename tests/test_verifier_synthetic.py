"""Unit tests for the independent verifier on synthetic (non-MFEM) artifacts.

These tests validate the verifier itself: its own Galerkin implementation must
reproduce textbook convergence rates, be invariant to dof numbering, and each
corruption must be rejected with the right failure code.
"""

import json

import numpy as np
import pytest

from sciagent.spec import StudySpec
from synthetic import make_attempt
from verifier.verifier import ACCEPTED, REJECTED, RETRY_ELIGIBLE, verify_attempt

SIN = {"exact_solution": "sin(pi*x)*sin(pi*y)", "source": "2*pi**2*sin(pi*x)*sin(pi*y)"}
COS = {"exact_solution": "cos(pi*x)*exp(y)", "source": "(pi**2 - 1)*cos(pi*x)*exp(y)"}


def study(case, geom="tri", p=1, levels=(8, 16, 32), neumann=(), validation=None):
    sides = [(1, "bottom"), (2, "right"), (3, "top"), (4, "left")]
    return StudySpec.model_validate({
        "schema_version": "0.2",
        "problem": {"id": "synthetic", "equation": "-laplacian(u) = f", "domain": "unit_square_2d", **case,
                    "boundaries": [{"attribute": a, "label": l, "kind": "neumann" if l in neumann else "dirichlet",
                                    "value": "from_exact_solution"} for a, l in sides]},
        "execution": {"mesh_type": geom, "polynomial_order": p, "mesh_levels": list(levels)},
        "validation": validation or {},
    })


@pytest.mark.parametrize("geom,p,case,neumann,levels", [
    ("tri", 1, SIN, (), (8, 16, 32)),
    ("quad", 1, SIN, (), (8, 16, 32)),
    ("tri", 2, SIN, (), (4, 8, 16)),
    ("quad", 2, COS, ("top",), (4, 8, 16)),
    ("tri", 1, COS, ("top", "right"), (8, 16, 32)),
])
def test_verifier_own_fem_converges(tmp_path, geom, p, case, neumann, levels):
    s = study(case, geom, p, levels, neumann)
    v = verify_attempt(make_attempt(tmp_path, s))
    assert v["verdict"] == ACCEPTED, v["failure_codes"]
    fin = v["convergence"]["pairs"][-1]
    assert abs(fin["l2_order"] - (p + 1)) < 0.2 and abs(fin["h1_semi_order"] - p) < 0.2


def test_numbering_invariance(tmp_path):
    s = study(SIN, "tri", 2, (4, 8, 16))
    a = verify_attempt(make_attempt(tmp_path / "a", s))
    b = verify_attempt(make_attempt(tmp_path / "b", s, shuffle=True))
    for la, lb in zip(a["levels"], b["levels"]):
        assert la["l2_error"] == pytest.approx(lb["l2_error"], rel=1e-10)
        assert la["h1_semi_error"] == pytest.approx(lb["h1_semi_error"], rel=1e-10)
    assert b["verdict"] == ACCEPTED


def test_lying_self_report_rejected(tmp_path):
    v = verify_attempt(make_attempt(tmp_path, study(SIN), self_report="zero"))
    assert v["verdict"] == REJECTED and "SELF_REPORT_MISMATCH" in v["failure_codes"]


def test_dirichlet_violation_rejected(tmp_path):
    def mutate(n, vals, xy):
        on_left = np.abs(xy[:, 0]) < 1e-12
        vals[np.where(on_left)[0][len(np.where(on_left)[0]) // 2]] += 1e-3
        return vals
    v = verify_attempt(make_attempt(tmp_path, study(SIN), mutate=mutate))
    assert v["verdict"] == REJECTED
    assert "DIRICHLET_VIOLATION" in v["failure_codes"] and "REFERENCE_MISMATCH" in v["failure_codes"]


def test_interior_perturbation_caught_by_reference_solve(tmp_path):
    def mutate(n, vals, xy):
        i = int(np.argmin(np.sum((xy - 0.5) ** 2, axis=1)))
        vals[i] *= 1.0 + 1e-5
        return vals
    v = verify_attempt(make_attempt(tmp_path, study(SIN), mutate=mutate))
    assert v["verdict"] == REJECTED and v["failure_codes"] == ["REFERENCE_MISMATCH"]


def test_nan_rejected(tmp_path):
    def mutate(n, vals, xy):
        vals[len(vals) // 3] = np.nan
        return vals
    v = verify_attempt(make_attempt(tmp_path, study(SIN), mutate=mutate))
    assert v["verdict"] == REJECTED and "NON_FINITE_SOLUTION" in v["failure_codes"]


def test_accuracy_target_gives_retry_eligible(tmp_path):
    v = verify_attempt(make_attempt(tmp_path, study(SIN, validation={"max_l2_error_finest": 1e-8})))
    assert v["verdict"] == RETRY_ELIGIBLE and v["failure_codes"] == ["ACCURACY_NOT_MET"]


def test_missing_artifact_rejected(tmp_path):
    att = make_attempt(tmp_path, study(SIN))
    (att / "levels" / "n0016" / "dof_values.txt").unlink()
    v = verify_attempt(att)
    assert v["verdict"] == REJECTED and "MALFORMED_ARTIFACT" in v["failure_codes"]


def test_claim_mismatch_rejected(tmp_path):
    att = make_attempt(tmp_path, study(SIN))
    rp = att / "levels" / "n0008" / "result.json"
    r = json.loads(rp.read_text()); r["case_source"] = "1"; rp.write_text(json.dumps(r))
    v = verify_attempt(att)
    assert v["verdict"] == REJECTED and "CLAIM_SPEC_MISMATCH" in v["failure_codes"]


def test_solver_success_flag_is_not_sufficient(tmp_path):
    """A run that claims convergence but whose data is the exact interpolant
    (not the Galerkin solution) is rejected by the reference solve."""
    from verifier import expr
    u = expr.compile_expr(SIN["exact_solution"])

    def mutate(n, vals, xy):
        return u(xy[:, 0], xy[:, 1])
    v = verify_attempt(make_attempt(tmp_path, study(SIN), mutate=mutate))
    assert v["verdict"] == REJECTED and "REFERENCE_MISMATCH" in v["failure_codes"]


def test_unconverged_flag_rejected(tmp_path):
    att = make_attempt(tmp_path, study(SIN))
    rp = att / "levels" / "n0032" / "result.json"
    r = json.loads(rp.read_text()); r["solver_converged"] = False; rp.write_text(json.dumps(r))
    v = verify_attempt(att)
    assert v["verdict"] == REJECTED and "SOLVER_NOT_CONVERGED" in v["failure_codes"]
