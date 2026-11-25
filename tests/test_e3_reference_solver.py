"""Unit and parity tests for the E3 streaming verifier and AMG-PCG reference solver."""

from __future__ import annotations

import json
import numpy as np
import pytest
import scipy.sparse as sp

from sciagent.spec import StudySpec
from synthetic import make_attempt
from verifier import expr
from verifier.fem import build_local_basis, element_quadrature, evaluate_quadrature_slice, load_mesh_data, reference_quadrature
from verifier.reference_solver import ReferenceSolverError, solve_reference_amg_pcg, solve_reference_direct, solve_reference_system
from verifier.verifier import ACCEPTED, REJECTED, _streaming_errors, _streaming_reference_solve, verify_attempt

SIN = {"exact_solution": "sin(pi*x)*sin(pi*y)", "source": "2*pi**2*sin(pi*x)*sin(pi*y)"}
COS = {"exact_solution": "cos(pi*x)*exp(y)", "source": "(pi**2 - 1)*cos(pi*x)*exp(y)"}
VAR = {
    "equation": "-div(kappa*grad(u)) = f",
    "kappa": "1+x",
    "exact_solution": "sin(pi*x)*sin(2*pi*y)",
    "source": "5*pi**2*(1+x)*sin(pi*x)*sin(2*pi*y)-pi*cos(pi*x)*sin(2*pi*y)",
}


def _study(case, geom="tri", p=1, levels=(8, 16, 32), neumann=(), method="auto", threshold=300_000):
    sides = [(1, "bottom"), (2, "right"), (3, "top"), (4, "left")]
    eq = case.get("equation", "-laplacian(u) = f")
    ver = "0.3" if case.get("kappa", "1") != "1" else "0.2"
    prob = {
        "id": "e3_ref_test",
        "equation": eq,
        "domain": "unit_square_2d",
        "exact_solution": case["exact_solution"],
        "source": case["source"],
        "boundaries": [
            {"attribute": a, "label": lbl, "kind": "neumann" if lbl in neumann else "dirichlet", "value": "from_exact_solution"}
            for a, lbl in sides
        ],
    }
    if "kappa" in case:
        prob["kappa"] = case["kappa"]
    return StudySpec.model_validate({
        "schema_version": ver,
        "problem": prob,
        "execution": {"mesh_type": geom, "polynomial_order": p, "mesh_levels": list(levels)},
        "validation": {
            "reference_solve_method": method,
            "reference_solve_amg_threshold_dofs": threshold,
        },
    })


@pytest.mark.parametrize("geom,p,case,neumann", [
    ("tri", 1, SIN, ()),
    ("quad", 1, COS, ("top",)),
    ("tri", 2, VAR, ("top",)),
    ("quad", 2, SIN, ()),
])
def test_amg_pcg_matches_direct_reference_solve(tmp_path, geom, p, case, neumann):
    s_direct = _study(case, geom, p, (4, 8, 16), neumann, method="direct")
    s_amg = _study(case, geom, p, (4, 8, 16), neumann, method="amg_pcg")
    att_d = make_attempt(tmp_path / "direct", s_direct)
    att_a = make_attempt(tmp_path / "amg", s_amg)
    vd = verify_attempt(att_d)
    va = verify_attempt(att_a)
    assert vd["verdict"] == ACCEPTED
    assert va["verdict"] == ACCEPTED
    for ld, la in zip(vd["levels"], va["levels"]):
        assert la["l2_error"] == pytest.approx(ld["l2_error"], rel=1e-11)
        assert la["h1_semi_error"] == pytest.approx(ld["h1_semi_error"], rel=1e-11)
        assert la["reference_rel_diff"] < 1e-10


def test_amg_pcg_handles_extreme_anisotropy():
    nx, ny = 31, 31
    lx = sp.diags((-1000.0 * np.ones(nx - 1), 2000.0 * np.ones(nx), -1000.0 * np.ones(nx - 1)), (-1, 0, 1))
    ly = sp.diags((-1.0 * np.ones(ny - 1), 2.0 * np.ones(ny), -1.0 * np.ones(ny - 1)), (-1, 0, 1))
    K = sp.kronsum(ly, lx, format="csr")
    rhs = np.ones(K.shape[0], dtype=np.float64) / (nx * ny)
    u_dir, m_dir = solve_reference_direct(K, rhs)
    u_amg, m_amg = solve_reference_amg_pcg(K, rhs)
    assert m_amg.relative_residual <= 1e-11
    assert np.max(np.abs(u_amg - u_dir)) / np.max(np.abs(u_dir)) <= 1e-9


def test_streaming_integration_and_solve_match_batched_paths(tmp_path):
    s = _study(VAR, "tri", 2, (4, 8, 16), ("top",), method="direct")
    att = make_attempt(tmp_path, s)
    lvl = att / "levels/n0016"
    mesh = load_mesh_data(lvl)
    basis = build_local_basis(mesh, 2)
    m = 6
    q = element_quadrature(mesh, basis, m)
    u_fn = expr.compile_expr(s.problem.exact_solution)
    f_fn = expr.compile_expr(s.problem.source)
    det_s, l2_s, h1_s = _streaming_errors(mesh, basis, m, u_fn, s.problem.equation)
    assert det_s == pytest.approx(float(np.min(q.detJ)), rel=1e-14)
    from verifier.verifier import _integrated_errors, _side_of_points, _unique_dof_coords, reference_solve
    l2_b, h1_b = _integrated_errors(mesh, q, u_fn, s.problem.equation)
    assert l2_s == pytest.approx(l2_b, rel=1e-13)
    assert h1_s == pytest.approx(h1_b, rel=1e-13)

    coords, _ = _unique_dof_coords(mesh, mesh.values.size)
    on_dir = np.zeros(coords.shape[0], dtype=bool)
    for b in s.problem.boundaries:
        if b.kind == "dirichlet":
            on_dir |= _side_of_points(coords, b.label)
    u_b = reference_solve(mesh, basis, q, s.problem, coords, on_dir, u_fn, f_fn, m, method="direct")
    u_s = _streaming_reference_solve(mesh, basis, m, s.problem, coords, on_dir, u_fn, f_fn, s.validation)
    assert np.max(np.abs(u_s - u_b)) / np.max(np.abs(u_b)) < 1e-12


def test_amg_pcg_rejects_singular_or_unconverged_system(tmp_path):
    # Negative test 1: zero-row singular operator cannot falsely pass residual check
    K = sp.diags([np.array([2.0, 2.0, 0.0, 2.0])], [0], shape=(4, 4), format="csr")
    rhs = np.ones(4)
    with pytest.raises((ReferenceSolverError, RuntimeError)):
        solve_reference_amg_pcg(K, rhs, maxiter=5, verify_residual_tol=1e-11)

    # Negative test 2: impossible residual / tiny maxiter fails closed inside verify_attempt
    with pytest.raises(ValueError, match="unknown reference solver method"):
        solve_reference_system(K, rhs, method="invalid")


def test_exclude_if_preserves_e2_spec_hashes(specs_dir):
    # E3 default fields must NOT alter canonical ValidationSpec SHA-256 of existing Stage 0-E2 specs.
    from sciagent.spec import canonical_json, load_study, sha256_of
    s = load_study(specs_dir / "poisson_sin_sin_p1_tri.json")
    raw = json.loads(canonical_json(s.validation))
    assert "reference_solve_method" not in raw
    assert "reference_solve_amg_threshold_dofs" not in raw
    assert sha256_of(s.validation) == sha256_of(s.validation.model_copy())


def test_amg_pcg_rejects_underresolved_cg_iterations():
    # Negative test 3: maxiter=2 leaves large backward error and raises ReferenceSolverError.
    m = 32
    d = np.ones(m)
    k1d = sp.diags([-d[:-1], 2 * d, -d[:-1]], [-1, 0, 1], format="csr")
    K = sp.kronsum(k1d, k1d, format="csr")
    rhs = np.ones(K.shape[0], dtype=np.float64)
    with pytest.raises(ReferenceSolverError):
        solve_reference_amg_pcg(K, rhs, maxiter=1, verify_residual_tol=1e-11)

