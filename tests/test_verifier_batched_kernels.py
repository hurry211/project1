"""Exact regression checks for bounded integration and local-matrix reuse."""
from dataclasses import replace

import numpy as np
import pytest

from verifier import expr, verifier
from verifier.fem import (MeshData, build_local_basis, element_quadrature,
                          eval_monomials, repeated_element_period)


def triangle_strip():
    # Each square uses the two opposite orientations found in native exports.
    corners = np.array([[[0, 0], [1, 1], [0, 1]],
                        [[1, 1], [0, 0], [1, 0]]], dtype=float)
    verts = np.concatenate([corners + [i, 0] for i in range(7)])
    dofs = np.arange(3 * len(verts)).reshape(-1, 3)
    return MeshData("tri", verts, dofs, verts.copy(), np.cos(dofs.ravel()),
                    np.empty(0, dtype=int), np.empty((0, 2, 2)))


def assert_same_bits(result, expected):
    np.testing.assert_array_equal(np.asarray(result).view(np.uint64),
                                  np.asarray(expected).view(np.uint64))


def test_basis_uses_actual_nodes_and_signed_zero():
    mesh = triangle_strip()
    zeros = np.array([[0.0], [-0.0]] * 3)
    assert repeated_element_period(zeros) == 2
    zeros[-1] = 0.0
    assert repeated_element_period(zeros) == len(zeros)
    for permute_last in (False, True):
        if permute_last:
            mesh.elem_node_xy[-1] = mesh.elem_node_xy[-1, [2, 0, 1]]
        basis = build_local_basis(mesh, 1)
        ref = basis.node_ref
        if permute_last:
            assert repeated_element_period(ref) == len(ref)
        full, _, _ = eval_monomials(basis.exps, ref[..., 0], ref[..., 1])
        assert_same_bits(basis.coeffs, np.linalg.inv(full))
        assert_same_bits(basis.max_cond, np.linalg.cond(full).max())


@pytest.mark.parametrize("variation", ["unaltered", "last_weight", "last_gradient",
                                       "pointwise_gradient", "variable_coefficient"])
def test_batched_forms_and_error_norms_match_full_integrals(monkeypatch, variation):
    mesh = triangle_strip()
    rule = element_quadrature(mesh, build_local_basis(mesh, 1), 4)
    if variation == "last_weight":
        rule.wdet[-1, 4] *= 1.125
    elif variation == "last_gradient":
        grad = rule.gphi[:, :1].copy()
        grad[-1] *= 1.25
        rule = replace(rule, gphi=np.broadcast_to(grad, rule.gphi.shape))
    elif variation == "pointwise_gradient":
        grad = rule.gphi.copy()
        grad[-1, 12] *= 1.25
        rule = replace(rule, gphi=grad)

    # Exercise the final short batch and an interior boundary between the two orientations.
    monkeypatch.setattr(verifier, "_quadrature_batches",
                        lambda q: (slice(i, min(i + 5, len(q.X))) for i in range(0, len(q.X), 5)))
    u = expr.compile_expr("sin(pi*x)*sin(pi*y)")
    f = expr.compile_expr("exp(x-y)")
    kappa = expr.compile_expr("1+x") if variation == "variable_coefficient" else None
    x, y = rule.X[..., 0], rule.X[..., 1]
    weights = rule.wdet * kappa(x, y) if kappa else rule.wdet
    ke, fe = verifier._volume_forms(rule, kappa, f)
    assert_same_bits(ke, np.einsum("eq,eqid,eqjd->eij", weights, rule.gphi, rule.gphi))
    assert_same_bits(fe, np.einsum("eq,eq,eqi->ei", rule.wdet, f(x, y), rule.phi))

    U = mesh.values[mesh.elem_dofs]
    uh = np.einsum("eqk,ek->eq", rule.phi, U)
    guh = np.einsum("eqki,ek->eqi", rule.gphi, U)
    gx, gy = expr.gradient(u, x, y)
    expected = (np.sqrt(np.sum(rule.wdet * (uh - u(x, y)) ** 2)),
                np.sqrt(np.sum(rule.wdet * ((guh[..., 0] - gx) ** 2 + (guh[..., 1] - gy) ** 2))))
    assert_same_bits(verifier._integrated_errors(mesh, rule, u), expected)
