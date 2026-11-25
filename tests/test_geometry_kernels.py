"""Local parity controls on adjacent cells with unequal geometry and material."""
import numpy as np
import pytest

from benchmarks.geometry_kernels import compare_matrices
from verifier import expr, verifier
from verifier.fem import MeshData, build_local_basis, element_quadrature


@pytest.mark.parametrize("geom", ["tri", "quad"])
@pytest.mark.parametrize("coefficient", ["1", "1+x"])
def test_stream_kernel_respects_each_affine_or_distorted_cell(geom, coefficient):
    if geom == "tri":
        verts = np.array([[[0, 0], [1, 0], [0, 1]], [[1, 0], [2, .2], [1.1, .8]]], dtype=float)
    else:
        verts = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]],
                          [[1, 0], [2, .1], [1.85, 1.1], [1, 1]]], dtype=float)
    dofs = np.arange(verts.shape[0] * verts.shape[1]).reshape(verts.shape[:2])
    mesh = MeshData(geom, verts, dofs, verts.copy(), np.zeros(dofs.size),
                    np.empty(0, dtype=int), np.empty((0, 2, 2)))
    q = element_quadrature(mesh, build_local_basis(mesh, 1), 6)
    kappa = None if coefficient == "1" else expr.compile_expr(coefficient)
    full = verifier._stiffness_block(q.gphi, q.wdet, q.X[..., 0], q.X[..., 1], kappa, verifier.POISSON)
    actual = verifier._stream_stiffness(q, kappa, verifier.POISSON)
    assert compare_matrices(full, actual) < 64 * np.finfo(float).eps
    with pytest.raises(ValueError, match="parity failed"):
        compare_matrices(full, np.broadcast_to(actual[:1], actual.shape))


def test_parity_checks_small_local_matrix_even_beside_large_one():
    expected = np.array([np.eye(2) * 1e6, np.eye(2) * 1e-12])
    actual = expected.copy()
    actual[1, 1, 1] *= 1.01
    with pytest.raises(ValueError, match="parity failed"):
        compare_matrices(expected, actual)
