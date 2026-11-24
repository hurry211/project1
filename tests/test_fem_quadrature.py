"""Cross-check shared quadrature data against independently evaluated local shapes."""
import numpy as np
import pytest

from synthetic import write_level
from verifier.fem import (boundary_edge_owners, build_local_basis, element_quadrature,
                          eval_monomials, geometry_map, inverse_map, load_mesh_data,
                          reference_quadrature)


@pytest.mark.parametrize("geom,order", [("tri", 1), ("tri", 2), ("quad", 1), ("quad", 2)])
@pytest.mark.parametrize("permuted", [False, True])
def test_quadrature_agrees_for_every_local_order(tmp_path, geom, order, permuted):
    folder = tmp_path / "mesh"
    write_level(folder, geom, 2, order, np.random.default_rng(42) if permuted else None)
    mesh = load_mesh_data(folder)
    if geom == "quad":
        # A bilinear distorted cell has a quadrature-dependent Jacobian.
        refs = inverse_map(geom, mesh.verts, mesh.elem_node_xy)[0]
        mesh.verts[0, 2] += [0.0625, -0.03125]
        mesh.elem_node_xy[0] = geometry_map(geom, mesh.verts[:1], *refs.T)[0][0]
    basis = build_local_basis(mesh, order)
    rule = element_quadrature(mesh, basis, 5)
    xi, eta, w = reference_quadrature(geom, 5)
    m, dx, dy = eval_monomials(basis.exps, xi, eta)
    for e, verts in enumerate(mesh.verts):
        inv_nodes = inverse_map(geom, verts[None], mesh.elem_node_xy[e:e+1])[0]
        vander = eval_monomials(basis.exps, *inv_nodes.T)[0]
        coeffs = np.linalg.inv(vander)
        x, j = geometry_map(geom, verts[None], xi, eta)
        expected_ref = np.stack([dx @ coeffs, dy @ coeffs], axis=-1)
        expected_grad = expected_ref @ np.linalg.inv(j[0])
        np.testing.assert_allclose(rule.X[e], x[0], atol=1e-14)
        np.testing.assert_allclose(rule.phi[e], m @ coeffs, atol=1e-14)
        np.testing.assert_allclose(rule.gphi[e], expected_grad, atol=1e-14)
        np.testing.assert_allclose(rule.wdet[e], w * abs(np.linalg.det(j[0])), atol=1e-14)


def test_neumann_edge_lookup_preserves_missing_and_last_owner(tmp_path):
    folder = tmp_path / "mesh"
    write_level(folder, "tri", 2, 1)
    mesh = load_mesh_data(folder)
    # Include a shared interior edge, a reversed physical edge, and a misplaced
    # segment. The edge map must preserve the result of exhaustive enumeration.
    mesh.bdr_xy[:3] = [mesh.verts[0, [2, 0]], mesh.bdr_xy[1, ::-1], [[0.1, 0.2], [0.2, 0.3]]]
    mesh.bdr_attr[:3] = 7
    expected = {}
    for segment in mesh.bdr_xy[mesh.bdr_attr == 7]:
        key = frozenset(map(tuple, segment))
        for e, tri in enumerate(mesh.verts):
            for a, b in zip(tri, np.roll(tri, -1, axis=0)):
                if frozenset((tuple(a), tuple(b))) == key:
                    expected[key] = e
    assert len(expected) == 2
    assert boundary_edge_owners(mesh, {7, 99}) == expected
