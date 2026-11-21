from types import SimpleNamespace

import numpy as np
import pytest

from verifier.fem import cartesian_partition, lattice_indices


SQUARE = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])


@pytest.mark.parametrize("cells", [[[2, 3, 0], [1, 2, 0]], [[3, 1, 2], [1, 3, 0]]])
def test_complementary_triangles_allow_either_diagonal_and_rotations(cells):
    mesh = SimpleNamespace(verts=SQUARE[cells], geom="tri")
    assert cartesian_partition(mesh, 1, 1e-12)


def test_triangle_pair_with_correct_total_area_can_overlap():
    mesh = SimpleNamespace(verts=SQUARE[[[0, 1, 2], [0, 1, 3]]], geom="tri")
    assert not cartesian_partition(mesh, 1, 1e-12)


def test_duplicated_square_with_unchanged_element_count():
    squares = np.array([(SQUARE + [i, j]) / 2 for j in range(2) for i in range(2)])
    mesh = SimpleNamespace(verts=squares, geom="quad")
    assert cartesian_partition(mesh, 2, 1e-12)
    squares[2] = squares[1]
    assert not cartesian_partition(mesh, 2, 1e-12)


def test_interior_displacement_with_fixed_boundary():
    squares = np.array([(SQUARE + [i, j]) / 2 for j in range(2) for i in range(2)])
    centers = np.all(squares == [.5, .5], axis=2)
    squares[centers, 0] += 2e-10
    assert not cartesian_partition(SimpleNamespace(verts=squares, geom="quad"), 2, 1e-12)
    assert lattice_indices(np.array([[.5 + 2e-10, .5]]), 4, 1e-12) is None
    assert np.array_equal(lattice_indices(np.array([[.5 + 2e-13, .5]]), 4, 1e-12), [[2, 2]])
