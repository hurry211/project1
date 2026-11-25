"""Closed form and independently checked bounds for offline P2 witness evaluation."""
from fractions import Fraction as Q

import pytest

from cfac.checker import witness_lower_sq
from cfac.evaluation import optimal_fields
from cfac.prototype import ReferenceSystem, uniform_mesh
from cfac.spec import Problem
from research.cfac.futile_limits import missed_fields, quadratic_witness, rt1_bound


def test_two_triangles_have_one_free_quadratic_witness_dof():
    p = Problem(uniform_mesh(1), [1, 1], 1, Q(1, 8), 16)
    system = ReferenceSystem(p)
    primal = system.zero_primal()
    vertices, midpoints = quadratic_witness(system, primal)
    # Only the diagonal edge basis is free. Its load is 1/3 and stiffness 16/3.
    assert not any(vertices)
    assert [v for v in midpoints if v] == [Q(1, 16)]
    assert witness_lower_sq(p, p.digest(), primal, vertices, midpoints) == Q(1, 48)


def test_sign_reversal_preserves_witness_energy_and_verified_enclosure():
    results = []
    for sign in (1, -1):
        p = Problem(uniform_mesh(2), [sign * v for v in (2, -1)*4], 2, Q(1, 20), 32)
        system = ReferenceSystem(p)
        primal, _ = optimal_fields(system)
        vertices, midpoints = quadratic_witness(system, primal)
        lower = witness_lower_sq(p, p.digest(), primal, vertices, midpoints)
        upper = Q(rt1_bound(p, primal, system.feasible_flux)["upper_sq"])
        assert 0 < lower <= upper
        results.append((lower, upper))
    assert results[0] == results[1]


def test_miss_extraction_requires_pinned_measurement(tmp_path):
    path = tmp_path / "other-receipt.gz"
    path.write_bytes(b"another measurement")
    with pytest.raises(ValueError, match="digest"):
        list(missed_fields(path))
