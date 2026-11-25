"""Exact discrete controls for the reconstruction-based stopping experiment."""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from cfac.checker import _geometry
from cfac.evaluation import optimal_fields, primal_energy, square_energy_interval
from cfac.prototype import ReferenceSystem, uniform_mesh
from cfac.spec import Mesh, Problem
from cfac.stopping import _tree_lift, inspect_stopping


def signed_problem():
    mesh = uniform_mesh(3)
    vertices = list(mesh.vertices)
    vertices[5] = (Q(2, 7), Q(3, 8))
    mesh = Mesh(vertices, mesh.triangles)
    loads = tuple((-1)**i * (i + 1) for i in range(len(mesh.triangles)))
    return Problem(mesh, loads, Q(5, 7), Q(1, 10), 60)


def test_tree_lift_reproduces_every_free_hat_on_a_skew_mesh():
    p = signed_problem()
    s = ReferenceSystem(p)
    cells, boundary, _ = _geometry(p.mesh)
    residual = [Q(0)] * len(s.vertices)
    for i in s.free_nodes:
        residual[i] = Q(2*i-11, 7)
    lift = _tree_lift(cells, s.vertices, boundary, residual)
    observed = [Q(0)] * len(s.vertices)
    for z, cell in zip(lift, s.cells, strict=True):
        for i, (gx, gy) in zip(cell.nodes, cell.gradients):
            observed[i] += cell.area * (z[0] * gx + z[1] * gy)
    assert [observed[i] for i in s.free_nodes] == [residual[i] for i in s.free_nodes]


def test_exact_algebraic_errors_bounded_at_arbitrary_fields_and_optimizer():
    p = signed_problem()
    s = ReferenceSystem(p)
    anchor = p.digest()
    optimum, _ = optimal_fields(s)
    arbitrary = tuple(Q(2*i-15, 3) if i in s.free_nodes else Q(0) for i in range(len(s.vertices)))
    for v in (s.zero_primal(), arbitrary, s.primal_step(arbitrary), optimum):
        obs = inspect_stopping(p, anchor, v, "classical")
        actual = primal_energy(s, [x-y for x, y in zip(optimum, v)])
        assert actual <= obs.algebraic_upper_sq
        assert obs.stop == (obs.algebraic_upper_sq <= obs.discretization_lower_sq / 4)
        assert obs.discretization_lower_sq > 0
    assert obs.algebraic_upper_sq == 0 and obs.stop


def test_discretization_bound_against_independent_continuous_square_energy():
    n = 3
    mesh = uniform_mesh(n)
    p = Problem(mesh, (-3,) * len(mesh.triangles), 2, 1, 20)
    s = ReferenceSystem(p)
    vstar, _ = optimal_fields(s)
    low, _ = square_energy_interval(-3, 2, 21)
    disc_bound = low - primal_energy(s, vstar)
    v = s.zero_primal()
    for _ in range(5):
        obs = inspect_stopping(p, p.digest(), v, "classical")
        assert 0 < obs.discretization_lower_sq == Q(9, 80 * 2 * n**2) <= disc_bound
        v = s.primal_step(v)
    assert obs.stop and obs.residual_sq > 0
    assert not inspect_stopping(p, p.digest(), v, "fixed").stop


def test_signed_discretization_bound_survives_cancelled_free_load():
    mesh = uniform_mesh(2)
    p = Problem(mesh, (1, -1) * 4, 1, 1, 20)
    s = ReferenceSystem(p)
    assert len(s.free_nodes) == 1
    for rule in ("fixed", "classical"):
        zero = inspect_stopping(p, p.digest(), s.zero_primal(), rule)
        assert zero.rhs_sq == zero.residual_sq == 0 and zero.stop
    assert zero.discretization_lower_sq == Q(1, 320)
    wrong = tuple(Q(i in s.free_nodes) for i in range(len(s.vertices)))
    assert not inspect_stopping(p, p.digest(), wrong, "fixed").stop


def test_relative_residual_equality_and_exact_threshold():
    mesh = uniform_mesh(2)
    p = Problem(mesh, (1,) * 8, 1, 1, 20)
    vstar, _ = optimal_fields(ReferenceSystem(p))
    half = tuple(x/2 for x in vstar)
    equal = inspect_stopping(p, p.digest(), half, "fixed", relative_tolerance=Q(1, 2))
    below = inspect_stopping(p, p.digest(), half, "fixed", relative_tolerance=Q(1, 2)-Q(1, 10**90))
    assert equal.stop and equal.residual_sq == equal.rhs_sq / 4
    assert not below.stop
    assert equal.algebraic_upper_sq is equal.discretization_lower_sq is None


def test_frozen_state_boundary_and_projected_load_gates():
    p = signed_problem()
    s = ReferenceSystem(p)
    v = s.zero_primal()
    first = inspect_stopping(p, p.digest(), v, "classical")
    second = inspect_stopping(p, p.digest(), s.primal_step(v), "classical")
    assert first.primal_digest != second.primal_digest
    relaxed = replace(p, epsilon=2)
    with pytest.raises(ValueError, match="digest"):
        inspect_stopping(relaxed, p.digest(), v, "fixed")
    with pytest.raises(ValueError, match="boundary"):
        inspect_stopping(p, p.digest(), (1,) + v[1:], "fixed")
    projected = replace(p, loads=(1,) * len(p.loads), original_affine=(1, 0, 0))
    with pytest.raises(ValueError, match="P0"):
        inspect_stopping(projected, projected.digest(), v, "classical")
    with pytest.raises(ValueError, match="parameter"):
        inspect_stopping(p, p.digest(), v, "classical", fraction=0)
