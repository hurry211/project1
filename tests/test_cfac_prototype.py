"""Exact arithmetic and independent-checker contracts for small CPU CFAC systems."""

from fractions import Fraction as F
import random

import pytest

from cfac.checker import check, edges
from cfac.evaluation import four_terms, flux_energy, optimal_fields, primal_energy, square_energy_interval
from cfac.prototype import ReferenceSystem, uniform_mesh
from cfac.spec import Mesh, Problem


def system_for(n=3, diffusion=2, loads=None):
    mesh = uniform_mesh(n)
    return ReferenceSystem(Problem(mesh, (1,) * len(mesh.triangles) if loads is None else loads,
                                   diffusion, F(1, 100), 10))


def inspect(s, u, q):
    problem = s.problem
    return check(problem, problem.digest(), u, q)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_constructive_lift_handles_signed_cell_loads_for_arbitrary_iterates(n):
    rng = random.Random(211 + n)
    loads = tuple(F(rng.randint(-5, 7), 11) for _ in range(2 * n * n))
    s = system_for(n, F(7, 4), loads)
    assert s.edges == tuple(edges(s.problem.mesh))
    u = list(s.zero_primal())
    for i in s.free_nodes:
        u[i] = F(rng.randint(-100, 100), 3)
    before = inspect(s, u, s.feasible_flux)
    assert before.exact_equilibrium
    after = inspect(s, s.primal_step(u), s.feasible_flux)
    assert after.exact_equilibrium and after.gap_sq <= before.gap_sq


def test_one_sweep_remains_inexact_and_direct_optima_have_zero_residuals():
    s = system_for()
    u_star, q_star = optimal_fields(s)
    u = s.primal_step(s.zero_primal())
    assert u != u_star
    assert all(sum(s.primal_matrix[i][j] * value for j, value in enumerate(u_star)) == s.load[i]
               for i in s.free_nodes)
    assert all(value == 0 for value in s.stream_rhs(q_star))
    assert sum(s.stream_rhs(s.feasible_flux)) == 0
    assert inspect(s, u_star, q_star).exact_equilibrium


@pytest.mark.parametrize("mode", ["smooth", "oscillatory"])
def test_dual_modes_and_coarse_space(mode):
    s = system_for()
    u, q_star = optimal_fields(s)
    if mode == "smooth":
        phi = [2*x - 3*y + min(x, y) for x, y in s.vertices]
    else:
        phi = [F((-1)**(int(x * 3) + int(y * 3)), 3) for x, y in s.vertices]
    q = s.stream_correction(q_star, phi)
    q_local = s.local_dual_sweep(q)
    q_coarse = s.coarse_dual_step(q)
    before = inspect(s, u, q)
    local, coarse = (inspect(s, u, z) for z in (q_local, q_coarse))
    assert before.exact_equilibrium and local.exact_equilibrium and coarse.exact_equilibrium
    assert local.gap_sq < before.gap_sq
    assert coarse.gap_sq <= before.gap_sq
    assert (q_coarse == q_star) == (mode == "smooth")
    assert s.coarse_dual_step(q_coarse) == q_coarse


@pytest.mark.parametrize("seed", [19, 47, 53])
def test_four_independent_terms_sum_to_checked_gap_for_nonoptimal_states(seed):
    s = system_for(diffusion=F(5, 3))
    rng = random.Random(seed)
    u = list(s.zero_primal())
    for i in s.free_nodes:
        u[i] = F(rng.randint(-20, 20), 31)
    q = s.stream_correction(s.feasible_flux, [F(rng.randint(-9, 9), 17) for _ in s.vertices])
    terms = four_terms(s, u, q)
    cert = inspect(s, u, q)
    assert cert.exact_equilibrium
    assert terms.primal_algebraic_sq > 0 and terms.dual_algebraic_sq > 0
    assert cert.gap_sq == terms.pair_floor_sq + terms.primal_algebraic_sq + terms.dual_algebraic_sq
    e_lo, e_hi = terms.primal_discretization_sq
    d_lo, d_hi = terms.dual_discretization_sq
    assert 0 < e_lo < e_hi and 0 < d_lo < d_hi
    assert e_lo + d_hi == e_hi + d_lo == terms.pair_floor_sq


def test_sine_energy_reference_agrees_with_energy_scaling_and_convergence():
    loose = square_energy_interval(1, 1, 5)
    tight = square_energy_interval(1, 1, 19)
    assert loose[0] < tight[0] < tight[1] < loose[1]
    # Independent coarse bracket for the square torsion integral.
    assert F(351, 10000) < tight[0] < tight[1] < F(352, 10000)
    assert square_energy_interval(-2, 2, 19) == tuple(2 * x for x in tight)


def test_reject_unbalanced_flux_and_invalid_primal_from_decomposition():
    s = system_for()
    q = list(s.feasible_flux)
    q[0] += F(1, 2**80)
    with pytest.raises(ValueError, match="equilibrium"):
        four_terms(s, s.zero_primal(), q)
    u = list(s.zero_primal())
    u[0] = 1
    with pytest.raises(ValueError, match="homogeneous"):
        four_terms(s, u, s.feasible_flux)


def test_rt0_zero_flux_weighted_energy_and_constant_stream_gauge():
    s = system_for(2)
    assert flux_energy(s, (0,) * len(s.edges)) == 0
    assert s.stream_correction(s.feasible_flux, [F(29, 4)] * len(s.vertices)) == s.feasible_flux


def test_oblique_nonuniform_mesh_and_discontinuous_signed_load():
    mesh = uniform_mesh(2)
    vertices = list(mesh.vertices)
    vertices[4] = (F(3, 8), F(4, 7))
    problem = Problem(Mesh(vertices, mesh.triangles), range(-4, 4), F(3, 7), F(1, 30), 5)
    s = ReferenceSystem(problem)
    u_star, q_star = optimal_fields(s)
    u = list(s.zero_primal())
    u[4] = F(-1, 13)
    q = s.coarse_dual_step(s.feasible_flux)
    primal_diff = tuple(x - y for x, y in zip(u, u_star))
    dual_diff = tuple(x - y for x, y in zip(q, q_star))
    c, optimal = inspect(s, u, q), inspect(s, u_star, q_star)
    assert c.exact_equilibrium and optimal.exact_equilibrium
    assert c.gap_sq == optimal.gap_sq + primal_energy(s, primal_diff) + flux_energy(s, dual_diff)
