"""Analytic integrals, exact thresholds and malformed CFAC certificate evidence."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
import random

import pytest

from cfac.checker import check, edges, witness_lower_sq
from cfac.spec import Mesh, Problem, rational


SQUARE = Mesh(((0, 0), (1, 0), (1, 1), (0, 1)), ((0, 1, 2), (0, 2, 3)))
ZERO = (0,) * 4


def problem(**changes):
    return replace(Problem(SQUARE, (1, 1), 1, 1, 10), **changes)


def integrated_flux(mesh, field):
    # A midpoint rule independently integrates the oriented trace of an affine vector.
    values = []
    for i, j in edges(mesh):
        a, b = mesh.vertices[i], mesh.vertices[j]
        u, v = field((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        values.append(u * (b[1] - a[1]) - v * (b[0] - a[0]))
    return values


def test_exact_rt0_identity_and_p2_witness():
    p = problem()
    flux = integrated_flux(p.mesh, lambda x, y: (-x / 2, -y / 2))
    cert = check(p, p.digest(), ZERO, flux)
    assert (cert.status, cert.accepted, cert.exact_equilibrium) == ("CERTIFIED", True, True)
    assert cert.upper_sq == cert.gap_sq == Q(1, 6)
    midpoint = [int(e == (0, 2)) for e in edges(p.mesh)]
    for scale in (-3, 1, 10):
        assert witness_lower_sq(p, p.digest(), ZERO, ZERO, [v * scale for v in midpoint]) == Q(1, 48)
    assert witness_lower_sq(p, p.digest(), ZERO, ZERO, [0] * 5) == 0
    assert check(p, p.digest(), ZERO, flux).state_digest == cert.state_digest


def test_global_normal_orientation_and_defect_correction():
    p = problem(epsilon=Q(1, 5))
    flux = integrated_flux(p.mesh, lambda x, y: (-x / 2 + 3, -y / 2))
    assert check(p, p.digest(), ZERO, flux).exact_equilibrium
    flux[edges(p.mesh).index((0, 2))] *= -1
    flipped = check(p, p.digest(), ZERO, flux)
    assert flipped.status == "CERTIFIED" and not flipped.exact_equilibrium and not flipped.accepted
    assert flipped.upper_sq > flipped.gap_sq and flipped.defect_sq > 0
    weak = check(p, p.digest(), ZERO, [0] * 5)
    assert not weak.accepted and not weak.exact_equilibrium
    assert weak.gap_sq == 0 and weak.upper_sq == Q(1, 18)
    small = replace(p, loads=(Q(1, 10),) * 2)
    assert check(small, small.digest(), ZERO, [0] * 5).accepted


def test_affine_original_and_independent_polynomial_integrals():
    p = problem(loads=(Q(2, 3), Q(1, 3)), original_affine=(0, 1, 0))
    cert = check(p, p.digest(), ZERO, [0] * 5)
    assert cert.defect_sq == Q(5, 18)
    assert cert.oscillation_sq == Q(1, 18)
    assert cert.upper_sq == Q(1, 54)
    w = [int(e == (0, 2)) for e in edges(p.mesh)]
    assert witness_lower_sq(p, p.digest(), ZERO, ZERO, w) == Q(1, 192)
    wrong = replace(p, loads=tuple(reversed(p.loads)))
    assert check(wrong, wrong.digest(), ZERO, [0] * 5).status == "ESTIMATED"


def test_digest_commits_to_all_mathematical_and_budget_data():
    p = problem()
    retained = p.digest()
    for other in (replace(p, loads=(2, 2)), replace(p, diffusion=2), replace(p, epsilon=2),
                  replace(p, max_actions=9), replace(p, original_affine=(1, 0, 0)),
                  replace(p, mesh=Mesh(p.mesh.vertices, p.mesh.triangles[::-1]))):
        cert = check(other, retained, ZERO, [0] * 5)
        assert not cert.accepted and cert.state_digest is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, "nan", "1/0"])
def test_nonfinite_and_invalid_coefficients_fail_closed(bad):
    p = problem()
    assert check(p, p.digest(), [bad] + [0] * 3, [0] * 5).status == "ESTIMATED"
    assert check(p, p.digest(), ZERO, [bad] + [0] * 4).status == "ESTIMATED"
    with pytest.raises(ValueError):
        replace(p, epsilon=bad)


@pytest.mark.parametrize("triangles", [
    ((0, 1, 2),), ((0, 1, 2), (0, 1, 2)), ((0, 2, 1), (0, 2, 3)),
    ((0, 1, 2), (1, 2, 3)), ((0, 1, 4), (0, 2, 3)), ((0, 0, 2), (0, 2, 3)),
])
def test_incomplete_overlapping_reversed_or_invalid_mesh(triangles):
    p = problem(mesh=Mesh(SQUARE.vertices, triangles), loads=(1,) * len(triangles))
    cert = check(p, p.digest(), ZERO, [0] * len(edges(p.mesh)))
    assert cert.status == "ESTIMATED" and not cert.accepted


def test_hanging_node_and_incomplete_interior_boundary():
    mesh = Mesh((*SQUARE.vertices, (Q(1, 2), Q(1, 2))), ((0, 1, 2), (0, 4, 3), (4, 2, 3)))
    p = problem(mesh=mesh, loads=(1, 1, 1))
    assert check(p, p.digest(), [0] * 5, [0] * len(edges(mesh))).status == "ESTIMATED"


def test_primal_and_witness_boundary_trace():
    p = problem()
    assert check(p, p.digest(), [0, 1, 0, 0], [0] * 5).status == "ESTIMATED"
    with pytest.raises(ValueError):
        witness_lower_sq(p, p.digest(), ZERO, ZERO, [1] * 5)
    with pytest.raises(ValueError):
        witness_lower_sq(p, p.digest(), ZERO, [1] * 4, [0] * 5)


def test_exact_acceptance_even_with_sub_float_threshold_difference():
    p = problem(loads=(0, 0), epsilon=5)
    flux = integrated_flux(p.mesh, lambda x, y: (3, 4))
    assert check(p, p.digest(), ZERO, flux).upper_sq == 25
    assert check(p, p.digest(), ZERO, flux).accepted
    below = replace(p, epsilon=p.epsilon - Q(1, 10**100))
    assert float(below.epsilon) == 5
    assert not check(below, below.digest(), ZERO, flux).accepted


def test_frozen_copy_and_binary_input_semantics():
    verts = [[0, 0], [1, 0], [1, 1], [0, 1]]
    loads = [1, 1]
    p = Problem(Mesh(verts, SQUARE.triangles), loads, "1/4", 1, 0)
    d = p.digest()
    loads[0] = 9
    verts[2][0] = 9
    assert d == p.digest() and p.loads == (1, 1) and p.mesh.vertices[2] == (1, 1)
    assert rational(0.1) != rational("0.1")
    with pytest.raises(FrozenInstanceError):
        p.epsilon = 30
    with pytest.raises(ValueError):
        replace(p, max_actions=-1)
    with pytest.raises(ValueError):
        replace(p, loads="11")
    assert check(p, d, "0000", [0] * 5).status == "ESTIMATED"


def test_properties_of_rational_affine_fluxes_and_outward_rounding():
    rng = random.Random(249)
    for _ in range(40):
        a, b, c = (Q(rng.randint(-13, 13), rng.randint(1, 15)) for _ in range(3))
        k = Q(rng.randint(1, 9), rng.randint(1, 8))
        f = Q(rng.randint(-20, 20), 3)
        p = problem(loads=(f, f), diffusion=k)
        flux = integrated_flux(p.mesh, lambda x, y: (a + c * x, b + c * y))
        cert = check(p, p.digest(), ZERO, flux)
        exact_gap = (a*a + b*b + c*(a + b) + 2*c*c/3) / k
        exact_correction = (f + 2*c)**2 / (18*k)
        assert cert.gap_sq == exact_gap
        remainder = cert.upper_sq - exact_gap - exact_correction
        assert remainder >= 0 and remainder**2 >= 4 * exact_gap * exact_correction
        assert cert.exact_equilibrium == (f == -2*c)
