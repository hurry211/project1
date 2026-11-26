from fractions import Fraction as Q

import pytest

from cfac.checker import check
from cfac.enrichment import EnrichmentRequest
from cfac.floor import assess
from cfac.primal_witness import screen_primal_floor
from cfac.prototype import ReferenceSystem, uniform_mesh
from cfac.rt1_checker import RT1Checker
from cfac.sparse_primal import PrimalSystem
from cfac.sparse_rt1 import dual_rt1_bounds, embed_rt0, optimize_rt1
from cfac.spec import Problem
from research.cfac.rt1_pair_reference import optimal_pair


@pytest.mark.parametrize("coefficient", [Q(1), Q(3, 5)])
def test_sparse_attribution_encloses_exact_primal_and_dual(coefficient):
    mesh = uniform_mesh(3)
    p = Problem(mesh, [i % 7 - 3 for i in range(len(mesh.triangles))], coefficient, Q(1, 100), 100)
    s = ReferenceSystem(p)
    v = s.primal_step(s.zero_primal())
    q = s.local_dual_sweep(s.feasible_flux)
    reference = assess(p, p.digest(), v, q, tight=True)
    sparse = assess(p, p.digest(), v, q, sparse_iterations=8)
    original = assess(p, p.digest(), v, q)
    for quantity in ("primal_alg", "dual_alg", "floor"):
        target, observed, baseline = [getattr(x, quantity) for x in (reference, sparse, original)]
        assert baseline.lower <= observed.lower <= target.lower <= observed.upper <= baseline.upper


def test_rounding_preserves_exact_rt1_equilibrium_and_bounds():
    mesh = uniform_mesh(3)
    p = Problem(mesh, [i % 3 - 1 for i in range(len(mesh.triangles))], Q(3, 2), Q(1, 1000), 100)
    anchor = p.digest()
    s = PrimalSystem(p, anchor)
    v = s.zero()
    for _, field, _ in s.proposals(v, 40):
        primal = s.frozen(field)
    q = s.flux
    parent = check(p, anchor, primal, q)
    request = EnrichmentRequest(p, anchor, parent.state_digest, primal, q, 1)
    seed = embed_rt0(request)
    inspect = RT1Checker().verify
    assert inspect(request, seed, seed.digest()).upper_sq == parent.upper_sq
    optimum_request, optimum_flux = optimal_pair(p)
    optimum_fixed_primal = type(seed)(anchor, parent.state_digest, 1,
                                     optimum_flux.face_moments, optimum_flux.cell_moments)
    optimum_gap = inspect(request, optimum_fixed_primal, optimum_fixed_primal.digest()).gap_sq
    candidate, cert, record = optimize_rt1(request, schedule="geometric", iterations=32)
    assert cert.exact_equilibrium and not cert.accepted
    assert cert.gap_sq < parent.gap_sq
    assert record["checks"] < record["iterations"] + 1
    dual = dual_rt1_bounds(request, candidate, cert)
    assert 0 <= dual.lower <= cert.gap_sq - optimum_gap <= dual.upper
    assert optimum_request.problem_digest == anchor


def test_sparse_quadratic_screen_is_bounded_by_independent_patch_certificate():
    mesh = uniform_mesh(3)
    p = Problem(mesh, [1] * len(mesh.triangles), 1, Q(1, 100), 100)
    s = PrimalSystem(p, p.digest())
    v = s.zero()
    original = check(p, p.digest(), v, s.flux)
    assert screen_primal_floor(p, p.digest(), v) <= original.upper_sq
    for _, field, _ in s.proposals(v, 50):
        v_last = s.frozen(field)
    optimum, moments = optimal_pair(p)
    upper = RT1Checker().verify(optimum, moments, moments.digest()).upper_sq
    assert p.epsilon**2 < screen_primal_floor(p, p.digest(), v_last) <= upper
