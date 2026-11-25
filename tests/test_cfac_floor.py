"""Independent discrete references and fail-closed checks for CFAC attribution."""

from dataclasses import replace
from fractions import Fraction as F
import random

import pytest

from cfac.evaluation import flux_energy, optimal_fields, primal_energy
from cfac.floor import Bounds, assess
from cfac.prototype import ReferenceSystem, uniform_mesh
from cfac.spec import Mesh, Problem


def make_problem(n=2, *, load=1, a=1, eps=F(1, 100)):
    mesh = uniform_mesh(n)
    return Problem(mesh, [load] * len(mesh.triangles), a, eps, 20)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_certified_intervals_against_discrete_optimizers(seed):
    # Skew the interior vertex, vary loads/signs and choose inexact P1 and
    # smooth + oscillatory stream modes independently of the checker.
    rng = random.Random(seed)
    mesh = uniform_mesh(2)
    vertices = list(mesh.vertices)
    offset = F(1, 2*seed+1)
    vertices[4] = (offset, F(1, 2) + offset/3)
    mesh = Mesh(vertices, mesh.triangles)
    p = Problem(mesh, [rng.randint(-3, 3) for _ in mesh.triangles], F(7, seed), F(1, 100), 10)
    system = ReferenceSystem(p)
    field = list(system.zero_primal())
    field[4] = F(rng.randint(-7, 7), 11)
    stream = [seed*x + 3*y + (-1)**i for i, (x, y) in enumerate(mesh.vertices)]
    flux = system.stream_correction(system.feasible_flux, stream)
    vstar, qstar = optimal_fields(system)
    ref_p = primal_energy(system, [v-r for v, r in zip(field, vstar)])
    ref_d = flux_energy(system, [q-r for q, r in zip(flux, qstar)])
    ref_floor = flux_energy(system, qstar) - primal_energy(system, vstar)
    for tight in (False, True):
        out = assess(p, p.digest(), field, flux, tight=tight)
        assert out.certificate.gap_sq == ref_floor + ref_p + ref_d
        for bounds, ref in [(out.primal_alg, ref_p), (out.dual_alg, ref_d), (out.floor, ref_floor)]:
            assert 0 <= bounds.lower <= ref <= bounds.upper
            if tight:
                assert bounds == Bounds(ref, ref)


def test_dual_nullspace_shift_and_pin_renumbering():
    p = make_problem(2)
    s = ReferenceSystem(p)
    v, q = optimal_fields(s)
    stream = tuple(x - y + (-1)**i for i, (x, y) in enumerate(s.vertices))
    q = s.stream_correction(q, stream)
    assert s.stream_correction(q, [17] * len(stream)) == q
    first = assess(p, p.digest(), v, q, tight=True)
    # Reverse global indices, choosing a physically different pinned vertex.
    n = len(v)
    reordered = Mesh(s.vertices[::-1], [tuple(n-1-i for i in t) for t in p.mesh.triangles])
    p2 = replace(p, mesh=reordered)
    s2 = ReferenceSystem(p2)
    by_edge = dict(zip(s.edges, q))
    q2 = [-by_edge[n-1-j, n-1-i] for i, j in s2.edges]
    second = assess(p2, p2.digest(), v[::-1], q2, tight=True)
    assert first.dual_alg == second.dual_alg
    assert first.dual_alg.lower > 0
    assert first.floor == second.floor


def test_futility_witness_uses_enriched_space_only_as_probe():
    p = make_problem(1, eps=F(1, 1000))
    s = ReferenceSystem(p)
    v = list(s.zero_primal())
    q = list(s.feasible_flux)
    before = (list(v), list(q))
    pair = assess(p, p.digest(), v, q, tight=True)
    primal = assess(p, p.digest(), v, q, tight=True, witness=True)
    assert pair.verdict == "PAIR_FUTILE"
    assert primal.verdict == "PRIMAL_SPACE_FUTILE"
    assert primal.primal_disc_lower_sq > p.epsilon**2
    assert primal.state_digest == pair.state_digest
    assert (v, q) == before


def test_uncertain_interval_and_tight_resolution():
    p = make_problem(3, eps=F(1, 2))
    s = ReferenceSystem(p)
    field = s.zero_primal()
    stream = [x + y + 3*(-1)**i for i, (x, y) in enumerate(s.vertices)]
    q = s.stream_correction(s.feasible_flux, stream)
    loose = assess(p, p.digest(), field, q)
    tight = assess(p, p.digest(), field, q, tight=True)
    assert loose.floor.lower <= p.epsilon**2 < loose.floor.upper
    assert loose.verdict == "UNDETERMINED"
    assert tight.verdict == "PAIR_SUFFICIENT"
    assert not tight.certificate.accepted


@pytest.mark.parametrize("eps, verdict", [
    (1, "ACCEPTED"), (1-F(1, 10**32), "PAIR_SUFFICIENT"), (1+F(1, 10**32), "ACCEPTED"),
])
def test_exact_current_threshold_and_pair_sufficiency(eps, verdict):
    p = make_problem(1, load=0, eps=eps)
    s = ReferenceSystem(p)
    q = s.stream_correction(s.feasible_flux, [y for x, y in s.vertices])
    out = assess(p, p.digest(), s.zero_primal(), q, tight=True)
    assert out.certificate.gap_sq == 1
    assert out.verdict == verdict
    assert out.floor == Bounds(0, 0)
    assert out.dual_alg == Bounds(1, 1)


def test_pair_equality_and_strict_futility():
    p = make_problem(1)
    s = ReferenceSystem(p)
    v, qstar = optimal_fields(s)
    p = replace(p, diffusion=flux_energy(s, qstar), epsilon=1)
    # At this diffusion the exact floor is one and the current gap exceeds one.
    q = s.stream_correction(qstar, [x-y for x, y in s.vertices])
    for eps, expected in [(1, "PAIR_SUFFICIENT"), (1-F(1, 10**32), "PAIR_FUTILE")]:
        current = replace(p, epsilon=eps)
        out = assess(current, current.digest(), v, q, tight=True)
        assert out.floor == Bounds(1, 1)
        assert out.verdict == expected
        assert not out.certificate.accepted


def test_stale_evidence_rejected_for_primal_flux_or_frozen_problem():
    p = make_problem()
    s = ReferenceSystem(p)
    v, q = s.zero_primal(), s.feasible_flux
    first = assess(p, p.digest(), v, q, tight=True)
    first.assert_current(p, p.digest(), list(v), list(q))
    changed = replace(p, epsilon=2*p.epsilon)
    for args in [(p, p.digest(), s.primal_step(v), q),
                 (p, p.digest(), v, s.local_dual_sweep(q)),
                 (changed, changed.digest(), v, q), (p, "wrong digest", v, q)]:
        with pytest.raises(ValueError, match="different state"):
            first.assert_current(*args)


def test_no_floor_for_defects_or_projected_target_even_with_tight_solve():
    p = make_problem(1, eps=F(1, 10000))
    s = ReferenceSystem(p)
    q = list(s.feasible_flux)
    q[0] += F(1, 2**54)
    defective = assess(p, p.digest(), s.zero_primal(), q, tight=True, witness=True)
    assert defective.floor is None
    assert defective.verdict == "UNDETERMINED"
    # Cellwise means on the rising-diagonal triangles for original f(x,y)=x.
    original = replace(p, loads=(F(2, 3), F(1, 3)), original_affine=(0, 1, 0))
    osys = ReferenceSystem(original)
    out = assess(original, original.digest(), osys.zero_primal(), osys.feasible_flux, tight=True)
    assert out.certificate.oscillation_sq > 0
    assert out.floor is None
    assert out.verdict == "UNDETERMINED"


def test_floor_cannot_coerce_scalar_text_into_a_certified_zero_field():
    p = make_problem(1, load=0)
    s = ReferenceSystem(p)
    result = assess(p, p.digest(), "0000", s.feasible_flux, tight=True)
    assert not result.certificate.accepted
    assert result.floor is None
    assert result.verdict == "UNDETERMINED"
