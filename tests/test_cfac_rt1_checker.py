"""Exact reference integrals and malformed/stale RT1 evidence checks."""

from dataclasses import replace
from fractions import Fraction as Q
import random

import pytest

from cfac.checker import check, edges
from cfac.enrichment import EnrichmentRequest, RTMoments
from cfac.rt1_checker import RT1Checker
from cfac.spec import Mesh, Problem


SQUARE = Mesh(((0, 0), (1, 0), (1, 1), (0, 1)), ((0, 1, 2), (0, 2, 3)))
IRREGULAR = Mesh(((Q(2, 5), Q(3, 7)), (1, 1), (0, 0), (1, 0), (0, 1)),
                 ((0, 2, 3), (1, 0, 3), (1, 4, 0), (0, 4, 2)))


def make_request(mesh=SQUARE, *, load=0, original=None, diffusion=1, epsilon=1, primal=None):
    if original is None:
        loads = (load,) * len(mesh.triangles)
    else:
        c, a, b = original
        loads = tuple(sum(c + a * mesh.vertices[i][0] + b * mesh.vertices[i][1] for i in tri) / 3
                      for tri in mesh.triangles)
    p = Problem(mesh, loads, diffusion, epsilon, 4, original)
    if primal is None:
        primal = (0,) * len(mesh.vertices)
    rt0 = (0,) * len(edges(mesh))
    cert = check(p, p.digest(), primal, rt0)
    return EnrichmentRequest(p, p.digest(), cert.state_digest, primal, rt0, 1)


def moments(request, field):
    """Obtain actual moments from an analytic vector function by rational rules."""
    mesh = request.problem.mesh
    faces, cells = [], []
    for i, j in edges(mesh):
        (x, y), (xx, yy) = mesh.vertices[i], mesh.vertices[j]
        dx, dy = xx - x, yy - y
        normal = []
        for t in (0, Q(1, 2), 1):
            u, v = field(x + t * dx, y + t * dy)
            normal.append(dy * u - dx * v)
        left, mid, right = normal
        faces.append(((left + 4 * mid + right) / 6, (2 * mid + right) / 6))
    for tri in mesh.triangles:
        a, b, c = (mesh.vertices[i] for i in tri)
        double_area = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        values = [field((p[0]+q[0])/2, (p[1]+q[1])/2) for p, q in ((a,b), (b,c), (c,a))]
        cells.append(tuple(double_area * sum(v) / 6 for v in zip(*values)))
    return RTMoments(request.problem_digest, request.parent_state_digest, 1, faces, cells)


def verify(request, candidate):
    cert = RT1Checker().verify(request, candidate, candidate.digest())
    assert cert.status == "CERTIFIED"
    assert cert.state_digest == candidate.digest()
    return cert


@pytest.mark.parametrize("mesh", [SQUARE, IRREGULAR])
def test_quartic_gap_and_exact_original_linear_divergence(mesh):
    request = make_request(mesh, original=(0, -3, 0), diffusion=2)
    candidate = moments(request, lambda x, y: (x*x, x*y))
    cert = verify(request, candidate)
    # div(q)=3x. Original equilibrium is exact. Projected equilibrium is not.
    assert cert.gap_sq == cert.upper_sq == Q(7, 45)
    assert cert.defect_sq == cert.oscillation_sq > 0
    assert cert.original_residual_sq == 0
    assert not cert.exact_equilibrium


def test_original_defect_includes_projection_divergence_cross_term():
    request = make_request(original=(0, 3, 0), epsilon=Q(27, 20))
    candidate = moments(request, lambda x, y: (x*x, x*y))
    cert = verify(request, candidate)
    # Original residual is 6x, integral of its square is 12. Adding the component
    # norms alone misses a positive cross term and would cross this threshold.
    gap = Q(14, 45)
    correction = Q(12, 18)
    assert cert.gap_sq == gap
    assert cert.defect_sq + cert.oscillation_sq == 11
    assert cert.original_residual_sq == 12
    wrong_correction = Q(11, 18)
    slack = request.problem.epsilon**2 - gap - wrong_correction
    assert slack > 0 and slack**2 >= 4 * gap * wrong_correction
    remainder = cert.upper_sq - gap - correction
    assert remainder >= 0 and remainder**2 >= 4 * gap * correction
    assert not cert.accepted


def test_interior_normal_moment_and_zero_mean_divergence_defects():
    r = make_request(epsilon=Q(1, 100))
    base = moments(r, lambda x, y: (0, 0))
    assert verify(r, base).upper_sq == 0
    for field in ("face_moments", "cell_moments"):
        rows = list(getattr(base, field))
        idx = edges(SQUARE).index((0, 2)) if field == "face_moments" else 0
        rows[idx] = (0, 1)
        perturbed = verify(r, replace(base, **{field: rows}))
        # All zeroth edge moments remain zero, so a total-flux test alone would pass.
        assert not perturbed.exact_equilibrium and not perturbed.accepted
        assert perturbed.defect_sq > 0 and perturbed.upper_sq > perturbed.gap_sq


def test_embedding_rt0_preserves_independent_certificate_on_inexact_primal():
    r = make_request(IRREGULAR, load=1, diffusion=Q(5, 3), primal=(Q(1, 8), 0, 0, 0, 0))
    embedded = moments(r, lambda x, y: (2 - x/2, 1 - y/2))
    baseline = check(r.problem, r.problem_digest, r.primal, [row[0] for row in embedded.face_moments])
    enriched = verify(r, embedded)
    assert baseline.exact_equilibrium and enriched.exact_equilibrium
    assert baseline.gap_sq == baseline.upper_sq == enriched.gap_sq == enriched.upper_sq
    assert enriched.state_digest != baseline.state_digest


def test_random_rt1_fields_against_global_square_monomials():
    rng = random.Random(49813)
    r = make_request(IRREGULAR, load=2, diffusion=Q(7, 3))
    for _ in range(12):
        a, b, c, d, e, g, h, i = [Q(rng.randint(-8, 8), rng.randint(1, 8)) for _ in range(8)]
        def field(x, y):
            return a+b*x+c*y+x*(h*x+i*y), d+e*x+g*y+y*(h*x+i*y)

        # Cartesian integrals on the unit square, independent of triangle rules.
        def square(poly):
            return sum(p*q / ((1+u+k)*(1+v+l)) for (u,v,p) in poly for (k,l,q) in poly)

        qx = [(0,0,a), (1,0,b), (0,1,c), (2,0,h), (1,1,i)]
        qy = [(0,0,d), (1,0,e), (0,1,g), (1,1,h), (0,2,i)]
        div_f = [(0,0,b+g+2), (1,0,3*h), (0,1,3*i)]
        cert = verify(r, moments(r, field))
        assert cert.gap_sq == (square(qx) + square(qy)) / r.problem.diffusion
        assert cert.defect_sq == square(div_f)


@pytest.mark.parametrize("eps, accepted", [(5, True), (5-Q(1,10**100), False)])
def test_exact_acceptance_threshold(eps, accepted):
    r = make_request(epsilon=eps)
    cert = verify(r, moments(r, lambda x, y: (3, 4)))
    assert cert.upper_sq == 25 and cert.accepted is accepted


def test_fail_closed_on_malformed_stale_or_unsupported_receipts():
    r = make_request()
    candidate = moments(r, lambda x, y: (x*x, x*y))
    other = make_request(epsilon=2)
    checker = RT1Checker()
    for req, item, trusted in (
        (r, candidate, "not-the-retained-digest"),
        (other, candidate, candidate.digest()),
        (replace(r, order=2), replace(candidate, order=2), replace(candidate, order=2).digest()),
        (r, replace(candidate, cell_moments=()), replace(candidate, cell_moments=()).digest()),
        (r, replace(candidate, face_moments=((1, 2),)), replace(candidate, face_moments=((1, 2),)).digest()),
        (r, None, candidate.digest()),
    ):
        verdict = checker.verify(req, item, trusted)
        assert verdict.status == "ESTIMATED" and not verdict.accepted and verdict.state_digest is None
        assert verdict.original_residual_sq is None
    with pytest.raises(ValueError):
        replace(candidate, cell_moments=((0, float("inf")), (0, 0)))
