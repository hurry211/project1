"""High-order proposals must retain the RT0 parent and expose all polynomial moments."""

from dataclasses import replace

import pytest

from cfac.checker import check, edges
from cfac.enrichment import EnrichmentRequest, RTMoments
from cfac.spec import Mesh, Problem


def parent(order):
    mesh = Mesh(((0, 0), (1, 0), (1, 1), (0, 1)), ((0, 1, 2), (0, 2, 3)))
    problem = Problem(mesh, (1, 1), 1, "1/100", 3)
    primal, flux = (0,) * 4, (0,) * 5
    trusted = problem.digest()
    state = check(problem, trusted, primal, flux)
    return EnrichmentRequest(problem, trusted, state.state_digest, primal, flux, order)


def moments(request):
    mesh, p = request.problem.mesh, request.order
    return RTMoments(request.problem_digest, request.parent_state_digest, p,
                     ((0,) * (p + 1),) * len(edges(mesh)),
                     ((0,) * (p * (p + 1)),) * len(mesh.triangles))


@pytest.mark.parametrize("order", (1, 2, 4))
def test_complete_layout_preserves_parent_and_never_accepts_a_field(order):
    request = parent(order)
    evidence = moments(request)
    assert evidence.validate_layout(request) is None
    assert not hasattr(evidence, "accepted")
    changed = replace(evidence, face_moments=((1,) * (order + 1),) + evidence.face_moments[1:])
    assert changed.digest() != evidence.digest()
    for bad in (replace(evidence, cell_moments=evidence.cell_moments[:-1]),
                replace(evidence, face_moments=evidence.face_moments[:-1]),
                replace(evidence, cell_moments=((),) + evidence.cell_moments[1:]),
                replace(evidence, face_moments=((),) + evidence.face_moments[1:]),
                replace(evidence, parent_state_digest="stale"),
                replace(evidence, order=order + 1)):
        with pytest.raises(ValueError):
            bad.validate_layout(request)


def test_enrichment_request_rejects_stale_parent_and_invalid_order():
    request = parent(1)
    for updates in ({"rt0_flux": (1,) * 5}, {"problem_digest": "unknown"}, {"order": True}):
        with pytest.raises(ValueError):
            replace(request, **updates)
