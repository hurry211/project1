"""Check global stationarity separately from the independent current-field check."""
from dataclasses import replace
from fractions import Fraction as Q
import json
from pathlib import Path

import pytest

from cfac.checker import check, edges
from cfac.enrichment import EnrichmentRequest
from cfac.prototype import uniform_mesh
from cfac.rt1_checker import RT1Checker
from cfac.spec import Mesh, Problem
from research.cfac.rt1_pair_reference import optimal_pair, replay, verify_optimum


def skew_problem(loads):
    mesh = uniform_mesh(2)
    points = list(mesh.vertices)
    points[4] = (Q(3, 8), Q(7, 16))
    return Problem(Mesh(points, mesh.triangles), loads, 2, Q(1, 2), 100)


def test_exact_stationarity_on_skew_triangles_and_constant_kernel():
    p = skew_problem([1, -2, 2, 0, 1, 3, -1, 1])
    req, moments = optimal_pair(p)
    report = verify_optimum(req, moments)
    assert report["free_primal_dofs"] == 1
    assert report["independent_stream_dofs"] == len(p.mesh.vertices) + len(edges(p.mesh)) - 1
    assert Q(report["gap_sq"]) > 0 and report["exact_equilibrium"]
    # A constant physical flux preserves divergence and normal continuity.
    # Its variation along a boundary stream mode must be detected.
    faces = []
    for (i, j), (m0, m1) in zip(edges(p.mesh), moments.face_moments):
        dy = p.mesh.vertices[j][1] - p.mesh.vertices[i][1]
        faces.append((m0 + dy, m1 + dy/2))
    cells = []
    for tri, (x, y) in zip(p.mesh.triangles, moments.cell_moments):
        a, b, c = (p.mesh.vertices[i] for i in tri)
        area = ((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))/2
        cells.append((x + area, y))
    shifted = replace(moments, face_moments=faces, cell_moments=cells)
    assert RT1Checker().verify(req, shifted, shifted.digest()).exact_equilibrium
    with pytest.raises(ValueError, match="stream stationarity"):
        verify_optimum(req, shifted)


def test_zero_reference_and_nonstationary_primal():
    p = skew_problem([0] * 8)
    req, moments = optimal_pair(p)
    assert verify_optimum(req, moments)["gap_sq"] == "0"
    primal = list(req.primal)
    primal[4] = Q(1, 16)
    parent = check(p, p.digest(), primal, req.rt0_flux)
    altered = EnrichmentRequest(p, p.digest(), parent.state_digest, primal, req.rt0_flux, 1)
    bound = replace(moments, parent_state_digest=parent.state_digest)
    assert RT1Checker().verify(altered, bound, bound.digest()).exact_equilibrium
    with pytest.raises(ValueError, match="Galerkin stationarity"):
        verify_optimum(altered, bound)


def test_archived_fields_close_both_original_targets():
    path = Path(__file__).resolve().parents[1] / "evidence/cfac/rt1-pair-reference.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    replay(receipt)
    for case, target in (("split_tight", Q(1, 14)), ("elbow_tight", Q(1, 12))):
        proof = receipt["cases"][case]["verification"]
        assert proof["accepted"] is True
        assert Q(proof["gap_sq"]) <= Q(proof["epsilon_sq"]) == target**2
