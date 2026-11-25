"""Independent weak equilibrium checks and patch stationarity on small rational meshes."""
from fractions import Fraction as Q

import pytest

from cfac.checker import check
from cfac.enrichment import EnrichmentRequest
from cfac.evaluation import optimal_fields
from cfac.prototype import ReferenceSystem, uniform_mesh
from cfac.rt1_reconstruction import RT1Reconstructor, _compatible_totals, _geometry, _patch
from cfac.spec import Mesh, Problem


def request(problem, primal, flux, order=1):
    digest = problem.digest()
    state = check(problem, digest, primal, flux)
    return EnrichmentRequest(problem, digest, state.state_digest, primal, flux, order)


def assert_divergence_moments(system, output):
    # Integration by parts against every barycentric hat is unisolvent for a
    # linear divergence. Endpoint data are never reconstructed by this test.
    for k, cell in enumerate(system.cells):
        for v, grad in zip(cell.nodes, cell.gradients):
            total = -sum(x*y for x, y in zip(output.cell_moments[k], grad))
            for edge, sign in cell.faces:
                first, second = system.edges[edge]
                zeroth, linear = output.face_moments[edge]
                if v == first:
                    total += sign * (zeroth - linear)
                if v == second:
                    total += sign * linear
            assert total == -cell.area * system.problem.loads[k] / 3


@pytest.mark.parametrize("iterate", ["zero", "sweep", "galerkin"])
def test_signed_cell_load_on_skew_patch_for_inexact_and_galerkin_fields(iterate):
    mesh = Mesh(((0, 0), (1, 0), (1, 1), (0, 1), (Q(2, 5), Q(3, 7))),
                ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))
    p = Problem(mesh, [3, -2, 5, -1], Q(9, 5), 1, 8)
    s = ReferenceSystem(p)
    u = {"zero": s.zero_primal(), "sweep": s.primal_step(s.zero_primal()),
         "galerkin": optimal_fields(s)[0]}[iterate]
    candidate = RT1Reconstructor().reconstruct(request(p, u, s.feasible_flux))
    assert_divergence_moments(s, candidate)


def test_galerkin_patch_totals_unchanged_and_inexact_residual_transferred():
    mesh = uniform_mesh(3)
    p = Problem(mesh, [1] * len(mesh.triangles), 2, 1, 8)
    system = ReferenceSystem(p)
    optimum, _ = optimal_fields(system)
    for u, needs_transfer in [(optimum, False), (system.primal_step(system.zero_primal()), True)]:
        req = request(p, u, system.feasible_flux)
        edges, cells, stars, owners = _geometry(mesh)
        totals, _ = _compatible_totals(req, edges, cells, owners)
        changed = False
        for k, cell in enumerate(system.cells):
            gx, gy = (sum(u[v]*g[d] for v, g in zip(cell.nodes, cell.gradients)) for d in (0, 1))
            original = [cell.area * (p.diffusion*(gx*dx + gy*dy) - p.loads[k]/3)
                        for dx, dy in cell.gradients]
            changed |= totals[k] != original
            assert sum(totals[k]) == -p.loads[k]*cell.area
        assert changed == needs_transfer
        for v in system.free_nodes:
            assert sum(totals[k][i] for k, i in stars[v]) == 0


def test_patch_minimization_stationary_in_every_admissible_curl_direction():
    mesh = uniform_mesh(2)
    p = Problem(mesh, [1]*len(mesh.triangles), Q(5, 3), 1, 8)
    s = ReferenceSystem(p)
    u, _ = optimal_fields(s)
    req = request(p, u, s.feasible_flux)
    edges, cells, stars, owners = _geometry(mesh)
    totals, gradients = _compatible_totals(req, edges, cells, owners)

    # Independent closed Newton-Cotes product integration after a Duffy map.
    # Boole weights exactly integrate degrees up to five along each axis.
    rule = list(zip((Q(i, 4) for i in range(5)), (Q(w, 90) for w in (7, 32, 12, 32, 7))))
    for center in (0, s.free_nodes[0]):
        derivatives = {("center", center): Q(0)}
        star = stars[center]
        for k, nodal, mids in _patch(center, star, cells, owners, totals, gradients, p):
            cell = s.cells[k]
            i = cell.nodes.index(center)
            for t, wt in rule:
                for h, wh in rule:
                    lam = ((1-t)*(1-h), t, (1-t)*h)
                    shape = [l*(2*l-1) for l in lam] + [4*lam[j-1]*lam[j-2] for j in range(3)]
                    flux = [sum(value[d]*w for value, w in zip(nodal+mids, shape)) for d in (0, 1)]
                    mis = [flux[d] - lam[i]*p.diffusion*gradients[k][d] for d in (0, 1)]
                    turns = {("center", center): cell.gradients[i]}
                    for j, v in enumerate(cell.nodes):
                        if j != i:
                            turns[("edge", v)] = [4*(lam[i]*cell.gradients[j][d]+lam[j]*cell.gradients[i][d])
                                                  for d in (0, 1)]
                    measure = 2*cell.area*(1-t)*wt*wh
                    for key, (dx, dy) in turns.items():
                        derivatives[key] = derivatives.get(key, 0) + measure*(mis[0]*dy-mis[1]*dx)
        assert all(val == 0 for val in derivatives.values())


def test_parent_provenance_is_distinct_from_feasible_patch_construction():
    mesh = uniform_mesh(2)
    p = Problem(mesh, (1,)*len(mesh.triangles), 1, 1, 8)
    s = ReferenceSystem(p)
    u = s.zero_primal()
    good = request(p, u, s.feasible_flux)
    wrong = request(p, u, [0]*len(s.edges))
    left, right = (RT1Reconstructor().reconstruct(r) for r in (good, wrong))
    assert left.digest() != right.digest()
    assert left.face_moments == right.face_moments and left.cell_moments == right.cell_moments
    assert_divergence_moments(s, right)
    with pytest.raises(ValueError):
        RT1Reconstructor().reconstruct(request(p, u, s.feasible_flux, order=2))
