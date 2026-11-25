"""Offline exact minimum of the P1 / equilibrated RT1 gap on a triangulated square.

Every divergence-free RT1 perturbation is affine on each triangle. Its
quadratic radial term has affine homogeneous divergence 3(h*r+i*s), so h=i=0.
It is the rotated gradient of a quadratic stream locally. Normal continuity
matches the tangential derivatives across edges, and simple connectivity lets
the local constants be chosen to give one continuous P2 stream. This space
has a single constant kernel and no boundary restrictions for Dirichlet data.
"""
import argparse
from fractions import Fraction as Q
import json
from pathlib import Path

from cfac.checker import _geometry, _grad, check
from cfac.enrichment import EnrichmentRequest, RTMoments
from cfac.evaluation import optimal_fields
from cfac.prototype import ReferenceSystem, _solve_spd
from cfac.rt1_checker import RT1Checker, _field_on_cell


MIDPOINTS = tuple(tuple(Q(i != j, 2) for i in range(3)) for j in range(3))


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1]


def _modes(cell, lam):
    curls = [(gy, -gx) for gx, gy in cell.gradients]
    for i, j in ((1, 2), (2, 0), (0, 1)):
        curls.append(tuple(4 * (lam[i]*curls[j][d] + lam[j]*curls[i][d]) for d in (0, 1)))
    return curls


def _lift(system, cell, lam):
    points = [system.vertices[v] for v in cell.nodes]
    x = tuple(sum(w*p[d] for w, p in zip(lam, points)) for d in (0, 1))
    return tuple(sum(s*system.feasible_flux[e]*(x[d]-p[d]) for (e, s), p in zip(cell.faces, points))
                 / (2*cell.area) for d in (0, 1))


def optimal_pair(p):
    if p.original_affine is not None:
        raise ValueError("global equilibrated RT1 reference requires exact P0 data")
    system = ReferenceSystem(p)
    primal, _ = optimal_fields(system)
    n = len(system.vertices)
    size = n + len(system.edges)
    gram = [[Q(0)] * size for _ in range(size)]
    rhs = [Q(0)] * size
    for cell in system.cells:
        ids = (*cell.nodes, *(n + e for e, _ in cell.faces))
        for lam in MIDPOINTS:
            modes = _modes(cell, lam)
            q = _lift(system, cell, lam)
            # Constant diffusion scales both sides by the same positive factor.
            weight = cell.area / 3
            for i, u in zip(ids, modes):
                rhs[i] -= weight * _dot(q, u)
                for j, v in zip(ids, modes):
                    gram[i][j] += weight * _dot(u, v)
    stream = (Q(0),) + _solve_spd([r[1:] for r in gram[1:]], rhs[1:])
    faces = []
    for e, (i, j) in enumerate(system.edges):
        mean = system.feasible_flux[e] + stream[j] - stream[i]
        # On the global oriented edge, curl(psi).n ds = d psi / dt dt.
        faces.append((mean, mean/2 - 2*stream[n + e]/3))
    cells = []
    centroid = (Q(1, 3),) * 3
    for cell in system.cells:
        ids = (*cell.nodes, *(n + e for e, _ in cell.faces))
        modes, base = _modes(cell, centroid), _lift(system, cell, centroid)
        cells.append(tuple(cell.area*(base[d] + sum(stream[i]*u[d] for i, u in zip(ids, modes)))
                           for d in (0, 1)))
    anchor = p.digest()
    parent = check(p, anchor, primal, system.feasible_flux)
    request = EnrichmentRequest(p, anchor, parent.state_digest, primal, system.feasible_flux, 1)
    moments = RTMoments(anchor, parent.state_digest, 1, faces, cells)
    return request, moments


def verify_optimum(request, moments):
    """Check both sets of first-order conditions directly on the materialized fields."""
    p = request.problem
    if p.original_affine is not None:
        raise ValueError("expected original P0 data")
    cert = RT1Checker().verify(request, moments, moments.digest())
    if cert.status != "CERTIFIED" or not cert.exact_equilibrium or cert.upper_sq != cert.gap_sq:
        raise ValueError("expected an exactly equilibrated certified RT1 field")
    geometry, boundary, _ = _geometry(p.mesh)
    n = len(p.mesh.vertices)
    residual = [Q(0)] * n
    dual = [Q(0)] * (n + len(moments.face_moments))
    for k, cell in enumerate(geometry):
        area = cell.twice_area/2
        grad_v = _grad(cell, request.primal)
        for i, g in zip(cell.tri, cell.grads):
            residual[i] += area*(p.diffusion*_dot(grad_v, g) - p.loads[k]/3)
        components = _field_on_cell(cell, moments.face_moments, moments.cell_moments[k])[:2]
        if any(any(component[3:]) for component in components):
            raise ValueError("a P0-equilibrated RT1 field must be cellwise affine")
        # Reconstruct physical values at all three vertices from checked moments.
        values = [tuple(a[0] for a in components),
                  tuple(a[0]+a[1] for a in components), tuple(a[0]+a[2] for a in components)]
        sums = tuple(map(sum, zip(*values)))
        curls = [(gy, -gx) for gx, gy in cell.grads]
        # Closed barycentric moments independently check the midpoint assembly.
        mass = [tuple(area*(s+x)/12 for s, x in zip(sums, q)) for q in values]
        for i, c in zip(cell.tri, curls):
            dual[i] += area*_dot(sums, c)/3
        for e, (i, j) in zip(cell.edge_ids, ((1, 2), (2, 0), (0, 1))):
            dual[n + e] += 4*(_dot(mass[i], curls[j]) + _dot(mass[j], curls[i]))
    if any(r for i, r in enumerate(residual) if i not in boundary):
        raise ValueError("P1 field fails Galerkin stationarity")
    if any(dual):
        raise ValueError("RT1 field fails stream stationarity")
    return {"gap_sq": str(cert.gap_sq), "epsilon_sq": str(p.epsilon**2),
            "accepted": cert.accepted, "exact_equilibrium": cert.exact_equilibrium,
            "free_primal_dofs": n - len(boundary), "independent_stream_dofs": len(dual) - 1,
            "free_primal_residual_sq": "0", "full_stream_residual_sq": "0"}


def _case_fields(p, request, moments):
    return {"problem_digest": p.digest(), "parent_digest": request.parent_state_digest,
            "candidate_digest": moments.digest(), "primal": list(map(str, request.primal)),
            "rt0_parent_flux": list(map(str, request.rt0_flux)),
            "rt1_faces": [list(map(str, row)) for row in moments.face_moments],
            "rt1_cells": [list(map(str, row)) for row in moments.cell_moments]}


def study():
    from .futile_limits import ARCHIVE_SHA, missed_fields
    receipt = {"schema": "cfac-equilibrated-rt1-optimum-v1", "source_archive_sha256": ARCHIVE_SHA,
               "cases": {}}
    for case_id, row, problem, _ in missed_fields():
        request, moments = optimal_pair(problem)
        receipt["cases"][case_id] = {"source_journal": row["journal_id"],
                                     "fields": _case_fields(problem, request, moments),
                                     "verification": verify_optimum(request, moments)}
    return receipt


def replay(receipt):
    from .futile_limits import ARCHIVE_SHA, missed_fields
    if (receipt["schema"] != "cfac-equilibrated-rt1-optimum-v1"
            or receipt["source_archive_sha256"] != ARCHIVE_SHA):
        raise ValueError("unexpected reference schema or source receipt")
    expected = set()
    for case_id, row, problem, _ in missed_fields():
        expected.add(case_id)
        record = receipt["cases"][case_id]
        fields = record["fields"]
        if record["source_journal"] != row["journal_id"] or fields["problem_digest"] != problem.digest():
            raise ValueError("reference problem differs from the retained numerical case")
        request = EnrichmentRequest(problem, fields["problem_digest"], fields["parent_digest"],
                                    fields["primal"], fields["rt0_parent_flux"], 1)
        moments = RTMoments(problem.digest(), request.parent_state_digest, 1,
                            fields["rt1_faces"], fields["rt1_cells"])
        if (moments.digest() != fields["candidate_digest"]
                or verify_optimum(request, moments) != record["verification"]):
            raise ValueError("stored moments or optimum verdict differ")
    if set(receipt["cases"]) != expected:
        raise ValueError("unexpected reference case set")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.output:
        receipt = study()
        replay(receipt)
        with args.output.open("x", encoding="utf-8", newline="\n") as out:
            json.dump(receipt, out, indent=2, sort_keys=True)
            out.write("\n")
    else:
        receipt = json.loads(args.verify.read_text(encoding="utf-8"))
        replay(receipt)
        print("Verified equilibrated P1/RT1 optima for", ", ".join(sorted(receipt["cases"])))


if __name__ == "__main__":
    main()
