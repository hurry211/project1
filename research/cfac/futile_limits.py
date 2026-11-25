"""Offline P1 attainability diagnostics for the two unresolved frozen load cases."""
from fractions import Fraction as Q
import gzip
from hashlib import sha256
import json
from pathlib import Path

from cfac.checker import check, witness_lower_sq
from cfac.enrichment import EnrichmentRequest
from cfac.evaluation import optimal_fields
from cfac.prototype import ReferenceSystem, _solve_spd
from cfac.rt1_checker import RT1Checker
from cfac.rt1_reconstruction import RT1Reconstructor
from cfac.spec import Mesh, Problem
from cfac.stopping import inspect_stopping
from . import futile_study, stopping_study


ARCHIVE = Path(__file__).resolve().parents[2] / "evidence/cfac/futile-study.json.gz"
ARCHIVE_SHA = "7e4145dec3fc8dfaac14846e7ae0179d873caf7eb31202d1bdd9ccdf533222a7"


def quadratic_witness(system, primal):
    """Solve the residual equation on continuous P2 using P1 hats and edge bubbles."""
    vertices = system.vertices
    n = len(vertices)
    points = list(vertices) + [tuple((x+y)/2 for x, y in zip(vertices[i], vertices[j]))
                               for i, j in system.edges]
    free = {i: k for k, i in enumerate(i for i, (x, y) in enumerate(points)
                                       if 0 < x < 1 and 0 < y < 1)}
    rhs = [Q(0)] * len(free)
    matrix = [[Q(0)] * len(free) for _ in free]
    diffusion = system.problem.diffusion
    for cell, load in zip(system.cells, system.problem.loads, strict=True):
        indices = [*cell.nodes, *(n + e for e, _ in cell.faces)]
        local = [(i, free[g]) for i, g in enumerate(indices) if g in free]
        grad_v = [sum(primal[i]*g[d] for i, g in zip(cell.nodes, cell.gradients)) for d in (0, 1)]
        # Three edge midpoints integrate the degree-two stiffness and P0 load terms exactly.
        for opposite in range(3):
            lam = tuple(Q(0) if k == opposite else Q(1, 2) for k in range(3))
            gradients = list(cell.gradients)
            values = list(lam)
            for k in range(3):
                i, j = (k + 1) % 3, (k + 2) % 3
                values.append(4 * lam[i] * lam[j])
                gradients.append(tuple(4*(lam[i]*cell.gradients[j][d] + lam[j]*cell.gradients[i][d])
                                       for d in (0, 1)))
            for i, gi in local:
                x, y = gradients[i]
                rhs[gi] += cell.area/3 * (load*values[i] - diffusion*(x*grad_v[0] + y*grad_v[1]))
                for j, gj in local:
                    matrix[gi][gj] += cell.area/3 * diffusion*(x*gradients[j][0] + y*gradients[j][1])
    coeff = _solve_spd(matrix, rhs)
    values = [coeff[free[i]] if i in free else Q(0) for i in range(len(points))]
    mid = tuple(z + (values[i] + values[j])/2 for z, (i, j) in zip(values[n:], system.edges))
    return tuple(values[:n]), mid


def rt1_bound(problem, primal, flux):
    anchor = problem.digest()
    parent = check(problem, anchor, primal, flux)
    request = EnrichmentRequest(problem, anchor, parent.state_digest, primal, flux, 1)
    candidate = RT1Reconstructor().reconstruct(request)
    cert = RT1Checker().verify(request, candidate, candidate.digest())
    if cert.status != "CERTIFIED" or cert.state_digest != candidate.digest():
        raise ValueError("unbound RT1 trial")
    return {"parent_digest": parent.state_digest, "candidate_digest": cert.state_digest,
            "upper_sq": str(cert.upper_sq), "accepted": cert.accepted}


def missed_fields(archive=ARCHIVE):
    """Yield the two digest-bound refined problems and archived RT0 parent fields."""
    raw = archive.read_bytes()
    if sha256(raw).hexdigest() != ARCHIVE_SHA:
        raise ValueError("futile study archive digest differs")
    receipt = json.loads(gzip.decompress(raw))
    design = stopping_study.design(futile_study.PROTOCOL)
    for case in (c for c in design["cases"] if c["id"] in ("split_tight", "elbow_tight")):
        row = next(r for r in receipt["trials"] if r["case"] == case["id"] and r["policy"] == "cfac_retry")
        journal = receipt["journals"][row["journal_id"]]
        stopping_study.validate_journal(case, journal)
        last = journal["trace"][-1]
        if last["action"] != "rt1_inspect" or journal["accepted"]:
            raise ValueError("expected a checked RT1 miss")
        state = last["after"]["parent"]
        anchor = state["problem_digest"]
        data = journal["problems"][anchor]
        p = Problem(Mesh(data["vertices"], data["triangles"]), data["loads"], data["diffusion"],
                    data["epsilon"], data["max_actions"], data["original_affine"])
        yield case["id"], dict(row, archived_candidate_digest=last["after"]["state_digest"]), p, state


def inspect_limits():
    cases = {}
    for case_id, row, p, state in missed_fields():
        anchor = state["problem_digest"]
        system = ReferenceSystem(p)
        start, flux = tuple(map(Q, state["primal"])), tuple(map(Q, state["flux"]))
        primal = start
        extra = []
        for k in range(9):
            trial = rt1_bound(p, primal, flux)
            if k == 0 and trial["candidate_digest"] != row["archived_candidate_digest"]:
                raise ValueError("reconstruction differs from archived trial")
            extra.append({"extra_sweeps": k, **trial})
            if k < 8:
                primal = system.primal_step(primal)
        optimal, _ = optimal_fields(system)
        residual = inspect_stopping(p, anchor, optimal, "fixed").residual_sq
        if residual:
            raise ArithmeticError("reference P1 field has a nonzero algebraic residual")
        wv, we = quadratic_witness(system, optimal)
        lower = witness_lower_sq(p, anchor, optimal, wv, we)
        exact = rt1_bound(p, optimal, flux)
        if lower > Q(exact["upper_sq"]):
            raise ArithmeticError("inconsistent verified bounds")
        cases[case_id] = {"problem_digest": anchor, "epsilon": str(p.epsilon),
                             "source_journal": row["journal_id"], "extra_sweeps": extra,
                             "exact_discrete_p1": {"residual_sq": str(residual), **exact,
                                 "witness_vertices": list(map(str, wv)), "witness_edges": list(map(str, we)),
                                 "p1_error_lower_sq": str(lower),
                                 "p1_futility_proved": lower > p.epsilon**2}}
    return {"archive_sha256": ARCHIVE_SHA, "cases": cases}


if __name__ == "__main__":
    print(json.dumps(inspect_limits(), indent=2))
