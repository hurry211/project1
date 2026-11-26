"""Experimental two-level CFAC controller. Exact checkers own field acceptance."""

from time import perf_counter

import numpy as np

from .checker import check
from .controller import _refine
from .enrichment import EnrichmentRequest, RTMoments
from .floor import assess
from .primal_witness import screen_primal_floor
from .rt1_checker import RT1Checker
from .rt1_reconstruction import RT1Reconstructor
from .sparse_primal import PrimalSystem
from .sparse_rt1 import dual_rt1_bounds, optimize_rt1
from .stopping import inspect_stopping


def _request(problem, field, flux):
    anchor = problem.digest()
    parent = check(problem, anchor, field, flux)
    return EnrichmentRequest(problem, anchor, parent.state_digest, field, flux, 1)


def run_sparse(problem, trusted_digest, *, policy, iterations=128, refinements=1, max_vertices=None):
    """Use the same CSR backend, refinement and RT1 producer across policy ablations."""
    if policy not in ("classical_sparse", "fixed_sparse", "cfac_sparse", "cfac_geometric", "cfac_no_coarse",
                      "classical_patch", "cfac_screened"):
        raise ValueError("unknown sparse CFAC policy")
    if problem.original_affine is not None:
        raise ValueError("the sparse controller requires an original P0 load")
    metrics = {"primal_iterations": 0, "rt1_iterations": 0, "certificate_checks": 0,
               "stopping_checks": 0, "floor_checks": 0, "flux_reconstructions": 0, "levels": []}
    adaptive = policy.startswith("cfac_")
    initial = (0,) * len(problem.mesh.vertices)
    for level in range(refinements + 1):
        if max_vertices is not None and len(problem.mesh.vertices) > max_vertices:
            return {"accepted": False, "status": "VERTEX_BUDGET", "work": metrics}
        system = PrimalSystem(problem, trusted_digest)
        flux = system.flux
        record = {"vertices": system.size, "triangles": len(problem.mesh.triangles)}
        metrics["levels"].append(record)
        field = system.frozen(initial)
        cg = system.proposals(field, iterations)
        old_step = 0
        def advance(*, pilot=False):
            nonlocal old_step, field
            start = perf_counter()
            fixed_scale = np.linalg.norm(system.rhs[system.free])
            last_trial = None
            last_step = old_step
            for step, trial, residual in cg:
                last_trial, last_step = trial, step
                if policy in ("classical_sparse", "classical_patch") and step & (step - 1) == 0:
                    metrics["stopping_checks"] += 1
                    if inspect_stopping(problem, trusted_digest, system.frozen(trial), "classical").stop:
                        break
                elif (pilot and step >= 4) or residual <= 1e-6 * fixed_scale:
                    break
            if last_trial is not None:
                field = system.frozen(last_trial)
                metrics["primal_iterations"] += last_step - old_step
                old_step = last_step
            return perf_counter() - start

        primal_seconds = advance(pilot=adaptive and policy != "cfac_screened")
        record["primal_first_seconds"] = primal_seconds
        parent = check(problem, trusted_digest, field, flux)
        metrics["certificate_checks"] += 1
        if parent.accepted:
            return {"accepted": True, "status": "ACCEPTED", "upper_sq": str(parent.upper_sq),
                    "problem": problem, "field": field, "rt0": flux, "rt1": None, "work": metrics}
        if policy == "cfac_screened":
            record["decision"] = "dual_first"
            if level < refinements:
                start = perf_counter()
                lower = screen_primal_floor(problem, trusted_digest, field)
                record["primal_disc_lower_sq"] = str(lower)
                record["floor_seconds"] = perf_counter() - start
                metrics["floor_checks"] += 1
                if lower > problem.epsilon**2:
                    record["decision"] = "refine_after_primal_floor"
                    problem, initial = _refine(problem, field)
                    trusted_digest = problem.digest()
                    continue
        elif adaptive:
            attr_start = perf_counter()
            attribution = assess(problem, trusted_digest, field, flux, sparse_iterations=6, witness=True)
            duration = perf_counter() - attr_start
            metrics["floor_checks"] += 1
            metrics["certificate_checks"] += 1
            record["floor_seconds"] = duration
            record["primal_disc_lower_sq"] = str(attribution.primal_disc_lower_sq)
            record["rt0_pair_floor"] = [str(attribution.floor.lower), str(attribution.floor.upper)]
            if attribution.verdict == "PRIMAL_SPACE_FUTILE" and level < refinements:
                record["decision"] = "refine_after_primal_floor"
                problem, initial = _refine(problem, field)
                trusted_digest = problem.digest()
                continue
            # Certified lower gains are proxies for one batch. Use measured pilot work
            # and the current volume scan as the initial costs of the two actions.
            primal_gain = min(attribution.primal_alg.lower, parent.upper_sq - problem.epsilon**2)
            dual_gain = min(attribution.dual_alg.lower, parent.upper_sq - problem.epsilon**2)
            primal_score = float(primal_gain) / max(primal_seconds, 1e-9)
            dual_score = float(dual_gain) / max(duration, 1e-9)
            record["estimated_gain_per_second"] = {"primal": primal_score, "dual": dual_score}
            record["decision"] = "primal_first" if primal_score >= dual_score else "dual_first"
            if record["decision"] == "primal_first":
                record["primal_second_seconds"] = advance()
        request = _request(problem, field, flux)
        metrics["flux_reconstructions"] += 1
        if policy == "classical_patch":
            candidate = RT1Reconstructor().reconstruct(request)
            cert = RT1Checker().verify(request, candidate, candidate.digest())
            metrics["certificate_checks"] += 1
        else:
            schedule = {"cfac_geometric": "geometric", "classical_sparse": "geometric",
                        "fixed_sparse": "geometric"}.get(policy, "predicted")
            candidate, cert, cost = optimize_rt1(request, coarse=policy != "cfac_no_coarse",
                                                 schedule=schedule, iterations=iterations)
            metrics["rt1_iterations"] += cost["iterations"]
            metrics["certificate_checks"] += cost["checks"]
            record["rt1"] = cost
        if not cert.accepted and adaptive and policy != "cfac_screened" and record["decision"] == "dual_first":
            record["primal_second_seconds"] = advance()
            request = _request(problem, field, flux)
            candidate = RTMoments(request.problem_digest, request.parent_state_digest, 1,
                                  candidate.face_moments, candidate.cell_moments)
            cert = RT1Checker().verify(request, candidate, candidate.digest())
            metrics["certificate_checks"] += 1
        record["upper_sq"] = str(cert.upper_sq)
        if cert.accepted or level == refinements:
            return {"accepted": cert.accepted, "status": "ACCEPTED" if cert.accepted else "CHECKED_MISS",
                    "upper_sq": str(cert.upper_sq), "problem": problem, "field": field,
                    "rt0": flux, "rt1": candidate, "work": metrics}
        problem, initial = _refine(problem, field)
        trusted_digest = problem.digest()
    raise ArithmeticError("refinement limit must be nonnegative")


def attribute_rt1(problem, field, rt0, candidate, iterations=6):
    """Offline four-component intervals. Neither the reporting probe nor its cost trains the policy."""
    request = _request(problem, field, rt0)
    certificate = RT1Checker().verify(request, candidate, candidate.digest())
    report = assess(problem, problem.digest(), field, rt0, sparse_iterations=iterations, witness=True)
    dual = dual_rt1_bounds(request, candidate, certificate)
    primal = report.primal_alg
    floor_lo = max(0, certificate.gap_sq - primal.upper - dual.upper)
    floor_hi = certificate.gap_sq - primal.lower - dual.lower
    disc_lo = report.primal_disc_lower_sq
    return {"primal_algebraic": [str(primal.lower), str(primal.upper)],
            "dual_algebraic": [str(dual.lower), str(dual.upper)],
            "pair_floor": [str(max(disc_lo, floor_lo)), str(floor_hi)],
            "primal_discretization": [str(disc_lo), str(floor_hi)],
            "dual_discretization": ["0", str(floor_hi - disc_lo)]}
