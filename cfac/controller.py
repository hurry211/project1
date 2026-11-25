"""Budgeted, deterministic CFAC demonstration for small rational CPU problems."""

from time import perf_counter_ns

from .checker import check, edges, validate_problem
from .enrichment import EnrichmentRequest
from .floor import FloorAssessment, assess
from .prototype import ReferenceSystem
from .rt1_checker import RT1Checker
from .rt1_reconstruction import RT1Reconstructor
from .spec import Mesh, Problem, digest_parts


def _problem_record(problem):
    return {
        "vertices": [[str(x), str(y)] for x, y in problem.mesh.vertices],
        "triangles": [list(cell) for cell in problem.mesh.triangles],
        "loads": list(map(str, problem.loads)), "diffusion": str(problem.diffusion),
        "epsilon": str(problem.epsilon), "max_actions": problem.max_actions,
        "original_affine": None if problem.original_affine is None else list(map(str, problem.original_affine)),
    }


def _state(problem_digest, primal, flux):
    u, q = list(map(str, primal)), list(map(str, flux))
    return {"problem_digest": problem_digest, "primal": u, "flux": q,
            "state_digest": digest_parts([problem_digest, u, q])}


def _observation(assessment):
    c = assessment.certificate
    info = {"verdict": assessment.verdict, "checker_status": c.status, "accepted": c.accepted,
            "exact_equilibrium": c.exact_equilibrium, "state_digest": c.state_digest, "reason": c.reason}
    for name in ("gap_sq", "upper_sq", "defect_sq", "oscillation_sq"):
        value = getattr(c, name)
        info[name] = None if value is None else str(value)
    for name in ("floor", "primal_alg", "dual_alg"):
        bound = getattr(assessment, name)
        info[name] = None if bound is None else [str(bound.lower), str(bound.upper)]
    for name in ("witness_lower_sq", "primal_disc_lower_sq"):
        value = getattr(assessment, name)
        info[name] = None if value is None else str(value)
    return info


def _refine(problem, primal):
    """Derive a child pair from the retained parent. Prolong P1 by interpolation."""
    if problem.original_affine is not None:
        raise ValueError("this refinement transfers original cellwise constant loads")
    vertices = list(problem.mesh.vertices)
    field = list(primal)
    middle = {}
    for i, j in edges(problem.mesh):
        middle[i, j] = len(vertices)
        vertices.append(tuple((x + y) / 2 for x, y in zip(vertices[i], vertices[j])))
        field.append((primal[i] + primal[j]) / 2)
    children = []
    for i, j, k in problem.mesh.triangles:
        a, b, c = (middle[min(u, v), max(u, v)] for u, v in ((i, j), (j, k), (k, i)))
        children.extend(((i, a, c), (a, j, b), (c, b, k), (a, b, c)))
    child = Problem(Mesh(vertices, children), [f for f in problem.loads for _ in range(4)],
                    problem.diffusion, problem.epsilon, problem.max_actions)
    return child, tuple(field)


def _choose(assessment, action, refined, can_refine, dual_steps):
    if assessment.certificate.accepted:
        return "ACCEPTED", "the independent checker accepted the inspected field"
    if assessment.verdict in ("PRIMAL_SPACE_FUTILE", "PAIR_FUTILE"):
        if can_refine and not refined:
            return "refine", "the certified floor exceeds the threshold on this pair"
        return "NO_ELIGIBLE_ACTION", "the demonstrated floor requires a space change outside the remaining actions"
    if assessment.verdict == "UNDETERMINED":
        if action == "inspect" and assessment.primal_alg is not None:
            return "witness_probe", "tighten both algebraic bounds and inspect a P2 witness"
        return "UNDETERMINED", "the available certified evidence does not select a correction"
    if assessment.primal_alg.lower > assessment.dual_alg.lower:
        return "primal_sweep", "the primal algebraic error has the larger lower bound"
    if dual_steps == 0:
        return "dual_coarse", "try the coarse stream space before local dual sweeps"
    return "dual_local", "reduce dual algebraic error in curl-of-hat directions"


def run(problem, trusted_digest, *, allow_rt1=False, ablation=None, refine_piecewise=False):
    """Return a JSON-serializable decision journal, including actual elapsed times.

    One action buys one setup, inspection, diagnostic inspection, correction or
    refinement. With allow_rt1, a certified RT0 pair floor can trigger one
    RT1 construction and a separately budgeted inspection on the same mesh.
    Only a completed inspection can accept its current field.
    """
    return _run(problem, trusted_digest, allow_rt1=allow_rt1, ablation=ablation,
                refine_piecewise=refine_piecewise)


def _run(problem, trusted_digest, *, allow_rt1, ablation=None, refine_piecewise=False, stopping=None):
    """Common numerical actions and independent acceptance for all scheduling policies."""
    start = perf_counter_ns()
    if ablation not in (None, "no_coarse", "no_witness", "no_floor"):
        raise ValueError(f"unknown CFAC ablation {ablation}")
    trace, problems = [], {}
    initial_digest = trusted_digest
    action, reason = "setup", "assemble the initial P1 system and lift the frozen load"
    status = "BUDGET_EXHAUSTED"
    state = previous_gap = None
    refined, dual_steps = False, 0
    prepared_rt1 = False
    scheduled_sweeps = ("primal_sweep", "dual_coarse", "dual_local")
    next_sweep = 0
    for index in range(problem.max_actions):
        tic = perf_counter_ns()
        before, observation, gain = state, None, None
        next_action, next_reason = "inspect", "inspect the resulting immutable state"
        if action == "setup":
            if problem.digest() != trusted_digest:
                next_action, next_reason = "INVALID_INPUT", "frozen problem digest mismatch"
            else:
                problems[trusted_digest] = _problem_record(problem)
                try:
                    validate_problem(problem, trusted_digest)
                    system = ReferenceSystem(problem)
                    primal, flux = system.zero_primal(), system.feasible_flux
                    state = _state(trusted_digest, primal, flux)
                    if stopping:
                        next_action, next_reason = "stopping_check", "inspect the primal stopping rule on this level"
                except ValueError as exc:
                    next_action, next_reason = "INVALID_INPUT", str(exc)
        elif action == "stopping_check":
            try:
                observation = stopping(problem, trusted_digest, primal)
            except ValueError as exc:
                next_action, next_reason = "INVALID_INPUT", str(exc)
            else:
                observation["state_digest"] = state["state_digest"]
                if observation["stop"]:
                    next_action, next_reason = "inspect", "the primal stopping rule permits certificate inspection"
                else:
                    next_action, next_reason = "primal_sweep", "the primal stopping rule requires another sweep"
        elif action in ("inspect", "witness_probe", "algebraic_probe"):
            if stopping or ablation == "no_floor":
                cert = check(problem, trusted_digest, primal, flux)
                report = FloorAssessment(cert, "ACCEPTED" if cert.accepted else "UNDETERMINED")
            else:
                report = assess(problem, trusted_digest, primal, flux, tight=(action != "inspect"),
                                witness=(action == "witness_probe"))
            observation = _observation(report)
            if report.certificate.state_digest != state["state_digest"]:
                next_action, next_reason = "INVALID_INPUT", "checker could not bind the inspected state"
            else:
                gap = report.certificate.gap_sq
                gain = None if previous_gap is None else str(previous_gap - gap)
                previous_gap = gap
                can_refine = problem.original_affine is None and (refine_piecewise or len(set(problem.loads)) == 1)
                next_action, next_reason = _choose(report, action, refined, can_refine, dual_steps)
                if stopping and not report.certificate.accepted:
                    next_action, next_reason = "rt1_reconstruct", "try patch RT1 after the stopped field missed the RT0 target"
                elif ablation == "no_floor" and not report.certificate.accepted:
                    if next_sweep < len(scheduled_sweeps):
                        next_action, next_reason = scheduled_sweeps[next_sweep], "take the next predeclared sweep on this mesh"
                        next_sweep += 1
                    elif allow_rt1:
                        next_action, next_reason = "rt1_reconstruct", "try patch RT1 after the scheduled sweeps"
                    elif can_refine and not refined:
                        next_action, next_reason = "refine", "the scheduled sweeps missed the target"
                    else:
                        next_action, next_reason = "NO_ELIGIBLE_ACTION", "the floor-free action schedule is exhausted"
                elif ablation == "no_witness" and next_action == "witness_probe":
                    next_action, next_reason = "algebraic_probe", "tighten algebraic bounds without a P2 witness"
                elif ablation == "no_coarse" and next_action == "dual_coarse":
                    next_action, next_reason = "dual_local", "use a local sweep with coarse correction disabled"
                elif allow_rt1 and report.verdict == "PAIR_FUTILE":
                    if not prepared_rt1 and report.primal_alg.lower > 0:
                        prepared_rt1 = True
                        next_action = "primal_sweep"
                        next_reason = "reduce proven primal algebraic error once before trying a richer flux"
                    else:
                        next_action = "rt1_reconstruct"
                        next_reason = "try RT1 on this field after the certified P1/RT0 pair-futility test"
        elif action == "rt1_reconstruct":
            request = EnrichmentRequest(problem, trusted_digest, state["state_digest"], primal, flux, 1)
            try:
                candidate = RT1Reconstructor().reconstruct(request)
            except (ArithmeticError, ValueError) as exc:
                next_action = "refine" if can_refine and not refined else "NO_ELIGIBLE_ACTION"
                next_reason = f"RT1 construction failed on the current field: {exc}"
            else:
                candidate_digest = candidate.digest()
                state = {"space": "P1/RT1", "problem_digest": trusted_digest,
                         "state_digest": candidate_digest, "parent": before, "order": candidate.order,
                         "face_moments": [list(map(str, row)) for row in candidate.face_moments],
                         "cell_moments": [list(map(str, row)) for row in candidate.cell_moments]}
                next_action, next_reason = "rt1_inspect", "independently verify the retained RT1 moments"
        elif action == "rt1_inspect":
            cert = RT1Checker().verify(request, candidate, candidate_digest)
            verdict = "ACCEPTED" if cert.accepted else "UNDETERMINED"
            observation = _observation(FloorAssessment(cert, verdict))
            observation["original_residual_sq"] = (
                None if cert.original_residual_sq is None else str(cert.original_residual_sq)
            )
            if cert.status != "CERTIFIED" or cert.state_digest != candidate_digest:
                next_action, next_reason = "INVALID_INPUT", "the RT1 checker could not bind a certified upper bound"
            else:
                gain = str(previous_gap - cert.gap_sq)
                if cert.accepted:
                    next_action, next_reason = "ACCEPTED", "the RT1 checker accepted this P1 field"
                elif can_refine and not refined:
                    next_action, next_reason = "refine", "try the allowed refinement after the RT1 trial missed the target"
                else:
                    next_action, next_reason = "NO_ELIGIBLE_ACTION", "the RT1 trial missed the target and no refinement remains"
        elif action == "refine":
            problem, primal = _refine(problem, primal)
            trusted_digest = problem.digest()
            problems[trusted_digest] = _problem_record(problem)
            validate_problem(problem, trusted_digest)
            system = ReferenceSystem(problem)
            flux = system.feasible_flux
            refined, dual_steps, previous_gap = True, 0, None
            prepared_rt1 = False
            next_sweep = 0
            state = _state(trusted_digest, primal, flux)
            if stopping:
                next_action, next_reason = "stopping_check", "inspect the stopping rule on the prolonged field"
        else:
            if action == "primal_sweep":
                primal = system.primal_step(primal)
                if stopping:
                    next_action, next_reason = "stopping_check", "inspect the stopping rule after the primal sweep"
            elif action == "dual_coarse":
                flux = system.coarse_dual_step(flux)
                dual_steps += 1
            else:
                flux = system.local_dual_sweep(flux)
                dual_steps += 1
            state = _state(trusted_digest, primal, flux)
        trace.append({"step": index + 1, "action": action, "reason": reason, "before": before,
                      "after": state, "observation": observation, "gap_gain_sq": gain,
                      "next_decision": next_action, "next_reason": next_reason,
                      "cost_ns": perf_counter_ns() - tic})
        if next_action.isupper():
            status = next_action
            break
        action, reason = next_action, next_reason
    stop_reason = ("the remaining action budget is zero" if status == "BUDGET_EXHAUSTED"
                   else trace[-1]["next_reason"])
    return {"status": status, "stop_reason": stop_reason,
            "accepted": status == "ACCEPTED", "budget": problem.max_actions,
            "initial_digest": initial_digest,
            "actions_used": len(trace), "elapsed_ns": perf_counter_ns() - start,
            "trace": trace, "problems": problems}
