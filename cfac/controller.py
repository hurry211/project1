"""Budgeted, deterministic CFAC demonstration for small rational CPU problems."""

from time import perf_counter_ns

from .checker import edges, validate_problem
from .floor import assess
from .prototype import ReferenceSystem
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
    # This action is enabled only for a constant P0 original.
    child = Problem(Mesh(vertices, children), [problem.loads[0]] * len(children),
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


def run(problem, trusted_digest):
    """Return a JSON-serializable decision journal, including actual elapsed times.

    One action buys one setup, inspection, diagnostic inspection, correction or
    refinement. Only a completed inspection can accept its current field.
    """
    start = perf_counter_ns()
    trace, problems = [], {}
    initial_digest = trusted_digest
    action, reason = "setup", "assemble the initial P1 system and lift the frozen load"
    status = "BUDGET_EXHAUSTED"
    state = previous_gap = None
    refined, dual_steps = False, 0
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
                except ValueError as exc:
                    next_action, next_reason = "INVALID_INPUT", str(exc)
        elif action in ("inspect", "witness_probe"):
            diagnostic = action == "witness_probe"
            report = assess(problem, trusted_digest, primal, flux, tight=diagnostic, witness=diagnostic)
            observation = _observation(report)
            if report.certificate.state_digest != state["state_digest"]:
                next_action, next_reason = "INVALID_INPUT", "checker could not bind the inspected state"
            else:
                gap = report.certificate.gap_sq
                gain = None if previous_gap is None else str(previous_gap - gap)
                previous_gap = gap
                can_refine = problem.original_affine is None and len(set(problem.loads)) == 1
                next_action, next_reason = _choose(report, action, refined, can_refine, dual_steps)
        elif action == "refine":
            problem, primal = _refine(problem, primal)
            trusted_digest = problem.digest()
            problems[trusted_digest] = _problem_record(problem)
            validate_problem(problem, trusted_digest)
            system = ReferenceSystem(problem)
            flux = system.feasible_flux
            refined, dual_steps, previous_gap = True, 0, None
            state = _state(trusted_digest, primal, flux)
        else:
            if action == "primal_sweep":
                primal = system.primal_step(primal)
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
