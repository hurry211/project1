"""Frozen primal stopping policies for the matched-accuracy experiment."""

from fractions import Fraction
from functools import partial

from .controller import _run, run
from .stopping import inspect_stopping


BALANCE_FRACTION = Fraction(1, 2)
RELATIVE_TOLERANCE = Fraction(1, 10**6)


def _stopping_observation(problem, anchor, primal, *, rule):
    result = inspect_stopping(problem, anchor, primal, rule,
                              fraction=BALANCE_FRACTION, relative_tolerance=RELATIVE_TOLERANCE)
    return {
        "rule": result.rule, "stop": result.stop, "primal_digest": result.primal_digest,
        "residual_sq": str(result.residual_sq), "rhs_sq": str(result.rhs_sq),
        "algebraic_upper_sq": None if result.algebraic_upper_sq is None else str(result.algebraic_upper_sq),
        "discretization_lower_sq": (None if result.discretization_lower_sq is None
                                    else str(result.discretization_lower_sq)),
    }


def run_baseline(problem, trusted_digest, *, policy):
    """Stop the primal solver, inspect tree RT0, then try patch RT1 and refinement.

    Every stopping inspection, primal sweep, certificate action and refinement
    goes through the same budget and journal as the CFAC controller.
    """
    if policy not in ("classical", "fixed"):
        raise ValueError(f"unknown baseline {policy}")

    return _run(problem, trusted_digest, allow_rt1=True, refine_piecewise=True,
                stopping=partial(_stopping_observation, rule=policy))


def run_policy(problem, trusted_digest, *, policy):
    """Dispatch a frozen benchmark policy without changing the foundation defaults."""
    if policy in ("fixed", "classical"):
        return run_baseline(problem, trusted_digest, policy=policy)
    if policy == "cfac_reserve":
        return _run(problem, trusted_digest, allow_rt1=True, refine_piecewise=True, progress_reserve=True)
    if policy == "cfac_retry":
        return _run(problem, trusted_digest, allow_rt1=True, refine_piecewise=True, progress_reserve=True,
                    futile_stopping=partial(_stopping_observation, rule="classical"))
    if policy not in ("cfac", "cfac_rt0", "no_coarse", "no_witness", "no_floor"):
        raise ValueError(f"unknown benchmark policy {policy}")
    ablation = None if policy in ("cfac", "cfac_rt0") else policy
    return run(problem, trusted_digest, allow_rt1=(policy != "cfac_rt0"),
               ablation=ablation, refine_piecewise=True)
