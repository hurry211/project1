"""Exercise the refined-pair futility gate on the historical checkerboard."""

from dataclasses import replace
from fractions import Fraction as Q

import pytest

from cfac import baselines, controller
from cfac.baselines import run_policy
from cfac.enrichment import EnrichmentRequest, RTMoments
from cfac.prototype import uniform_mesh
from cfac.rt1_checker import RT1Checker
from cfac.spec import Mesh, Problem
from cfac.stopping import inspect_stopping


def run_checker(budget=128, policy="cfac_retry"):
    loads = [value for value in (2, -1, 2, -1, 2, -1, 2, -1, 2) for _ in range(2)]
    problem = Problem(uniform_mesh(3), loads, 1, Q(1, 12), budget)
    return run_policy(problem, problem.digest(), policy=policy)


def test_refined_futile_pair_uses_current_primal_and_independent_rt1_check():
    old = run_checker(policy="cfac_reserve")
    run = run_checker()
    assert old["status"] == "NO_ELIGIBLE_ACTION" and old["actions_used"] == 16
    assert run["accepted"] and run["actions_used"] == 31
    gate_events = [e for e in run["trace"] if e["action"] == "futile_stopping_check"]
    assert [e["step"] for e in gate_events] == list(range(15, 30, 2))
    source = run["trace"][13]
    assert source["observation"]["verdict"] == "PAIR_FUTILE"
    assert source["observation"]["futile_gate"] == {"entered": True, "remaining_actions": 114}
    anchor = source["after"]["problem_digest"]
    record = run["problems"][anchor]
    p = Problem(Mesh(record["vertices"], record["triangles"]), record["loads"],
                record["diffusion"], record["epsilon"], record["max_actions"])
    assert p.digest() == anchor and len(p.loads) == 72
    for e in gate_events:
        state, obs = e["after"], e["observation"]
        assert e["before"] == state and obs["state_digest"] == state["state_digest"]
        reference = inspect_stopping(p, anchor, state["primal"], "classical", fraction=Q(1, 2))
        assert obs["primal_digest"] == reference.primal_digest
        assert Q(obs["algebraic_upper_sq"]) == reference.algebraic_upper_sq
        assert Q(obs["discretization_lower_sq"]) == reference.discretization_lower_sq
        assert obs["stop"] == (reference.algebraic_upper_sq <= reference.discretization_lower_sq / 4)
        assert e["next_decision"] == ("rt1_reconstruct" if obs["stop"] else "primal_sweep")
    last = run["trace"][-1]
    enriched = last["before"]
    parent = enriched["parent"]
    assert parent == gate_events[-1]["after"] and last["action"] == "rt1_inspect"
    req = EnrichmentRequest(p, anchor, parent["state_digest"], parent["primal"], parent["flux"], 1)
    moments = RTMoments(anchor, parent["state_digest"], 1, enriched["face_moments"], enriched["cell_moments"])
    cert = RT1Checker().verify(req, moments, enriched["state_digest"])
    assert cert.accepted and cert.upper_sq == Q(last["observation"]["upper_sq"]) <= p.epsilon**2
    assert last["observation"]["gap_gain_from_state_digest"] == source["after"]["state_digest"]
    assert Q(last["gap_gain_sq"]) == Q(source["observation"]["gap_sq"]) - cert.gap_sq


@pytest.mark.parametrize("budget,entered,checks,last,accepted", [
    (15, False, 0, "rt1_reconstruct", False),
    (16, False, 0, "rt1_inspect", False),
    (17, True, 1, "rt1_inspect", False),
    (18, True, 1, "rt1_inspect", False),
    (19, True, 2, "rt1_inspect", False),
    (30, True, 7, "rt1_inspect", True),
])
def test_budget_protects_reconstruction_and_independent_check(budget, entered, checks, last, accepted):
    run = run_checker(budget)
    assert run["actions_used"] <= budget and run["accepted"] is accepted
    trace = run["trace"]
    assert trace[-1]["action"] == last
    assert trace[13]["observation"]["futile_gate"] == {"entered": entered, "remaining_actions": budget - 14}
    gates = [e for e in trace if e["action"] == "futile_stopping_check"]
    assert len(gates) == checks
    if entered:
        assert gates[-1]["observation"]["budget_limited"] and not gates[-1]["observation"]["stop"]
        assert gates[-1]["observation"]["remaining_actions"] in (2, 3)
        assert [e["action"] for e in trace[-2:]] == ["rt1_reconstruct", "rt1_inspect"]
        assert trace[-1]["observation"]["accepted"] is accepted


def test_primal_space_futility_reaches_original_terminal(monkeypatch):
    assess = controller.assess

    def primal_futile(p, *args, **kwargs):
        decision = assess(p, *args, **kwargs)
        if len(p.loads) == 72 and decision.verdict == "PAIR_FUTILE":
            return replace(decision, verdict="PRIMAL_SPACE_FUTILE")
        return decision

    monkeypatch.setattr(controller, "assess", primal_futile)
    run = run_checker()
    assert run["status"] == "NO_ELIGIBLE_ACTION"
    assert run["trace"][-1]["observation"]["verdict"] == "PRIMAL_SPACE_FUTILE"
    assert not any(e["action"] == "futile_stopping_check" for e in run["trace"])


def test_gate_failure_cannot_deliver_a_field(monkeypatch):
    def failed_bound(*args, **kwargs):
        raise ArithmeticError("unavailable algebraic bound")

    monkeypatch.setattr(baselines, "inspect_stopping", failed_bound)
    run = run_checker()
    assert run["status"] == "INVALID_INPUT" and not run["accepted"]
    assert run["trace"][-1]["action"] == "futile_stopping_check"
    assert run["stop_reason"] == "unavailable algebraic bound"
