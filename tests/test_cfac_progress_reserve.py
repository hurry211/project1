"""Progress detection, budget boundaries and real field checks for the refined RT1 trial."""
from dataclasses import replace
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from cfac import controller
from cfac.baselines import run_policy
from cfac.checker import check
from cfac.enrichment import EnrichmentRequest, RTMoments
from cfac.prototype import uniform_mesh
from cfac.rt1_checker import RT1Checker
from cfac.rt1_reconstruction import RT1Reconstructor
from cfac.spec import Mesh, Problem


def run_stripes(budget=128, epsilon=Q(1, 9)):
    p = Problem(uniform_mesh(3), (2, 2, -1, -1, -2, -2) * 3, 1, epsilon, budget)
    return run_policy(p, p.digest(), policy="cfac_reserve")


def current_problem(journal, state):
    s = journal["problems"][state["problem_digest"]]
    return Problem(Mesh(s["vertices"], s["triangles"]), s["loads"], s["diffusion"], s["epsilon"], s["max_actions"])


def trigger_steps(journal):
    return [t["step"] for t in journal["trace"] if
            (t["observation"] or {}).get("progress_reserve", {}).get("triggered")]


def test_progress_requires_two_slow_distinct_intervals_and_resets():
    progress = controller._ProgressReserve()

    def observe(digest, excess, verdict="PAIR_SUFFICIENT"):
        report = SimpleNamespace(state_digest=digest, verdict=verdict,
                                 certificate=SimpleNamespace(upper_sq=1 + Q(excess)))
        return progress.observe(report, Q(1), 4)

    assert observe("a", 100)["slow_intervals"] == 0
    assert observe("b", 90)["slow_intervals"] == 1
    assert observe("b", 90)["slow_intervals"] == 1
    assert observe("c", 80)["slow_intervals"] == 0
    assert observe("d", 80)["slow_intervals"] == 1
    assert observe("u", 80, "UNDETERMINED")["slow_intervals"] == 0
    assert observe("e", 80)["previous_excess_sq"] is None
    assert not observe("f", 72)["triggered"]
    assert observe("g", 73)["triggered"]


@pytest.mark.parametrize("budget,accepted", [(44, False), (45, True)])
def test_signed_regression_requires_four_remaining_actions(budget, accepted):
    run = run_stripes(budget)
    trace = run["trace"]
    probe = trace[40]["observation"]["progress_reserve"]
    assert probe["remaining_actions"] == budget - 41 and probe["slow_intervals"] == 2
    assert run["actions_used"] == budget and run["accepted"] is accepted
    if not accepted:
        assert trigger_steps(run) == [] and run["status"] == "BUDGET_EXHAUSTED"
        return
    assert trigger_steps(run) == [41]
    sweep, rt0, construction, inspection = trace[-4:]
    assert [s["action"] for s in trace[-4:]] == ["primal_sweep", "inspect", "rt1_reconstruct", "rt1_inspect"]
    assert rt0["before"] == rt0["after"] == sweep["after"] == construction["before"]
    assert not rt0["observation"]["accepted"]
    state = inspection["after"]
    parent = state["parent"]
    p = current_problem(run, state)
    request = EnrichmentRequest(p, p.digest(), parent["state_digest"], parent["primal"], parent["flux"], 1)
    candidate = RTMoments(p.digest(), parent["state_digest"], 1, state["face_moments"], state["cell_moments"])
    cert = RT1Checker().verify(request, candidate, state["state_digest"])
    assert cert.accepted and Q(inspection["observation"]["upper_sq"]) == cert.upper_sq <= p.epsilon**2
    assert Q(inspection["gap_gain_sq"]) == Q(rt0["observation"]["gap_sq"]) - cert.gap_sq


@pytest.mark.parametrize("budget", [45, 48])
def test_valid_missed_rt1_uses_charged_restore_only_with_remaining_budget(monkeypatch, budget):
    reconstruct = RT1Reconstructor.reconstruct

    def poorer_flux(self, request):
        candidate = reconstruct(self, request)
        if len(request.problem.loads) == 72:
            # Preserve the RT1 layout. Changed interior moments produce a larger certified defect.
            return replace(candidate, cell_moments=[(x + 1, y) for x, y in candidate.cell_moments])
        return candidate

    monkeypatch.setattr(RT1Reconstructor, "reconstruct", poorer_flux)
    run = run_stripes(budget)
    assert trigger_steps(run) == [41] and not run["accepted"] and run["actions_used"] == budget
    miss = run["trace"][44]
    assert miss["observation"]["checker_status"] == "CERTIFIED" and not miss["observation"]["accepted"]
    assert miss["before"] == miss["after"] and miss["next_decision"] == "restore_rt0"
    if budget > 45:
        restored, inspected = run["trace"][45:47]
        assert restored["action"] == "restore_rt0" and restored["after"] == miss["before"]["parent"]
        assert inspected["action"] == "inspect" and inspected["before"] == restored["after"]


@pytest.mark.parametrize("fault", ["construction", "malformed"])
def test_reserved_construction_and_checker_failures(monkeypatch, fault):
    reconstruct = RT1Reconstructor.reconstruct

    def faulty(self, request):
        if len(request.problem.loads) == 72 and fault == "construction":
            raise ArithmeticError("test solve failed")
        candidate = reconstruct(self, request)
        if len(request.problem.loads) == 72:
            return replace(candidate, cell_moments=())
        return candidate

    monkeypatch.setattr(RT1Reconstructor, "reconstruct", faulty)
    run = run_stripes(48)
    assert trigger_steps(run) == [41] and not run["accepted"]
    if fault == "construction":
        failure = run["trace"][43]
        assert failure["action"] == "rt1_reconstruct" and failure["before"] == failure["after"]
        assert failure["next_decision"] == "inspect" and "test solve failed" in failure["next_reason"]
        assert run["status"] == "BUDGET_EXHAUSTED"
    else:
        assert run["trace"][-1]["action"] == "rt1_inspect" and run["status"] == "INVALID_INPUT"


def test_reserved_sweep_can_deliver_a_rechecked_rt0_field_early(monkeypatch):
    # Isolate the post-trigger path on the previously measured loose stripes.
    # After its refined primal sweep this case already has an acceptable RT0 field.
    choose = controller._choose

    def request_tight_probe(report, action, refined, can_refine, dual_steps):
        selected = choose(report, action, refined, can_refine, dual_steps)
        if refined and action == "inspect" and selected[0] == "primal_sweep":
            return "witness_probe", "controlled opportunity for an early RT0 acceptance"
        return selected

    seen = 0

    def controlled_trigger(self, report, threshold, remaining):
        nonlocal seen
        seen += 1
        self.attempted = self.active = seen == 2
        return {"triggered": self.active}

    monkeypatch.setattr(controller, "_choose", request_tight_probe)
    monkeypatch.setattr(controller._ProgressReserve, "observe", controlled_trigger)
    run = run_stripes(epsilon=Q(1, 6))
    assert trigger_steps(run) == [run["actions_used"] - 2] and run["accepted"]
    assert [s["action"] for s in run["trace"][-2:]] == ["primal_sweep", "inspect"]
    last = run["trace"][-1]["after"]
    p = current_problem(run, last)
    assert check(p, p.digest(), last["primal"], last["flux"]).accepted
