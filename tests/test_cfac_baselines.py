"""Stopping, refinement and acceptance through the benchmark policy entry point."""

from dataclasses import replace
from fractions import Fraction as Q

import pytest

from cfac import controller, floor
from cfac.baselines import run_baseline, run_policy
from cfac.enrichment import EnrichmentRequest, RTMoments
from cfac.prototype import ReferenceSystem, uniform_mesh
from cfac.rt1_checker import RT1Checker
from cfac.spec import Problem
from cfac.stopping import inspect_stopping


def problem(n=2, epsilon=Q(3, 20), budget=75):
    mesh = uniform_mesh(n)
    return Problem(mesh, [1] * len(mesh.triangles), 1, epsilon, budget)


@pytest.mark.parametrize("name", ["fixed", "classical"])
def test_both_baselines_pay_for_stopping_and_recheck_certified_field(name):
    p = problem()
    out = run_baseline(p, p.digest(), policy=name)
    assert out["accepted"]
    assert [row["action"] for row in out["trace"]] == [
        "setup", "stopping_check", "primal_sweep", "stopping_check", "inspect",
        "rt1_reconstruct", "rt1_inspect",
    ]
    for row in out["trace"]:
        if row["action"] == "stopping_check":
            assert row["after"] == row["before"]
            observation, state = row["observation"], row["before"]
            reference = inspect_stopping(p, p.digest(), state["primal"], name)
            assert observation["stop"] == reference.stop
            assert observation["primal_digest"] == reference.primal_digest
            assert observation["state_digest"] == state["state_digest"]
            assert Q(observation["residual_sq"]) == reference.residual_sq
    last = out["trace"][-1]
    state = last["after"]
    old = state["parent"]
    req = EnrichmentRequest(p, p.digest(), old["state_digest"], old["primal"], old["flux"], 1)
    m = RTMoments(p.digest(), old["state_digest"], 1, state["face_moments"], state["cell_moments"])
    cert = RT1Checker().verify(req, m, state["state_digest"])
    assert cert.accepted and cert.upper_sq == Q(last["observation"]["upper_sq"])
    assert sum(row["cost_ns"] for row in out["trace"]) <= out["elapsed_ns"]


@pytest.mark.parametrize("name", ["fixed", "classical"])
def test_stopping_decision_and_uninspected_rt1_never_accept(name):
    for budget, end_action in [(4, "stopping_check"), (6, "rt1_reconstruct")]:
        p = problem(budget=budget)
        out = run_baseline(p, p.digest(), policy=name)
        assert out["status"] == "BUDGET_EXHAUSTED" and not out["accepted"]
        assert out["trace"][-1]["action"] == end_action and out["actions_used"] == budget


def test_classical_stops_on_a_nonzero_algebraic_residual():
    p = problem(n=3, epsilon=Q(1, 2))
    results = [run_baseline(p, p.digest(), policy=name) for name in ("fixed", "classical")]
    assert all(item["accepted"] for item in results)
    stops = [next(t["observation"] for t in item["trace"] if t["action"] == "stopping_check"
                  and t["observation"]["stop"]) for item in results]
    assert Q(stops[1]["residual_sq"]) > Q(stops[0]["residual_sq"])
    assert 0 < Q(stops[1]["algebraic_upper_sq"]) <= Q(stops[1]["discretization_lower_sq"])/4
    assert stops[0]["algebraic_upper_sq"] is None


def test_piecewise_load_refinement_is_opt_in_and_each_child_inherits_parent():
    parent = problem(epsilon=Q(1, 1000), budget=20)
    p = replace(parent, loads=(1,-2,1,-2,1,-2,1,-2))
    legacy = controller.run(p, p.digest(), allow_rt1=True)
    assert not any(row["action"] == "refine" for row in legacy["trace"])
    for name in ("cfac", "classical", "fixed"):
        run = run_policy(p, p.digest(), policy=name)
        child = next(row for row in run["trace"] if row["action"] == "refine")
        spec = run["problems"][child["after"]["problem_digest"]]
        assert spec["loads"] == [str(value) for value in p.loads for _ in range(4)]
        assert len(spec["triangles"]) == 4*len(p.mesh.triangles)
        assert spec["max_actions"] == p.max_actions
        assert run["actions_used"] <= p.max_actions


@pytest.mark.parametrize("name", ["no_floor","no_witness","no_coarse"])
def test_ablation_never_calls_disabled_mechanism(name, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(f"{name} invoked disabled operation")

    target, attribute = {
        "no_floor": (controller, "assess"),
        "no_witness": (floor, "_probe"),
        "no_coarse": (ReferenceSystem, "coarse_dual_step"),
    }[name]
    monkeypatch.setattr(target, attribute, forbidden)
    p = problem(epsilon=Q(1, 8))
    out = run_policy(p, p.digest(), policy=name)
    assert out["accepted"] and out["trace"][-1]["observation"]["accepted"]
    if name == "no_witness":
        assert any(row["action"] == "algebraic_probe" for row in out["trace"])
    if name == "no_floor":
        assert all(s["observation"]["floor"] is None for s in out["trace"] if s["observation"])


def test_invalid_policy_and_changed_spec_fail_closed():
    p = problem()
    with pytest.raises(ValueError):
        run_policy(p, p.digest(), policy="unknown")
    for name in ("fixed", "classical"):
        out = run_policy(replace(p, epsilon=2), p.digest(), policy=name)
        assert out["status"] == "INVALID_INPUT" and not out["accepted"]
