"""Exercise enrichment through the controller and recheck its archived states."""

from dataclasses import replace
from fractions import Fraction as Q
import json
from pathlib import Path

from cfac.controller import run
from cfac.checker import check
from cfac.enrichment import EnrichmentRequest, RTMoments
from cfac.evaluation import optimal_fields
from cfac.prototype import ReferenceSystem, uniform_mesh
from cfac.rt1_checker import RT1Checker
from cfac.rt1_reconstruction import RT1Reconstructor
from cfac.spec import Problem


def unit_load(epsilon=Q(3, 20), budget=24):
    mesh = uniform_mesh(2)
    return Problem(mesh, (1,) * len(mesh.triangles), 1, epsilon, budget)


def test_rt1_current_field_can_accept_after_rt0_pair_futility():
    p = unit_load()
    archive = run(p, p.digest(), allow_rt1=True)
    assert archive["accepted"]
    actions = [row["action"] for row in archive["trace"]]
    assert "refine" not in actions
    assert actions[-2:] == ["rt1_reconstruct", "rt1_inspect"]
    floor_proof, construction, inspection = archive["trace"][-3:]
    assert floor_proof["observation"]["verdict"] == "PAIR_FUTILE"
    assert Q(floor_proof["observation"]["floor"][0]) > p.epsilon**2
    old, new = construction["before"], construction["after"]
    assert construction["observation"] is None
    assert new["parent"] == old == floor_proof["after"]
    assert inspection["before"] == inspection["after"] == new
    req = EnrichmentRequest(p, p.digest(), old["state_digest"], old["primal"], old["flux"], 1)
    moments = RTMoments(p.digest(), old["state_digest"], 1, new["face_moments"], new["cell_moments"])
    cert = RT1Checker().verify(req, moments, new["state_digest"])
    assert cert.accepted and cert.exact_equilibrium
    assert cert.upper_sq == Q(inspection["observation"]["upper_sq"])
    assert inspection["observation"]["floor"] is None
    assert Q(inspection["gap_gain_sq"]) == Q(floor_proof["observation"]["gap_sq"]) - cert.gap_sq


def test_budget_must_cover_separate_rt1_inspection():
    p = unit_load()
    complete = run(p, p.digest(), allow_rt1=True)
    construction = next(row for row in complete["trace"] if row["action"] == "rt1_reconstruct")
    limited = replace(p, max_actions=construction["step"])
    result = run(limited, limited.digest(), allow_rt1=True)
    assert result["status"] == "BUDGET_EXHAUSTED" and not result["accepted"]
    assert result["trace"][-1]["action"] == "rt1_reconstruct"
    assert result["trace"][-1]["observation"] is None


def test_construction_failure_keeps_parent_and_charges_attempt(monkeypatch):
    def cannot_solve(self, request):
        raise ArithmeticError("patch solve failed")

    monkeypatch.setattr(RT1Reconstructor, "reconstruct", cannot_solve)
    p = unit_load()
    result = run(p, p.digest(), allow_rt1=True)
    failed = next(row for row in result["trace"] if row["action"] == "rt1_reconstruct")
    assert failed["before"] == failed["after"] and failed["cost_ns"] > 0
    assert failed["next_decision"] == "refine" and "patch solve failed" in failed["next_reason"]
    assert result["trace"][failed["step"]]["action"] == "refine"


def test_proven_primal_futility_skips_rt1_on_that_space():
    p = unit_load(Q(1, 100))
    result = run(p, p.digest(), allow_rt1=True)
    row = next(row for row in result["trace"] if row["next_decision"] == "refine")
    assert row["observation"]["verdict"] == "PRIMAL_SPACE_FUTILE"
    assert all(step["action"] != "rt1_reconstruct" for step in result["trace"][:row["step"]])


def test_patch_trial_can_be_worse_than_optimal_rt0_even_at_galerkin():
    mesh = uniform_mesh(1)
    p = Problem(mesh, (1, 1), 1, Q(41, 200), 8)
    anchor = p.digest()
    v, q = optimal_fields(ReferenceSystem(p))
    parent = check(p, anchor, v, q)
    req = EnrichmentRequest(p, anchor, parent.state_digest, v, q, 1)
    moments = RT1Reconstructor().reconstruct(req)
    trial = RT1Checker().verify(req, moments, moments.digest())
    assert parent.gap_sq == Q(1, 24) < p.epsilon**2 < Q(13, 300) == trial.gap_sq
    assert parent.accepted and not trial.accepted and trial.exact_equilibrium


def test_optional_policy_receipt_replays_both_success_and_failed_trial():
    from research.cfac.controller_demo import campaign, deterministic

    path = Path(__file__).resolve().parents[1] / "evidence/cfac/rt1-policy.json"
    saved = json.loads(path.read_bytes())
    assert deterministic(campaign(allow_rt1=True)) == deterministic(saved)
    for policies in saved.values():
        for result in policies.values():
            costs = [row["cost_ns"] for row in result["trace"]]
            assert all(type(c) is int and c >= 0 for c in costs)
            assert sum(costs) <= result["elapsed_ns"]
    trial = next(row for row in saved["tighter"]["RT0_then_RT1"]["trace"]
                 if row["action"] == "rt1_inspect")
    assert trial["next_decision"] == "refine" and not trial["observation"]["accepted"]
    assert trial["observation"]["verdict"] == "UNDETERMINED"
    assert trial["observation"]["floor"] is None
