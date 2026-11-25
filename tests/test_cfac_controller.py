"""Controller traces replay against the independent certificate checker."""

from dataclasses import replace
from fractions import Fraction as F
import json

import pytest

from cfac.checker import check
from cfac.controller import run
from cfac.prototype import uniform_mesh
from cfac.spec import Mesh, Problem


def problem_with(epsilon, budget=20):
    mesh = uniform_mesh(2)
    return Problem(mesh, [1] * len(mesh.triangles), 1, epsilon, budget)


def replay(result):
    journal = json.loads(json.dumps(result))
    specs = {}
    for digest, s in journal["problems"].items():
        mesh = Mesh(s["vertices"], s["triangles"])
        p = Problem(mesh, s["loads"], s["diffusion"], s["epsilon"], s["max_actions"], s["original_affine"])
        assert p.digest() == digest
        specs[digest] = p
    previous = None
    last = {}
    for row in journal["trace"]:
        assert row["before"] == previous
        assert row["cost_ns"] >= 0
        previous = row["after"]
        if previous is None:
            continue
        digest = previous["problem_digest"]
        cert = check(specs[digest], digest, previous["primal"], previous["flux"])
        assert cert.exact_equilibrium
        assert cert.state_digest == previous["state_digest"]
        if row["observation"] is not None:
            o = row["observation"]
            assert row["before"] == row["after"]
            assert o["accepted"] == cert.accepted
            assert o["state_digest"] == cert.state_digest
            assert F(o["upper_sq"]) == cert.upper_sq
            gain = row["gap_gain_sq"]
            assert gain is None if digest not in last else F(gain) == last[digest] - cert.gap_sq
            last[digest] = cert.gap_sq
    if result["accepted"]:
        assert journal["trace"][-1]["observation"]["accepted"]
    assert len(journal["trace"]) == journal["actions_used"] <= journal["budget"]
    assert sum(r["cost_ns"] for r in journal["trace"]) <= journal["elapsed_ns"]


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 5])
def test_exhaustion_never_promotes_uninspected_candidates(budget):
    p = problem_with(F(1, 100), budget)
    result = run(p, p.digest())
    assert not result["accepted"]
    if result["actions_used"] == budget and result["trace"]:
        assert result["status"] in {"BUDGET_EXHAUSTED", "NO_ELIGIBLE_ACTION"}
    replay(result)


def test_refinement_preserves_physical_contract_and_single_global_budget():
    p = problem_with(F(1, 9), 15)
    out = run(p, p.digest())
    replay(out)
    refines = [row for row in out["trace"] if row["action"] == "refine"]
    assert len(refines) == 1
    parent, child = (out["problems"][refines[0][key]["problem_digest"]] for key in ("before", "after"))
    assert parent["epsilon"] == child["epsilon"] == str(p.epsilon)
    assert parent["diffusion"] == child["diffusion"]
    assert parent["max_actions"] == child["max_actions"] == p.max_actions
    assert len(child["loads"]) == 4 * len(parent["loads"])
    assert set(child["loads"]) == set(parent["loads"])


def test_decisions_are_deterministic_and_acceptance_is_replayable():
    p = problem_with(F(1, 4), 25)
    results = [run(p, p.digest()) for _ in range(2)]
    for r in results:
        replay(r)
        assert r["status"] == "ACCEPTED"
    left, right = results
    for a, b in zip(left["trace"], right["trace"], strict=True):
        assert {k: v for k, v in a.items() if k != "cost_ns"} == {k: v for k, v in b.items() if k != "cost_ns"}


def test_problem_mutation_after_digest_retention_fails_before_field_generation():
    original = problem_with(F(1, 100))
    relaxed = replace(original, epsilon=100)
    out = run(relaxed, original.digest())
    assert out["status"] == "INVALID_INPUT" and not out["accepted"]
    assert out["trace"][0]["after"] is None
    replay(out)


def test_projected_affine_stops_without_applying_exact_floor_formula():
    p = problem_with(F(1, 100))
    mesh = p.mesh
    means = [sum(mesh.vertices[i][0] for i in tri) / 3 for tri in mesh.triangles]
    p = replace(p, loads=means, original_affine=(0, 1, 0))
    out = run(p, p.digest())
    assert out["status"] == "UNDETERMINED" and not out["accepted"]
    assert out["trace"][-1]["observation"]["floor"] is None
    replay(out)


def test_saved_decisions_replay_from_independently_recreated_problems():
    from pathlib import Path
    from research.cfac.controller_demo import campaign, deterministic

    path = Path(__file__).resolve().parents[1] / "evidence/cfac/controller.json"
    assert deterministic(campaign()) == deterministic(json.loads(path.read_bytes()))
