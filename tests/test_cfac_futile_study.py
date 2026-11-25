"""Audit the fixed trial design, the extra smoothing costs and the returned certificates."""
from collections import Counter
from copy import deepcopy
import sys

import pytest

from research.cfac import futile_study as futile
from research.cfac import reserve_study as reserve
from research.cfac import stopping_study as study


def test_frozen_cases_and_balanced_pairing():
    spec = study.design(futile.PROTOCOL)
    schedule = list(study.schedule(spec))
    assert len(schedule) == len({(c["id"], block, p) for c, block, _, p in schedule}) == 360
    assert set(Counter((c["id"], position, p) for c, _, position, p in schedule).values()) == {2}
    cases = {c["id"]: study.problem_for(c).digest() for c in spec["cases"]}
    prior = study.design(reserve.PROTOCOL)
    assert all(cases[c["id"]] == study.problem_for(c).digest() for c in prior["cases"])
    assert len(set(cases.values())) == len(cases) == 20
    assert spec["classical_fraction"] == prior["classical_fraction"] == "1/2"
    assert set(futile.CONTRASTS) == {("cfac_retry", "cfac_reserve"), ("cfac_retry", "classical"),
                                    ("cfac_reserve", "classical")}


def test_gate_summary_accounts_for_checks_sweeps_and_unavailable_returns():
    def event(action, decision, observation=None):
        return {"action": action, "next_decision": decision, "observation": observation}

    gate = event("witness_probe", "futile_stopping_check", {"futile_gate": {"entered": True}})
    bypass = event("witness_probe", "rt1_reconstruct", {"futile_gate": {"entered": False}})
    repeat = event("futile_stopping_check", "primal_sweep", {"stop": False, "budget_limited": False})
    limited = event("futile_stopping_check", "rt1_reconstruct", {"stop": False, "budget_limited": True})
    stopped = event("futile_stopping_check", "rt1_reconstruct", {"stop": True, "budget_limited": False})
    build = event("rt1_reconstruct", "rt1_inspect")
    traces = {
        "smooth": ("ACCEPTED", [gate, repeat, event("primal_sweep", "futile_stopping_check"),
                                stopped, build, event("rt1_inspect", "ACCEPTED")]),
        "fallback": ("NO_ELIGIBLE_ACTION", [gate, limited, build, event("rt1_inspect", "NO_ELIGIBLE_ACTION")]),
        "partial": ("BUDGET_EXHAUSTED", [bypass, build]),
        "invalid": ("INVALID_INPUT", [gate, stopped, build, event("rt1_inspect", "INVALID_INPUT")]),
        "failed": ("NO_ELIGIBLE_ACTION", [gate, stopped, event("rt1_reconstruct", "NO_ELIGIBLE_ACTION")]),
        "early": ("ACCEPTED", [event("inspect", "ACCEPTED")]),
    }
    journals = {k: {"trace": events} for k, (_, events) in traces.items()}
    rows = [{"journal_id": k, "step_ns": [10] + [5] * (len(events) - 1), "status": status}
            for k, (status, events) in traces.items()]
    rows.append({"journal_id": None, "step_ns": [], "status": "TIMEOUT"})
    result = futile.gate_outcomes(rows, journals)
    assert result["counts"] == {"eligible": 5, "entered": 4, "budget_bypass": 1,
                                "gate_satisfied": 3, "gate_budget_limited": 1, "rt1_accepted": 1,
                                "rt1_checked_miss": 1, "no_rt1_inspection": 2, "rt1_invalid": 1,
                                "rt1_construction_failed": 1, "no_opportunity": 1, "unavailable_journal": 1}
    assert result["eligible_probe_total_ns"] == 50
    assert result["charged_action_counts"] == {"futile_stopping_check": 5, "primal_sweep": 1,
                                             "rt1_reconstruct": 5, "rt1_inspect": 3}
    assert result["charged_action_total_ns"] == {k: 5 * n for k, n in result["charged_action_counts"].items()}
    assert result["eligible_terminal_statuses"] == {"ACCEPTED": 1, "NO_ELIGIBLE_ACTION": 2,
                                                   "BUDGET_EXHAUSTED": 1, "INVALID_INPUT": 1}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux wait4 and pidfd measurement")
def test_checked_checkerboard_return_replays_and_charges_the_gate(tmp_path):
    protocol = futile.PROTOCOL
    case = next(c for c in study.design(protocol)["cases"] if c["id"] == "checker_tight")
    journals = {}
    row = study.collect(case, 0, 0, "cfac_retry", 30, journals, tmp_path, protocol)
    spec = {"cases": [case], "policies": ["cfac_retry"], "blocks": 1, "timeout_s": 30}
    receipt = {"schema": protocol.schema, "design_sha256": protocol.design_sha,
               "design_commit": protocol.design_commit, "revision": "a" * 40, "journals": journals, "trials": [row]}
    study._replay_trials(receipt, spec, protocol)
    assert row["accepted"] and row["delivery_bytes"] > 1000
    summary = futile.gate_outcomes([row], journals)
    assert summary["counts"] == {"eligible": 1, "entered": 1, "gate_satisfied": 1, "rt1_accepted": 1}
    assert summary["charged_action_counts"]["primal_sweep"] > 0
    with pytest.raises(ValueError, match="identity"):
        study._replay_trials(receipt, spec, reserve.PROTOCOL)
    forged = deepcopy(receipt)
    forged["trials"][0]["action_counts"]["futile_stopping_check"] -= 1
    with pytest.raises(ValueError, match="action totals"):
        study._replay_trials(forged, spec, protocol)
