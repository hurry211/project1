"""Guard the reserve study's provenance, paired design, action costs and restored fields."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import gzip
import json
import sys

import pytest

from research.cfac import reserve_study as reserve
from research.cfac import stopping_study as study


def test_reserve_design_rotates_positions_and_retains_historical_problems():
    spec = study.design(reserve.PROTOCOL)
    rows = list(study.schedule(spec))
    assert len(rows) == 252 and len({(c["id"], b, p) for c, b, _, p in rows}) == 252
    assert set(Counter((c["id"], pos, p) for c, _, pos, p in rows).values()) == {2}
    cases = {c["id"]: study.problem_for(c).digest() for c in spec["cases"]}
    for historical in study.design()["cases"]:
        assert cases[historical["id"]] == study.problem_for(historical).digest()
    assert len({p for pair in reserve.CONTRASTS for p in pair}) == 3


def test_legacy_replay_and_cross_protocol_rejection():
    path = Path(__file__).resolve().parents[1] / "evidence/cfac/stopping-study.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        receipt = json.load(f)
    with pytest.raises(ValueError, match="identity"):
        study.replay(receipt, reserve.PROTOCOL)
    # Reuse the original measured return, its independently evaluated certificate
    # and its exact deterministic policy. The whole historic run has 392 entries.
    assert len(receipt["trials"]) == 392
    case = study.design()["cases"][0]
    row = receipt["trials"][0]
    key = row["journal_id"]
    spec = {"cases": [case], "policies": [row["policy"]], "blocks": 1, "timeout_s": 30}
    study._replay_trials(dict(receipt, trials=[row], journals={key: receipt["journals"][key]}), spec)

    bad = deepcopy(receipt["journals"][key])
    enriched = next(e for e in bad["trace"] if e["action"] == "rt1_reconstruct")
    enriched["after"]["parent"]["primal"][0] = "1"
    with pytest.raises(ValueError, match="changed its parent"):
        study.validate_journal(case, bad)

    bad = deepcopy(receipt["journals"][key])
    bad["trace"][-1]["next_decision"] = "restore_rt0"
    final = bad["trace"][-1]["after"]
    bad["trace"].append({"step": len(bad["trace"]) + 1, "action": "restore_rt0",
                         "before": final, "after": final})
    bad["actions_used"] += 1
    with pytest.raises(ValueError, match="restoration differs"):
        study.validate_journal(case, bad)


def test_reserve_summary_charges_actions_through_misses_and_partial_restoration():
    def journal(actions):
        events = [{"action": "witness_probe", "observation": {"progress_reserve": {"triggered": True}},
                   "next_decision": "primal_sweep"}]
        events.extend({"action": action, "observation": None, "next_decision": decision}
                      for action, decision in actions)
        return {"trace": events}

    sweep = [("primal_sweep", "inspect")]
    construct = sweep + [("inspect", "rt1_reconstruct")]
    trial = construct + [("rt1_reconstruct", "rt1_inspect")]
    miss = trial + [("rt1_inspect", "restore_rt0")]
    journals = {"hit0": journal(sweep + [("inspect", "ACCEPTED")]),
                "hit1": journal(trial + [("rt1_inspect", "ACCEPTED")]),
                "failed": journal(construct + [("rt1_reconstruct", "inspect"), ("inspect", "ACCEPTED")]),
                "spent": journal(miss),
                "restored": journal(miss + [("restore_rt0", "inspect"), ("inspect", "ACCEPTED")])}
    rows = [{"status": "BUDGET_EXHAUSTED" if k == "spent" else "ACCEPTED", "journal_id": k,
             "step_ns": [3] * len(j["trace"])} for k, j in journals.items()]
    summary = reserve.reserve_outcomes(rows, journals)
    assert summary["counts"] == {"triggered": 5, "rt0_accepted": 1, "rt1_accepted": 1,
                                 "construction_failed": 1, "certified_miss": 2, "restored": 1}
    assert summary["post_miss_statuses"] == {"BUDGET_EXHAUSTED": 1, "ACCEPTED": 1}
    assert summary["post_construction_failure_statuses"] == {"ACCEPTED": 1}
    assert summary["trigger_probe_ns"] == 15
    assert summary["charged_action_total_ns"] == {"primal_sweep": 15, "inspect": 15,
                                                 "rt1_reconstruct": 12, "rt1_inspect": 9, "restore_rt0": 3}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux wait4 and pidfd measurement")
def test_reserve_worker_returns_checked_field_and_receipt_is_bound_to_schedule(tmp_path):
    protocol = reserve.PROTOCOL
    case = next(c for c in study.design(protocol)["cases"] if c["id"] == "null")
    spec = {"cases": [case], "policies": ["cfac_reserve"], "blocks": 1, "timeout_s": 30}
    journals = {}
    row = study.collect(case, 0, 0, "cfac_reserve", 30, journals, tmp_path, protocol)
    receipt = {"schema": protocol.schema, "design_sha256": protocol.design_sha,
               "design_commit": protocol.design_commit, "revision": "b"*40, "journals": journals, "trials": [row]}
    assert row["accepted"] and row["delivery_bytes"] > 1000
    study._replay_trials(receipt, spec, protocol)
    row["policy"] = "classical"
    with pytest.raises(ValueError, match="policy replay"):
        study._replay_trials(receipt, dict(spec, policies=["classical"]), protocol)
