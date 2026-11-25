"""Measure and replay algebraic stopping before RT1 on a futile refined P1/RT0 pair."""
from collections import Counter
from pathlib import Path

from . import stopping_study as study
from .stopping_analysis import summarize


PROTOCOL = study.StudyProtocol(
    Path(__file__).with_name("futile_design.json"),
    "dc9dcf98a5bff71c82e0c96d4d5f0a8f3e5032010458a08b745f35aaab7352c1",
    "5d7d920ace7e6497bf8792e5cb279db108838d62",
    "cfac-futile-measurements-v1",
    "research.cfac.futile_study",
)
CONTRASTS = (("cfac_retry", "cfac_reserve"), ("cfac_retry", "classical"),
             ("cfac_reserve", "classical"))


def gate_outcomes(rows, journals):
    """Charge the eligible probe separately from all subsequent checks, sweeps and RT1 work."""
    counts, statuses = Counter(), Counter()
    action_counts, action_times = Counter(), Counter()
    probe_ns = 0
    for row in rows:
        if row["journal_id"] is None:
            counts["unavailable_journal"] += 1
            continue
        events = journals[row["journal_id"]]["trace"]
        eligible = next((i for i, e in enumerate(events)
                         if "futile_gate" in (e["observation"] or {})), None)
        if eligible is None:
            counts["no_opportunity"] += 1
            continue
        entry = events[eligible]["observation"]["futile_gate"]
        counts["eligible"] += 1
        counts["entered" if entry["entered"] else "budget_bypass"] += 1
        statuses[row["status"]] += 1
        probe_ns += row["step_ns"][eligible]
        rt1_inspected = False
        for i in range(eligible + 1, len(events)):
            event = events[i]
            action = event["action"]
            decision = event["next_decision"]
            obs = event["observation"]
            action_counts[action] += 1
            action_times[action] += row["step_ns"][i]
            if action == "futile_stopping_check" and obs and "stop" in obs:
                if obs["stop"]:
                    counts["gate_satisfied"] += 1
                elif obs["budget_limited"]:
                    counts["gate_budget_limited"] += 1
            if action == "rt1_reconstruct" and decision != "rt1_inspect":
                counts["rt1_construction_failed"] += 1
            if action == "rt1_inspect":
                rt1_inspected = True
                if decision == "INVALID_INPUT":
                    counts["rt1_invalid"] += 1
                else:
                    counts["rt1_accepted" if decision == "ACCEPTED" else "rt1_checked_miss"] += 1
        if not rt1_inspected:
            counts["no_rt1_inspection"] += 1
    return {"counts": dict(counts), "eligible_terminal_statuses": dict(statuses),
            "eligible_probe_total_ns": probe_ns, "charged_action_counts": dict(action_counts),
            "charged_action_total_ns": dict(action_times)}


def summarize_receipt(receipt, digest):
    spec = study.design(PROTOCOL)
    cases = summarize(receipt["trials"], spec, comparisons=CONTRASTS)
    for case_id, case in cases.items():
        rows = [r for r in receipt["trials"] if r["case"] == case_id and r["policy"] == "cfac_retry"]
        case["futile_gate"] = gate_outcomes(rows, receipt["journals"])
    return {"receipt_sha256": digest, "design_sha256": PROTOCOL.design_sha, "cases": cases}


def main():
    study.main(PROTOCOL, summarize_receipt=summarize_receipt)


if __name__ == "__main__":
    main()
