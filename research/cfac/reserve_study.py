"""Measure and replay the preregistered one-shot progress reserve on matched problems."""
from collections import Counter
from pathlib import Path

from . import stopping_study as study


PROTOCOL = study.StudyProtocol(
    Path(__file__).with_name("reserve_design.json"),
    "da54e3ec3167470d3bf1daccd3ae90d016fb2a8cb9f1060c1a12cae862ccc498",
    "128acdc2e2c644826e86272e0d53521cd8db81aa",
    "cfac-reserve-measurements-v1",
    "research.cfac.reserve_study",
)
CONTRASTS = (("cfac_reserve", "cfac"), ("cfac_reserve", "classical"), ("cfac", "classical"))


def reserve_outcomes(rows, journals):
    """Aggregate paid actions following the trigger, ending at the check, restore or failed build."""
    counts, post_miss, post_failure = Counter(), Counter(), Counter()
    actions, costs = Counter(), Counter()
    trigger_probe_ns = 0
    for row in rows:
        if row["journal_id"] is None:
            counts["unavailable_journal"] += 1
            continue
        trace = journals[row["journal_id"]]["trace"]
        start = next((i for i, e in enumerate(trace) if
                      (e["observation"] or {}).get("progress_reserve", {}).get("triggered")), None)
        if start is None:
            counts["no_trigger"] += 1
            continue
        counts["triggered"] += 1
        trigger_probe_ns += row["step_ns"][start]
        for i in range(start + 1, len(trace)):
            event = trace[i]
            action = event["action"]
            decision = event["next_decision"]
            actions[action] += 1
            costs[action] += row["step_ns"][i]
            if decision == "INVALID_INPUT":
                counts["invalid_input"] += 1
                break
            if decision == "ACCEPTED":
                counts["rt1_accepted" if action == "rt1_inspect" else "rt0_accepted"] += 1
                break
            if action == "rt1_reconstruct" and decision != "rt1_inspect":
                counts["construction_failed"] += 1
                post_failure[row["status"]] += 1
                break
            if action == "rt1_inspect":
                counts["certified_miss"] += 1
                post_miss[row["status"]] += 1
            if action == "restore_rt0":
                counts["restored"] += 1
                break
    return {"counts": dict(counts), "post_miss_statuses": dict(post_miss),
            "post_construction_failure_statuses": dict(post_failure), "trigger_probe_ns": trigger_probe_ns,
            "charged_action_counts": dict(actions), "charged_action_total_ns": dict(costs)}


def summarize_receipt(receipt, digest):
    from .stopping_analysis import summarize
    spec = study.design(PROTOCOL)
    cases = summarize(receipt["trials"], spec, comparisons=CONTRASTS)
    for name, result in cases.items():
        rows = [t for t in receipt["trials"] if t["case"] == name and t["policy"] == "cfac_reserve"]
        result["reserve"] = reserve_outcomes(rows, receipt["journals"])
    return {"receipt_sha256": digest, "design_sha256": PROTOCOL.design_sha, "cases": cases}


def main():
    study.main(PROTOCOL, summarize_receipt=summarize_receipt)


if __name__ == "__main__":
    main()
