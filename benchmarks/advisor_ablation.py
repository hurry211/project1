"""Attribute adaptive decisions to live model nominations in a pinned comparison."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from benchmarks.planning import pinned_json
from sciagent import adaptive_advisor, local_llm, planner


def audit(result):
    model, digest = result["design"]["model"], result["design"]["model_digest"]
    if not model or not digest:
        raise ValueError("a pinned live model is required")
    task_ids = result["held_out_task_ids"]
    arms = ("ai_assisted", "advisor_rules")
    rows = [r for r in result["trials"] if r["strategy"] in arms]
    indexed = {(r["task_id"], r["repeat"], r["strategy"]): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != {
            (task, rep, arm) for task in task_ids for rep in range(result["repeats"]) for arm in arms}:
        raise ValueError("incomplete or duplicated paired comparison")
    pairs = []
    statuses = Counter()
    usage = Counter()
    for task in task_ids:
        for rep in range(result["repeats"]):
            live, control = (indexed[task, rep, arm] for arm in arms)
            if live["target_l2"] != control["target_l2"] or control["meta"].get("used_live_llm"):
                raise ValueError("mismatched target or contaminated control")
            rec = live["meta"].get("advisor")
            live_used = live["meta"]["used_live_llm"]
            fallback = None
            if rec:
                payload = json.loads(rec["prompt"])
                preview = payload["candidates"]
                ids = [c["candidate_id"] for c in preview]
                if (rec["model"] != model or rec["model_digest"] != digest
                        or rec["system"] != adaptive_advisor.SYSTEM or not 1 <= len(set(ids)) == len(ids) <= 5
                        or payload["problem"]["target_l2"] != live["target_l2"]):
                    raise ValueError("inconsistent prompt or provider identity")
                fallback = ids[0]
                if rec["used_live_llm"] != live_used:
                    raise ValueError("inconsistent nomination flag")
                response = rec.get("response")
                if (rec["provider_completed"] != bool(response) or (live_used and not response)
                        or live_used != (rec["status"] == "nominated")):
                    raise ValueError("provider completion disagrees with evidence")
                if response:
                    if (response["provider"] != "ollama_local" or response["model"] != model
                            or response["model_digest"] != digest or response["options"] != local_llm.OPTIONS):
                        raise ValueError("unverified model response")
                    for k, v in response["usage"].items():
                        if v is not None:
                            usage[k] += v
                    try:
                        proposal = planner._strict_object(response["text"], 512)
                    except ValueError:
                        proposal = {}
                    nomination = (proposal.get("candidate_id") if response["done_reason"] == "stop"
                                  and set(proposal) == {"candidate_id"} else None)
                    valid = isinstance(nomination, str) and nomination in ids
                    if valid != live_used or (valid and live["selected_initial_candidate"] != nomination):
                        raise ValueError("selected candidate differs from live nomination")
                if not live_used and live["selected_initial_candidate"] != fallback:
                    raise ValueError("fallback choice differs from first feasible forecast")
                statuses[rec["status"]] += 1
            else:
                if live_used:
                    raise ValueError("missing live model response")
                statuses["no_call"] += 1
            paired = live["accepted"] and control["accepted"]
            pairs.append({"task_id": task, "repeat": rep, "matched_acceptance": paired,
                          "used_live_llm": live_used, "fallback_in_live_run": fallback,
                          "selected_live_arm": live["selected_initial_candidate"],
                          "selected_rules_arm": control["selected_initial_candidate"],
                          "changed_own_fallback": live_used and live["selected_initial_candidate"] != fallback,
                          "live_minus_control_seconds": (live["total_end_to_end_seconds"] -
                                                        control["total_end_to_end_seconds"]) if paired else None})
    deltas = [p["live_minus_control_seconds"] for p in pairs if p["matched_acceptance"]]
    return {"model": model, "model_digest": digest, "paired_trials": len(pairs), "paired_accepted": len(deltas),
            "provider_outcomes": dict(statuses), "tokens_reported": dict(usage),
            "changed_own_fallback": sum(p["changed_own_fallback"] for p in pairs),
            "median_matched_live_minus_rules_seconds": median(deltas) if deltas else None,
            "pairs": pairs}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summary = audit(pinned_json(args.report, args.sha256))
    with args.output.open("x") as f:
        json.dump({"comparison_sha256": args.sha256, **summary}, f, indent=2, allow_nan=False)
        f.write("\n")
