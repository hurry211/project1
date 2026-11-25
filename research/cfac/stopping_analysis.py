"""Summarize submitted times for all attempts and paired times for accepted solutions."""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


DESIGN = Path(__file__).with_name("stopping_design.json")
CONTRASTS = (("cfac", "classical"), ("classical", "fixed"), ("cfac", "cfac_rt0"),
             ("cfac", "no_coarse"), ("cfac", "no_witness"), ("cfac", "no_floor"))


def spread(values):
    if not values:
        return None
    return {"min": min(values), "median": median(values), "max": max(values)}


def summarize(trials, design):
    policies = design["policies"]
    expected_cases = {case["id"] for case in design["cases"]}
    grouped = {(case, policy): {} for case in expected_cases for policy in policies}
    for trial in trials:
        key = (trial["case"], trial["policy"])
        group = grouped[key]
        if trial["block"] in group:
            raise ValueError(f"duplicate trial in block for {key}")
        if type(trial["accepted"]) is not bool or type(trial["elapsed_ns"]) is not int or trial["elapsed_ns"] <= 0:
            raise ValueError("invalid acceptance or submitted time")
        group[trial["block"]] = trial
    case_results = {}
    for case in design["cases"]:
        name = case["id"]
        groups = {policy: grouped[name, policy] for policy in policies}
        blocks = sorted(groups[policies[0]])
        if len(blocks) != design["blocks"] or any(set(g) != set(blocks) for g in groups.values()):
            raise ValueError(f"incomplete paired design for {name}")
        outcomes = {}
        for policy, group in groups.items():
            rows = list(group.values())
            accepted = [r for r in rows if r["accepted"]]
            costs, counts = Counter(), Counter()
            for trial in rows:
                costs.update(trial.get("action_ns") or {})
                counts.update(trial.get("action_counts") or {})
            outcomes[policy] = {
                "statuses": dict(Counter(r["status"] for r in rows)), "accepted": len(accepted),
                "attempts": len(rows), "submitted_ms": spread([r["elapsed_ns"]/1e6 for r in rows]),
                "accepted_ms": spread([r["elapsed_ns"]/1e6 for r in accepted]),
                "controller_ms": spread([r["controller_ns"]/1e6 for r in rows if r["controller_ns"] is not None]),
                "decision_peak_rss_mib": spread([r["decision_peak_rss_kib"]/1024 for r in rows
                                                 if r.get("decision_peak_rss_kib") is not None]),
                "peak_rss_mib": spread([r["peak_rss_kib"]/1024 for r in rows if r["peak_rss_kib"] is not None]),
                "submitted_total_ns": sum(r["elapsed_ns"] for r in rows),
                "unsuccessful_total_ns": sum(r["elapsed_ns"] for r in rows if not r["accepted"]),
                "action_counts": dict(counts), "action_total_ns": dict(costs),
            }
        contrasts = {}
        for left, right in CONTRASTS:
            ratios = []
            for block in blocks:
                a, b = groups[left][block], groups[right][block]
                if a["accepted"] and b["accepted"]:
                    ratios.append({"block": block, "ratio": a["elapsed_ns"] / b["elapsed_ns"]})
            contrasts[f"{left}/{right}"] = {
                "paired_accepted": len(ratios), "ratios": ratios,
                "distribution": spread([r["ratio"] for r in ratios]),
            }
        case_results[name] = {"policies": outcomes, "paired": contrasts}
    return case_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="independently retained compressed receipt digest")
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    if sha256(raw).hexdigest() != args.sha256:
        parser.error("receipt digest mismatch")
    data = json.loads(gzip.decompress(raw))
    design_raw = DESIGN.read_bytes()
    if sha256(design_raw).hexdigest() != data["design_sha256"]:
        parser.error("design digest mismatch")
    cases = summarize(data["trials"], json.loads(design_raw))
    result = {"receipt_sha256": args.sha256, "design_sha256": data["design_sha256"], "cases": cases}
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
