"""Paired timings and nested bootstrap by PDE and then run block."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np
from .sparse_study import read_receipt


def analyze(path):
    receipt, digest = read_receipt(path)
    rows = receipt["rows"]
    policies = sorted({r["policy"] for r in rows})
    cohort = receipt["cohort"]
    candidate = "cfac_screened" if cohort == "screen" else "cfac_sparse"
    rng = np.random.default_rng(20261126)
    out = {"archive_sha256": digest, "cohort": cohort,
           "submitted": len(rows), "policies": {}, "contrasts": {}}
    for policy in policies:
        selected = [r for r in rows if r["policy"] == policy]
        out["policies"][policy] = {"accepted": sum(r["accepted"] for r in selected), "submitted": len(selected),
                                   "median_submission_seconds": statistics.median(r["seconds"] for r in selected),
                                   "total_failure_seconds": sum(r["seconds"] for r in selected if not r["accepted"]),
                                   "max_peak_rss_kib": max((r.get("peak_rss_kib", 0) for r in selected), default=0),
                                   "certified_futile_levels_skipped": sum(l.get("decision") == "refine_after_primal_floor"
                                                                        for r in selected for l in r.get("work", {}).get("levels", [])),
                                   "median_checker_calls": statistics.median(r["work"]["certificate_checks"] for r in selected)
                                   if all("work" in r for r in selected) else None}
    indexed = {(r["case"], r["block"], r["policy"]): r for r in rows}
    for ref in ("classical", "classical_sparse", "fixed_sparse", "cfac_sparse"):
        if ref == candidate or ref not in policies:
            continue
        by_case = defaultdict(list)
        for trial in rows:
            if trial["policy"] == candidate:
                baseline = indexed[trial["case"], trial["block"], ref]
                if trial["accepted"] and baseline["accepted"]:
                    by_case[trial["case"]].append(trial["seconds"]/baseline["seconds"])
        strata = list(by_case.values())
        if not strata:
            continue
        bootstrap = []
        for _ in range(4096):
            sample = [value for i in rng.integers(len(strata), size=len(strata))
                      for value in rng.choice(strata[i], size=len(strata[i]), replace=True)]
            bootstrap.append(statistics.median(sample))
        eligible = [x for c in strata for x in c]
        out["contrasts"][ref] = {"eligible_pairs": len(eligible), "cases": len(strata),
                                  "ratio_of_paired_times_median": statistics.median(eligible),
                                  "nested_percentile_95pct": list(np.percentile(bootstrap, [2.5, 97.5])),
                                  "case_ratios": {case: statistics.median(samples) for case, samples in by_case.items()}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(analyze(args.receipt), stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
