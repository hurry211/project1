"""Score challenge outcomes and quantify sampling uncertainty."""

from collections import Counter

from scipy.stats import binomtest


def metrics(trials):
    def interval(hits, total):
        ci = binomtest(hits, total).proportion_ci(.95, method="exact")
        return [float(ci.low), float(ci.high)]

    output = {}
    for cohort in ("panel", "iid"):
        groups = output[cohort] = {}
        for kind in ("numerical_fault", "numerical_clean", "archive_fault", "archive_clean"):
            rows = [r for r in trials if (r["cohort"], r["kind"]) == (cohort, kind)]
            fault = kind.endswith("fault")
            verdicts = Counter(r["verdict"] for r in rows)
            bad = verdicts["ACCEPTED"] if fault else len(rows) - verdicts["ACCEPTED"]
            group = {"n": len(rows), "verdicts": dict(sorted(verdicts.items())),
                     "false_accepts" if fault else "false_rejects": bad,
                     "by_family": {}, "by_configuration": {}}
            if cohort == "iid":
                group["error_rate_exact_95pct"] = interval(bad, len(rows))
            for field, key in (("family", "by_family"), ("case", "by_configuration")):
                for label in sorted({r[field] for r in rows}):
                    group[key][label] = dict(sorted(Counter(r["verdict"] for r in rows if r[field] == label).items()))
            if fault:
                group["union_detected"] = sum(r["verdict"] == "REJECTED" and bool(r["failed"]) for r in rows)
                if cohort == "iid":
                    group["union_rate_exact_95pct"] = interval(group["union_detected"], len(rows))
                group["gates"] = {}
                for gate in sorted({g for r in rows for g in r["evaluated"]}):
                    reached = sum(gate in r["evaluated"] for r in rows)
                    hits = sum(gate in r["failed"] for r in rows)
                    only = sum(r["failed"] == [gate] for r in rows)
                    group["gates"][gate] = {"evaluated": reached, "detected": hits, "sole_detection": only}
                    if cohort == "iid":
                        group["gates"][gate].update(rate_over_all_exact_95pct=interval(hits, len(rows)),
                                                    rate_when_evaluated_exact_95pct=interval(hits, reached))
            groups[kind] = group
    return output
