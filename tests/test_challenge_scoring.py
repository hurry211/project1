from benchmarks.scoring import metrics


def test_detection_union_and_evaluated_denominators():
    rows = []
    for cohort in ("panel", "iid"):
        for kind in ("numerical_fault", "archive_fault", "numerical_clean", "archive_clean"):
            for verdict, reached, failed in [("REJECTED", ["a", "b"], ["a", "b"]),
                                            ("REJECTED", ["a"], ["a"]),
                                            ("ACCEPTED", ["a", "b"], []), ("ERROR", [], [])]:
                rows.append(dict(cohort=cohort, kind=kind, verdict=verdict, evaluated=reached,
                                 failed=failed, case="one", family="challenge"))
    report = metrics(rows)
    group = report["panel"]["numerical_fault"]
    assert group["n"] == 4 and group["false_accepts"] == 1 and group["union_detected"] == 2
    assert group["gates"] == {"a": {"evaluated": 3, "detected": 2, "sole_detection": 1},
                              "b": {"evaluated": 2, "detected": 1, "sole_detection": 0}}
    assert group["verdicts"]["ERROR"] == 1
    assert report["panel"]["numerical_clean"]["false_rejects"] == 3
    iid = report["iid"]["numerical_fault"]
    assert iid["error_rate_exact_95pct"][0] < .25 < iid["error_rate_exact_95pct"][1]
    assert "error_rate_exact_95pct" not in group
