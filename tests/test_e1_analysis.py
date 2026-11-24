from copy import deepcopy

import pytest

from benchmarks.e1_analysis import analyze, choose_solver, fit_policy


def samples(count=30, outcome=(1.0, 0.6, 1.0)):
    records = []
    for name, size in [("C1", 4), ("C2", 8), ("H1", 16), ("H2", 8)]:
        for block in range(count):
            for arm, elapsed in zip(("gs", "jacobi", "gs_repeat"), outcome, strict=True):
                levels = [{"n": n, "iterations": n, "native_wall_seconds": n**3 / 20000,
                           "solve_seconds": n**3 / (60000 if arm == "jacobi" else 30000)}
                          for n in (size, size*2, size*4)]
                records.append({"workload": name, "warmup": False, "block": block, "arm": arm,
                                "eligible": True, "state": "ACCEPTED", "elapsed_seconds": elapsed,
                                "levels": levels})
    return {"records": records, "status": "COMPLETE",
            "workloads": {w: {"levels": [n, 2*n, 4*n]}
                          for w, n in (("C1", 4), ("C2", 8), ("H1", 16), ("H2", 8))}}


def test_paired_intervals_compare_to_practical_margin_and_aa_noise():
    result = analyze(samples())
    c = result["workloads"]["C1"]
    assert c["result"] == "jacobi_materially_faster"
    assert c["paired_bootstrap_95pct_seconds"] == pytest.approx([-0.4, -0.4])
    assert c["aa_bootstrap_95pct_seconds"] == [0, 0]
    assert c["jacobi_to_gs_ratio_of_medians"] == 0.6
    assert analyze(samples(outcome=(1, 0.99, 1)))["workloads"]["C1"]["result"] == "within_margin"
    assert analyze(samples(outcome=(1, 0.6, 1.2)))["workloads"]["C1"]["result"] == "noise_unresolved"
    assert analyze(samples(5))["workloads"]["C1"]["result"] == "insufficient_blocks"


def test_calibration_ignores_holdout_and_warmups_even_if_modified():
    campaign = samples()
    rows = campaign["records"]
    policy = fit_policy(rows)
    rows.append({**rows[0], "elapsed_seconds": 999999, "warmup": True})
    for r in rows:
        if r["workload"] in ("H1", "H2"):
            r.update(eligible=False, state="FAILED", elapsed_seconds=999999, levels=[])
    assert fit_policy(rows) == policy and policy["enabled"]


def test_failure_time_remains_charged_and_failed_blocks_cannot_promote():
    campaign = samples()
    r = next(r for r in campaign["records"] if (r["workload"], r["arm"]) == ("C1", "jacobi"))
    r.update(state="FAILED", eligible=False, levels=[], elapsed_seconds=12.3)
    report = analyze(campaign)
    score = report["workloads"]["C1"]
    assert score["qualified_complete_blocks"] == 29
    assert score["arms"]["jacobi"]["elapsed_seconds_failed_attempts"] == 12.3
    assert score["result"] == "failed_or_incomplete_blocks"
    assert not report["policy"]["enabled"]
    assert report["workloads"]["H1"]["pilot_selection"]["pilot_native_wall_seconds_charged"] == 0
    assert report["workloads"]["H1"]["pilot_selection"]["probe_both_native_prefixes_seconds_if_always_probed"] > 0


def test_failed_selected_default_is_charged_without_calling_it_a_winner():
    campaign = samples(2)
    row = next(r for r in campaign["records"] if r["workload"] == "H1" and r["arm"] == "gs")
    row.update(eligible=False, state="FAILED", levels=[], elapsed_seconds=14)
    d = analyze(campaign)["workloads"]["H1"]["pilot_selection"]
    assert d["policy_actions_scored"] == 2 and d["blocks_with_complete_observations"] == 1
    assert d["verified_selected"] == 1
    assert d["elapsed_seconds_including_selected_failures_and_charged_pilots"] == 15
    assert d["matched_oracle_comparisons"] == 1


def test_decision_charges_both_real_prefixes_and_refuses_unavailable_resources():
    rows = samples()["records"]
    policy = fit_policy(rows)
    pilots = {s: deepcopy(r["levels"][:2]) for s, r in zip(("cg_gs", "cg_jacobi"), rows)}
    opts = dict(target_n=64, authorized_solvers=list(pilots), hardware_ok=True)
    selected = choose_solver(policy, pilots, **opts)
    assert selected["solver"] == "cg_jacobi"
    assert selected["pilot_seconds"] == sum(l["native_wall_seconds"] for v in pilots.values() for l in v)
    blocked = choose_solver(policy, pilots, **{**opts, "hardware_ok": False})
    assert blocked["solver"] is None
    assert choose_solver(policy, pilots, **{**opts, "authorized_solvers": ["cg_gs"]})["solver"] == "cg_gs"
    assert choose_solver(policy, pilots, **opts, pilot_elapsed_seconds=20)["solver"] == "cg_gs"
    with pytest.raises(ValueError, match="include all"):
        choose_solver(policy, pilots, **opts, pilot_elapsed_seconds=0)
    pilots["cg_gs"][0]["iterations"] = 0
    assert choose_solver(policy, pilots, **opts)["reason"] == "unusable_pilot"


def test_late_holdout_timings_do_not_enter_prefix_decision():
    c = samples()
    before = analyze(c)["workloads"]["H1"]["pilot_selection"]
    for r in c["records"]:
        if r["workload"] == "H1":
            r["levels"][-1]["solve_seconds"] *= 10
            if r["arm"] == "gs":
                r["elapsed_seconds"] *= 100
    after = analyze(c)["workloads"]["H1"]["pilot_selection"]
    assert (before["selected_solvers"], before["forecast_solvers"]) == (after["selected_solvers"], after["forecast_solvers"])
    assert before["median_excess_seconds_vs_fixed_gs"] != after["median_excess_seconds_vs_fixed_gs"]


def test_invalid_pairs_fail_closed():
    c = samples(1)
    c["records"].append(c["records"][0])
    with pytest.raises(ValueError, match="duplicate"):
        analyze(c)
    c["records"].pop()
    c["records"][0].update(state="TIMEOUT")
    with pytest.raises(ValueError, match="accepted"):
        analyze(c)


def test_noise_sensitivity_vetoes_an_apparent_gain_without_rewriting_primary_result():
    c = samples(120)
    for r in c["records"]:
        if r["workload"] == "C1" and r["arm"] == "gs_repeat":
            r["elapsed_seconds"] += (-0.3, 0, 0.3)[r["block"] % 3]
    a = analyze(c)
    score = a["workloads"]["C1"]
    assert score["result"] == "jacobi_materially_faster"
    assert score["conservative_result"] == "absolute_aa_noise_unresolved"
    assert not a["policy"]["enabled"] and a["policy"]["reason"] == "unresolved_absolute_aa_noise"
    noise = score["absolute_aa_sensitivity"]
    assert noise["median_absolute_aa_seconds"] == pytest.approx(0.3)
    assert noise["p90_absolute_aa_seconds"] == pytest.approx(0.3)
    assert noise["assessed_blocks"] == 120 and noise["absolute_aa_above_margin_count"] == 80
    assert noise["absolute_aa_above_margin_fraction"] == pytest.approx(2/3)


def test_missing_counterpart_keeps_default_elapsed_and_excludes_oracle_comparison():
    c = samples(1)
    c["records"] = [r for r in c["records"] if not (r["workload"] == "H1" and r["arm"] == "jacobi")]
    score = analyze(c)["workloads"]["H1"]
    assert score["missing_arm_blocks"] == 1 and score["complete_but_ineligible_blocks"] == 0
    assert score["result"] == score["conservative_result"] == "failed_or_incomplete_blocks"
    assert score["absolute_aa_sensitivity"]["assessed_blocks"] == 0
    assert score["absolute_aa_sensitivity"]["absolute_aa_above_margin_fraction"] is None
    policy = score["pilot_selection"]
    assert policy["policy_actions_scored"] == 1 and policy["matched_oracle_comparisons"] == 0
    assert policy["blocks_with_complete_observations"] == 0
