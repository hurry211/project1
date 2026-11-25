"""A quick failure must never be counted as a fast verified solution."""
import pytest

from research.cfac.stopping_analysis import summarize


def example():
    policies = ["fixed", "classical", "cfac", "cfac_rt0", "no_coarse", "no_witness", "no_floor"]
    design = {"cases": [{"id": "a"}], "blocks": 3, "policies": policies}
    rows = []
    for policy in policies:
        for block, elapsed in enumerate((10, 20, 100)):
            accepted = not (policy == "cfac" and block == 1)
            if policy == "cfac":
                elapsed = (20, 1, 50)[block]
            rows.append({"case": "a", "policy": policy, "block": block, "elapsed_ns": elapsed,
                         "accepted": accepted, "status": "ACCEPTED" if accepted else "BUDGET_EXHAUSTED",
                         "controller_ns": 1, "peak_rss_kib": 1024, "action_counts": {"inspect": 1},
                         "action_ns": {"inspect": 1}})
    return design, rows


def test_pairs_are_block_matched_and_failures_are_charged():
    design, rows = example()
    case = summarize(rows, design)["a"]
    r = case["policies"]["cfac"]
    assert (r["attempts"], r["accepted"], r["unsuccessful_total_ns"], r["submitted_total_ns"]) == (3, 2, 1, 71)
    contrast = case["paired"]["cfac/classical"]
    assert contrast["paired_accepted"] == 2
    assert [v["block"] for v in contrast["ratios"]] == [0, 2]
    assert contrast["distribution"] == {"min": 0.5, "median": 1.25, "max": 2}


def test_duplicate_or_missing_rows_fail_closed():
    design, rows = example()
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows + rows[:1], design)
    with pytest.raises(ValueError, match="incomplete"):
        summarize(rows[1:], design)
