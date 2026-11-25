"""Regression tests for matched archive timing, verdict parity and source isolation."""
import random
import sys

import pytest

from benchmarks import verification_cost as cost


def test_schedule_balances_first_position_and_repeats_by_seed():
    for n in (2, 3, 30, 31):
        order = cost._orders(n, random.Random(410))
        assert len(order) == n
        assert order == cost._orders(n, random.Random(410))
        assert all(set(pair) == set(cost.ARMS) for pair in order)
        assert abs(n - 2 * sum(pair[0] == "baseline" for pair in order)) <= 1


def test_paired_effect_uses_whole_matched_blocks():
    rows = [{"baseline": {"seconds": a}, "candidate": {"seconds": b}}
            for a, b in ((10, 9), (8, 7), (1, 100))]
    effect = cost._paired_effect(rows, "seconds")
    assert effect["median_candidate_minus_baseline"] == -1
    assert effect["paired_bootstrap_95pct"] == [-1, 99]


@pytest.fixture
def replay(tmp_path):
    if sys.platform != "linux":
        pytest.skip("fresh-worker RSS measurement uses Linux accounting")
    root = tmp_path / "source"
    package = root / "verifier"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    # A stand-alone source tree demonstrates that each worker uses the selected arm.
    (package / "verifier.py").write_text(
        "def verify_attempt(attempt, *, require_run_records):\n"
        "    assert require_run_records\n"
        "    return {'verdict': 'ACCEPTED', 'checks': [1]}\n"
    )
    attempt = tmp_path / "run/attempts/attempt_01"
    attempt.mkdir(parents=True)
    (attempt / "input.txt").write_text("field")
    return root, attempt


def test_fresh_worker_uses_selected_source_and_excludes_warmup(replay):
    source, attempt = replay
    result = cost.compare(source, source, [attempt], 2, 44, expect_verdict="ACCEPTED")
    r, = result["attempts"]
    assert [row["block"] for row in r["samples"]] == [0, 1]
    assert r["inventory_files"] == 1
    assert result["source_tree_sha256"]["baseline"] == result["source_tree_sha256"]["candidate"]
    for row in [r["warmup"], *r["samples"]]:
        assert row["baseline"]["verdict_sha256"] == row["candidate"]["verdict_sha256"]
        assert row["baseline"]["verdict"] == "ACCEPTED"
        assert row["candidate"]["process_peak_rss_kib"] > 0


def test_equal_final_state_does_not_hide_changed_check(replay, tmp_path):
    import shutil
    source, attempt = replay
    candidate = tmp_path / "candidate"
    shutil.copytree(source, candidate)
    path = candidate / "verifier/verifier.py"
    path.write_text(path.read_text().replace("[1]", "[2]"))
    with pytest.raises(ValueError, match="complete verdict differs"):
        cost.compare(source, candidate, [attempt], 2, 1)


@pytest.mark.parametrize("change,reason", [
    ("input", "archive changed"), ("verdict", "between blocks"), ("source", "source files changed")
])
def test_rejects_drift(replay, monkeypatch, change, reason):
    source, attempt = replay
    calls = 0

    def measure(attempt, roots, order, script):
        nonlocal calls
        calls += 1
        if calls == 2:
            if change == "input":
                (attempt / "input.txt").write_text("different")
            elif change == "source":
                (source / "verifier/verifier.py").write_text("changed")
        digest = "new" if calls >= 2 and change == "verdict" else "initial"
        data = {"seconds": 1, "process_peak_rss_kib": 5, "verdict_sha256": digest, "verdict": "ACCEPTED"}
        return {"order": order, "baseline": data, "candidate": data}

    monkeypatch.setattr(cost, "_measure_pair", measure)
    with pytest.raises(ValueError, match=reason):
        cost.compare(source, source, [attempt], 2, 1)
