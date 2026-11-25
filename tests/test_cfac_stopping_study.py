"""Audit the matched design, timed return path and adversarial journal mutations."""
from collections import Counter
from copy import deepcopy
import subprocess
import sys

import pytest

from cfac.baselines import run_policy
from research.cfac import stopping_study as study


def test_frozen_design_balances_positions_at_each_case():
    spec = study.design()
    rows = list(study.schedule(spec))
    assert len(rows) == 392
    assert len({(c["id"], b, p) for c, b, _, p in rows}) == len(rows)
    positions = Counter((c["id"], pos, p) for c, _, pos, p in rows)
    assert set(positions.values()) == {1}
    assert spec["fixed_relative"] == "1/1000000" and spec["classical_fraction"] == "1/2"


def test_revision_binding_checks_worktree_index_and_untracked_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "0")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    source = tmp_path / "cfac"
    source.mkdir()
    module = source / "model.py"
    module.write_text("a=1\n")
    git("add", "cfac")
    git("-c", "user.name=fixture", "-c", "user.email=fixture@localhost",
        "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    head = git("rev-parse", "HEAD")
    study.verify_checkout(head, tmp_path)
    with pytest.raises(ValueError, match="clean Git checkout"):
        study.verify_checkout("0"*40, tmp_path)
    extra = source / "extra.py"
    extra.write_text("new\n")
    with pytest.raises(ValueError, match="extra.py"):
        study.verify_checkout(head, tmp_path)
    extra.unlink()
    module.write_text("a=2\n")
    with pytest.raises(ValueError, match="model.py"):
        study.verify_checkout(head, tmp_path)
    git("add", "cfac")
    with pytest.raises(ValueError, match="model.py"):
        study.verify_checkout(head, tmp_path)


def sample(case):
    p = study.problem_for(case)
    journal = run_policy(p, p.digest(), policy="classical")
    del journal["elapsed_ns"]
    for e in journal["trace"]:
        del e["cost_ns"]
    return journal


def test_refrozen_target_and_stale_final_state_fail_independent_checks():
    case = study.design()["cases"][-2]
    journal = sample(case)
    study.validate_journal(case, journal)
    replacement = dict(case, epsilon="10")
    with pytest.raises(ValueError, match="initial problem"):
        study.validate_journal(replacement, journal)
    tampered = deepcopy(journal)
    tampered["trace"][-1]["observation"]["state_digest"] = "0"*64
    with pytest.raises(ValueError, match="certificate"):
        study.validate_journal(case, tampered)


def test_uninspected_budget_outcome_cannot_be_credited():
    case = study.design()["cases"][-1]
    journal = sample(case)
    assert journal["status"] == "BUDGET_EXHAUSTED"
    journal["accepted"], journal["status"] = True, "ACCEPTED"
    journal["trace"][-1]["next_decision"] = "ACCEPTED"
    with pytest.raises(ValueError, match="accepted final field"):
        study.validate_journal(case, journal)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux wait4 and pidfd measurement")
def test_real_worker_delivers_proof_and_wrong_policy_label_fails(tmp_path):
    case = study.design()["cases"][-2]
    spec = {"cases": [case], "policies": ["classical"], "blocks": 1, "timeout_s": 30}
    journals = {}
    row = study.collect(case, 0, 0, "classical", 30, journals, tmp_path)
    receipt = {"schema": study.SCHEMA, "design_sha256": study.DESIGN_SHA,
               "design_commit": study.DESIGN_COMMIT, "revision": "a"*40,
               "journals": journals, "trials": [row]}
    study._replay_trials(receipt, spec)
    assert row["accepted"] and row["delivery_bytes"] > 1000
    assert row["controller_ns"] < row["elapsed_ns"]
    assert row["peak_rss_kib"] > 0 and row["decision_peak_rss_kib"] > 0
    row["policy"] = "fixed"
    with pytest.raises(ValueError, match="policy replay"):
        study._replay_trials(receipt, dict(spec, policies=["fixed"]))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux wait4 and pidfd measurement")
def test_real_deadline_and_worker_error_are_recorded(tmp_path):
    case = study.design()["cases"][0]
    timeout = study.collect(case, 0, 0, "cfac", 0.00001, {}, tmp_path)
    assert timeout["status"] == "TIMEOUT"
    assert timeout["elapsed_ns"] >= 10_000 and timeout["peak_rss_kib"] >= 0
    error = study.collect(case, 0, 0, "unregistered", 30, {}, tmp_path)
    assert error["status"] == "WORKER_ERROR"
    assert error["returncode"] and "unknown policy" in error["error"]
    assert error["elapsed_ns"] > 0 and error["peak_rss_kib"] > 0
    for row in (error, timeout):
        assert not row["accepted"] and row["journal_id"] is None
