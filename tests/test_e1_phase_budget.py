import json
import math
import os

import pytest

from benchmarks import e1
from conftest import requires_mfem
from sciagent.evidence import sha256_file
from sciagent.mfem_runner import solver_binary


def clocks():
    return ({"submitted": 7, "before_spawn": 8, "completed": 17},
            {"worker_entry": 9, "before_capture": 10, "after_capture": 14,
             "before_receipt_write": 15, "after_receipt_write": 16},
            {"elapsed_seconds": 3.9, "phase_seconds": {"native.process": 2, "verify.study": 1.9}})


def test_cross_process_wall_reconciliation_keeps_independent_capture_gap():
    parent, worker, profile = clocks()
    worker["completed"] = 700  # Only the parent timestamps the workflow endpoint.
    measured = e1.external_phases(parent, worker, profile)
    assert measured["status"] == "MEASURED"
    assert measured["accounting_gap_seconds"] == pytest.approx(0.1)
    assert measured["phase_seconds"]["study_capture"] == 3.9
    assert measured["capture_window_seconds"] == 4


@pytest.mark.parametrize("damage,code", [
    ("missing", "TIMING_UNMEASURED"), ("negative", "TIMING_UNRECONCILED"),
    ("nan", "TIMING_UNRECONCILED"), ("overlap", "TIMING_UNRECONCILED"),
    ("under_accounted", "TIMING_UNRECONCILED"), ("swapped", "TIMING_UNRECONCILED")])
def test_incomplete_or_inconsistent_clocks_fail_measurement(damage, code):
    parent, worker, profile = clocks()
    if damage == "missing":
        del worker["after_receipt_write"]
    elif damage == "negative":
        profile["phase_seconds"]["native.process"] = -1
    elif damage == "nan":
        worker["worker_entry"] = math.nan
    elif damage == "overlap":
        profile.update(elapsed_seconds=4.1, phase_seconds={"combined": 4.1})
    elif damage == "under_accounted":
        profile.update(elapsed_seconds=3, phase_seconds={"combined": 3})
    elif damage == "swapped":
        worker["before_receipt_write"] = 11
    accounting = e1.external_phases(parent, worker, profile)
    assert accounting["status"] == code
    json.dumps(accounting, allow_nan=False)


@pytest.mark.mfem
@requires_mfem
def test_real_worker_accounts_for_full_verified_endpoint(tmp_path):
    w = e1.workload_definitions("compact")["C1"]
    row = e1.run_trial(tmp_path / "trial", w, "gs", cpu=min(os.sched_getaffinity(0)), timeout=120,
                       binary_sha256=sha256_file(solver_binary()))
    assert row["eligible"] and row["verifier_verdict"] == "ACCEPTED"
    accounting = row["external_phases"]
    assert accounting["status"] == "MEASURED" and abs(accounting["accounting_gap_fraction"]) <= 0.02
    assert all(v > 0 for v in accounting["phase_seconds"].values())
    assert accounting["phase_seconds"]["study_capture"] == pytest.approx(row["profile"]["elapsed_seconds"])
    times = accounting["monotonic_boundaries_seconds"]
    assert row["elapsed_seconds"] == times["completed"] - times["submitted"]
