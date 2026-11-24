import argparse
import os
from pathlib import Path

import pytest

from benchmarks.e1 import ARMS, block_schedule, campaign, make_request, qualification, run_trial, workload_definitions
from benchmarks.e1_audit import audit
from sciagent.evidence import sha256_file
from sciagent.mfem_runner import solver_binary
from sciagent.spec import sha256_of


def test_problem_and_validation_match_between_arms():
    for scale in ("compact", "memo"):
        for workload in workload_definitions(scale).values():
            req = make_request(workload, "gs")
            other = make_request(workload, "jacobi")
            duplicate = make_request(workload, "gs_repeat")
            assert sha256_of(req) == sha256_of(duplicate)
            assert req.problem == other.problem and req.policy == other.policy
            assert req.validation == other.validation
            assert req.validation.require_reference_solve
            assert req.execution.estimated_dofs(workload["levels"][-1]) <= req.validation.reference_solve_max_dofs
            assert req.execution.model_dump(exclude={"solver"}) == other.execution.model_dump(exclude={"solver"})
            assert req.execution.solver != other.execution.solver


def test_paired_sequence_reproducible_and_balanced():
    sequence = list(block_schedule(["C1","H2"], 5, 2, 91))
    assert sequence == list(block_schedule(["C1","H2"], 5, 2, 91))
    assert sequence != list(block_schedule(["C1","H2"], 5, 2, 90))
    for name in ("C1", "H2"):
        for block in range(-2, 5):
            assert sorted(a for n, b, a in sequence if (n, b) == (name, block)) == sorted(ARMS)


@pytest.mark.parametrize("changed", ["state", "verdict", "binary", "frozen", "request", "h1", "l2", "cap", "levels"])
def test_ineligible_runs_never_score(changed):
    w = {"levels": [4,8,16], "h1_target": 1., "l2_target": 1.}
    m = {"final_state": "ACCEPTED", "environment": {"solver_binary_sha256": "x"}, "frozen_sha256": {"v":"y"},
         "attempts": [{}]}
    v = {"verdict": "ACCEPTED", "failure_codes": [], "levels": [{"n": n, "l2_error": .1, "h1_semi_error": .2}
                                                                  for n in w["levels"]]}
    kwargs = {"workload": w, "request_matches": True, "binary_sha256": "x", "frozen": {"v":"y"}}
    assert qualification(m, v, **kwargs) == (True, "OK")
    if changed == "state":
        m.update(final_state="BUDGET_EXHAUSTED", final_code="TIMEOUT")
    elif changed == "verdict":
        v["verdict"] = "REJECTED"
    elif changed == "binary":
        kwargs["binary_sha256"] = "z"
    elif changed == "frozen":
        kwargs["frozen"] = {"v":"z"}
    elif changed == "request":
        kwargs["request_matches"] = False
    elif changed == "levels":
        v["levels"].pop()
    elif changed == "cap":
        v["failure_codes"] = ["REFERENCE_SKIPPED_TOO_LARGE"]
    else:
        v["levels"][-1]["h1_semi_error" if changed == "h1" else "l2_error"] = float("inf")
    assert not qualification(m, v, **kwargs)[0]


def test_deadline_and_storage_limits_fail_closed(tmp_path):
    w = workload_definitions("compact")["C1"]
    cpu = min(os.sched_getaffinity(0))
    for i, (timeout, storage, code) in enumerate([(1e-6, None, "TIMEOUT"), (10., 1, "STORAGE_LIMIT")]):
        row = run_trial(tmp_path / str(i), w, "gs", binary_sha256="0"*64, cpu=cpu,
                        timeout=timeout, storage_limit_bytes=storage)
        assert row["code"] == code and not row["eligible"] and row["elapsed_seconds"] > 0
    with pytest.raises(FileExistsError):
        run_trial(tmp_path / "0", w, "gs", binary_sha256="0"*64, cpu=cpu, timeout=1.)


@pytest.mark.mfem
@pytest.mark.skipif(not solver_binary().is_file(), reason="requires real serial MFEM")
def test_mixed_case_records_and_replays_without_binary(tmp_path, monkeypatch):
    directory = tmp_path / "campaign"
    args = argparse.Namespace(directory=directory, scale="compact", workloads=["H2"], cpu=None, seed=91,
                              study_timeout=180., wall_budget=300., min_free_gib=0.,
                              max_output_gib=1., warmups=0, repetitions=1, compact_archives=True)
    report = campaign(args)
    assert report["status"] == "COMPLETE"
    assert all(row["eligible"] and len(row["levels"]) == 3 for row in report["records"])
    for row in report["records"]:
        assert row["profile"]["worker_peak_rss_kib"] > 0 and row["elapsed_seconds"] >= row["profile"]["elapsed_seconds"]
    monkeypatch.setenv("POISSON_MMS_BIN", str(tmp_path / "absent"))
    relocated = tmp_path / "relocated"
    directory.rename(relocated)
    digest = sha256_file(relocated / "campaign.json")
    assert audit(relocated, digest, all_trials=True)["replayed"] == 3
    with pytest.raises(ValueError, match="campaign digest"):
        audit(relocated, "0"*64)
    row = report["records"][0]
    archive = relocated / row["directory"] / row["archive"]
    (archive / "frozen/validation.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        audit(relocated, digest)
