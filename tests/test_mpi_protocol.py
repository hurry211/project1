"""Regression checks for the acceptance boundary around MPI benchmarking."""

import json

import pytest

from sciagent import mpi_study
from sciagent.__main__ import main
from sciagent.evidence import EvidenceStore, sha256_file


def test_matrix_names_the_only_reviewed_mpi_problem(capsys):
    assert main(["matrix", "--backend", "mpi_cpu"]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["status"] == "tested" and row["ranks"] == [1, 2, 4]
    assert row["element"] == "P1 triangle" and row["entrypoint"] == "mpi-study"
    assert main(["matrix", "--backend", "mpi_cpu", "--mesh-source", "gmsh22"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "not_verified"


def test_missing_build_preserves_failure_and_cannot_benchmark(tmp_path, monkeypatch):
    directory = tmp_path / "study"
    monkeypatch.setenv("POISSON_MPI_BIN", str(tmp_path / "missing"))
    result = mpi_study.run(directory)
    assert result["state"] == "FAILED"
    assert (directory / "manifest.json").is_file() and not (directory / "timing").exists()
    with pytest.raises(mpi_study.ExperimentError, match="rejected"):
        mpi_study.replay(directory, result["manifest_sha256"])
    with pytest.raises(FileExistsError):
        mpi_study.run(directory)


def test_unverified_screen_prevents_every_timed_sample(tmp_path, monkeypatch):
    from verifier import mpi

    calls = []

    def sample(root, label, ranks, n, serial, build):
        calls.append(label)
        return {"n": n}, {}

    def orders(levels):
        if calls[-1] == "screen/r4_n64":
            raise ValueError("convergence gate failed")
        return {}

    monkeypatch.setattr(mpi_study, "_checked_sample", sample)
    monkeypatch.setattr(mpi, "verify_mpi_orders", orders)
    with pytest.raises(ValueError, match="convergence"):
        mpi_study._experiment(tmp_path, tmp_path, 3, {})
    assert len(calls) == 12 and all(c.startswith("screen/") for c in calls)
    assert not (tmp_path / "results.json").exists()


def test_failed_timing_verification_invalidates_the_campaign(tmp_path, monkeypatch):
    from verifier import mpi

    def sample(root, label, ranks, n, serial, build):
        if label == "timing/repeat_03/r1_n32":
            raise ValueError("parallel field mismatch")
        return {"n": n}, {"workflow_seconds": 2., "launch_seconds": 1., "solve_seconds_max": .01}

    monkeypatch.setattr(mpi_study, "_checked_sample", sample)
    monkeypatch.setattr(mpi, "verify_mpi_orders", lambda levels: {})
    with pytest.raises(ValueError, match="field mismatch"):
        mpi_study._experiment(tmp_path, tmp_path, 3, {})


def test_mpi_archive_checks_files_before_reading_numerical_results(tmp_path):
    directory = tmp_path / "study"
    store = EvidenceStore(directory)
    store.write_json_once("plan.json", mpi_study.plan(3))
    store.finish_manifest({"kind": mpi_study.KIND, "state": "ACCEPTED"})
    digest = sha256_file(directory / "manifest.json")
    (directory / "plan.json").write_text("{}")
    with pytest.raises(mpi_study.ExperimentError, match="changed"):
        mpi_study.replay(directory, digest)
    with pytest.raises(ValueError, match="SHA-256"):
        mpi_study.replay(directory, "0" * 64)


def test_replay_checks_launcher_configuration_before_numerical_work(tmp_path):
    build = {"launcher": "/usr/bin/mpiexec", "binary": "/build/poisson_mpi"}
    (tmp_path / "mpi_environment.json").write_text(json.dumps(build))
    label = "screen/r2_n8"
    sample = tmp_path / label
    sample.mkdir(parents=True)
    record = {"exit_code": 0, "timed_out": False,
              "argv": mpi_study._mpi_argv(build, 1, 8, "/previous/location/" + label)}
    (sample / "process.json").write_text(json.dumps(record))
    with pytest.raises(mpi_study.ExperimentError, match="launcher arguments"):
        mpi_study._checked_sample(tmp_path, label, 2, 8, tmp_path, None)
