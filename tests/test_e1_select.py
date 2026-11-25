import json
import os
from copy import deepcopy
from argparse import Namespace

import pytest

from benchmarks.e1 import TIMED_SOURCES, campaign, make_request, source_fingerprints, workload_definitions
from benchmarks.e1_select import calibration, performance_provenance, run_selection
from sciagent.controller import reverify
from sciagent.environment import MFEM_FLAGS
from sciagent.evidence import sha256_file
from sciagent.mfem_runner import solver_binary
from sciagent.spec import sha256_of


@pytest.mark.parametrize("kwargs", [{"authorized_solvers":[]}, {"authorized_solvers":["cg_gs","cg_gs"]},
                                    {"authorized_solvers":["unsupported"]}, {"wall_budget":float("nan")},
                                    {"memory_mib":0}])
def test_invalid_selection_envelope(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run_selection(tmp_path / "out", tmp_path, "a"*64, "H2", **kwargs)
    assert not (tmp_path / "out").exists()


def test_selection_refuses_changed_pins_and_honors_hard_timeout(tmp_path):
    src = tmp_path / "inputs"
    src.mkdir()
    (src / "campaign.json").write_text('{"config_sha256":"fake"}')
    (src / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="campaign.json"):
        calibration(src, "wrong")
    with pytest.raises(ValueError, match="config.json"):
        calibration(src, sha256_file(src / "campaign.json"))
    row = run_selection(tmp_path / "timed", src, "invalid", "H2", wall_budget=0.001)
    assert row["code"] == "TIMEOUT" and row["eligible"] is False
    assert not (tmp_path / "timed" / "final").exists()


def test_performance_identity_checks_sources_runtime_and_build():
    sources = {k: "a" * 64 for k in TIMED_SOURCES}
    env = {"python": "3.12.3", "python_packages": {"numpy": "2.5", "scipy": "1.18", "pydantic": "2.14"},
           "mfem": {"git": "v4.10-0-gd964264", "version": "4.10.0", "library_sha256": "b" * 64,
                    "config": {flag: False for flag in MFEM_FLAGS}}}
    cfg = {"host": {"environment": deepcopy(env)}, "source_sha256": sources.copy()}
    assert performance_provenance(cfg, env, sources)["matched"]
    del cfg["source_sha256"]["verifier/fem.py"]
    cfg["source_sha256"]["sciagent/profiling.py"] = "c" * 64
    env["mfem"]["library_sha256"] = None
    env["mfem"]["config"]["MFEM_USE_MPI"] = True
    env["python_packages"]["scipy"] = "1.19"
    gate = performance_provenance(cfg, env, sources)
    assert not gate["matched"]
    assert {k: d["reason"] for k, d in gate["differences"].items()} == {
        "verifier/fem.py": "missing_calibration_value", "sciagent/profiling.py": "value_changed",
        "scipy": "value_changed", "mfem.library_sha256": "missing_runtime_value", "mfem.MFEM_USE_MPI": "value_changed"}
    cfg.pop("source_sha256")
    assert set(TIMED_SOURCES) <= performance_provenance(cfg, env, sources)["differences"].keys()


def test_source_pins_use_git_line_endings(tmp_path):
    f = tmp_path / "code.py"
    f.write_bytes(b"pass\n")
    a = source_fingerprints(["code.py"], tmp_path)
    f.write_bytes(b"pass\r\n")
    assert source_fingerprints(["code.py"], tmp_path) == a
    f.unlink()
    assert source_fingerprints(["code.py"], tmp_path)["code.py"] is None


@pytest.mark.mfem
@pytest.mark.skipif(not solver_binary().is_file(), reason="requires reviewed real serial MFEM")
def test_live_selection_captures_pilots_then_runs_full_verifier(tmp_path):
    source = tmp_path / "calibration"
    campaign(Namespace(directory=source, scale="compact", workloads=["C1","C2"], cpu=min(os.sched_getaffinity(0)),
                       seed=122, warmups=0, repetitions=1, study_timeout=180., wall_budget=300.,
                       min_free_gib=0., max_output_gib=1., compact_archives=False))
    key = sha256_file(source / "campaign.json")
    output = tmp_path / "select"
    result = run_selection(output, source, key, "H2", measure_pilots=True)
    assert result["eligible"] and result["solver"] == "cg_gs"
    decision = json.loads((output / "decision.json").read_bytes())
    assert result["decision_sha256"] == sha256_file(output / "decision.json")
    assert decision["performance_provenance"]["matched"]
    assert not decision["fit"]["enabled"] and decision["fit"]["reason"] == "unresolved_calibration"
    assert result["pilot_seconds"] > 0 and decision["elapsed_before_execution_seconds"] >= result["pilot_seconds"]
    assert result["elapsed_seconds"] >= result["final"]["elapsed_seconds"] + result["pilot_seconds"]
    assert {s: [x["n"] for x in levels] for s, levels in decision["pilots"].items()} == {
        "cg_gs":[8,16], "cg_jacobi":[8,16]}
    assert decision["request_sha256"] == sha256_of(make_request(workload_definitions("compact")["H2"], "gs"))
    archive = output / "final" / result["final"]["archive"]
    assert reverify(archive / "attempts/attempt_01", result["final"]["manifest_sha256"])["verdict"] == "ACCEPTED"
    refused = run_selection(tmp_path / "low_memory", source, key, "H2", memory_mib=64)
    assert not refused["eligible"] and not (tmp_path / "low_memory" / "pilots").exists()
    # Reproduce an older calibration receipt lacking source pins. Requested pilots
    # still measure the new environment, but cannot authorize a calibrated switch.
    cfg_path = source / "config.json"
    cfg = json.loads(cfg_path.read_bytes())
    cfg.pop("source_sha256")
    cfg_path.write_text(json.dumps(cfg))
    report_path = source / "campaign.json"
    updated = json.loads(report_path.read_bytes())
    updated["config_sha256"] = sha256_file(cfg_path)
    report_path.write_text(json.dumps(updated))
    stale = tmp_path / "stale"
    fallback = run_selection(stale, source, sha256_file(report_path), "H2", measure_pilots=True)
    detail = json.loads((stale / "decision.json").read_bytes())
    assert fallback["eligible"] and fallback["solver"] == "cg_gs" and fallback["pilot_seconds"] > 0
    assert detail["fit"]["reason"] == "unresolved_calibration" and not detail["automatic_switch_enabled"]
    assert detail["selection"]["reason"] == "calibration_provenance_mismatch"
    assert set(detail["performance_provenance"]["differences"]) == set(TIMED_SOURCES)
    only = run_selection(tmp_path / "single", source, sha256_file(report_path), "H2", authorized_solvers=["cg_jacobi"])
    assert only["eligible"] and only["solver"] == "cg_jacobi" and only["pilot_seconds"] == 0
    # A receipt for an unvalidated arm must never train the policy, even with otherwise valid hashes.
    report = json.loads((source / "campaign.json").read_bytes())
    report["records"][0]["verifier_verdict"] = "REJECTED"
    (source / "campaign.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="inconsistent eligible"):
        calibration(source, sha256_file(source / "campaign.json"))
