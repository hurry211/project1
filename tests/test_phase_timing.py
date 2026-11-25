import json
from pathlib import Path

import pytest

from conftest import requires_mfem
from sciagent import profiling
from sciagent.controller import run_study
from sciagent.spec import StudySpec
from synthetic import make_attempt
from verifier.verifier import verify_attempt


def test_nested_exclusive_timers_restore_parent_after_failure(monkeypatch):
    clock = iter(range(20))
    monkeypatch.setattr(profiling.time, "monotonic", lambda: next(clock))
    with profiling.capture() as profile:
        with profiling.phase("outer"):
            with pytest.raises(ValueError), profiling.phase("inner"):
                profiling.mark("inner.check")
                raise ValueError
    report = profile.summary()
    assert report["elapsed_seconds"] == 6
    assert report["phase_seconds"] == {"controller.other": 2, "outer": 2, "inner": 1, "inner.check": 1}
    assert not profiling.enabled()


def test_missing_or_invalid_old_binary_telemetry_is_unknown():
    with profiling.capture() as profile:
        profiling.record_native(4, 0.01, "")
        profiling.record_native(8, 0.1, 'MFEM_PROFILE_V1 {"export_seconds":-1,"peak_rss_kib":true}')
        profiling.record_native(16, 0.2, "MFEM_PROFILE_V1 truncated")
    for level in profile.summary()["native_levels"]:
        assert level["peak_rss_kib"] is None and level["export_seconds"] is None


@pytest.mark.parametrize("cap,expected", [(300000, "ACCEPTED"), (100, "REJECTED")])
def test_timing_preserves_every_verdict_field(specs_dir, tmp_path, cap, expected):
    raw = json.loads((specs_dir / "poisson_sin_sin_p1_tri.json").read_text())
    raw["execution"]["mesh_levels"] = [4, 8, 16]
    raw["validation"]["reference_solve_max_dofs"] = cap
    attempt = make_attempt(tmp_path, StudySpec.model_validate(raw))
    baseline = verify_attempt(attempt)
    with profiling.capture() as profile:
        timed = verify_attempt(attempt)
    assert timed == baseline and timed["verdict"] == expected
    report = profile.summary()
    assert sum(report["phase_seconds"].values()) == pytest.approx(report["elapsed_seconds"])
    for name in ("verify.geometry_boundary", "verify.integration", "verify.reference",
                 "verify.field_load", "verify.input_hashes"):
        assert report["phase_seconds"][name] > 0
    if expected == "REJECTED":
        assert "REFERENCE_SKIPPED_TOO_LARGE" in timed["failure_codes"]


@pytest.mark.mfem
@requires_mfem
def test_fresh_real_study_measures_native_memory_and_export(specs_dir, tmp_path):
    with profiling.capture() as profile:
        result = run_study(specs_dir / "poisson_sin_sin_p1_tri.json", tmp_path / "runs")
    assert result["final_state"] == "ACCEPTED"
    report = profile.summary()
    assert report["worker_peak_rss_kib"] > 0
    assert report["phase_seconds"]["native.process"] > 0
    assert report["phase_seconds"]["archive.finalize"] > 0
    assert report["phase_seconds"]["verify.process"] > 0
    assert report["phase_seconds"]["verify.reference"] > 0
    assert sum(report["phase_seconds"].values()) == pytest.approx(report["elapsed_seconds"])
    assert report["verifier_peak_rss_kib"] == result["attempts"][0]["verification"]["peak_rss_kib"] > 0
    for level in report["native_levels"]:
        assert level["peak_rss_kib"] > 0
        assert level["export_seconds"] > 0
        assert sum(level[k] for k in ("setup_seconds", "solve_seconds", "error_seconds", "export_seconds")) <= level["process_seconds"]
    assert len(report["native_levels"]) == 4
    attempt = Path(result["run_dir"]) / "attempts/attempt_01"
    assert json.loads((attempt / "verdict.json").read_bytes()) == verify_attempt(attempt, require_run_records=True)
