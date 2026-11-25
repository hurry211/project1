"""Check the archived floating identity experiment separately from the certifier."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research" / "cfac"
EVIDENCE = ROOT / "evidence" / "cfac"


def test_pinned_microcheck_and_archive_agree_with_review():
    records = EVIDENCE / "microcheck"
    env = json.loads((records / "environment.json").read_bytes())
    raw = (records / "stdout.jsonl").read_bytes()
    assert env["returncode"] == 0
    assert not (records / "stderr.txt").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == env["stdout_sha256"]
    assert hashlib.sha256((RESEARCH / "microcheck.py").read_bytes()).hexdigest() == env["script_sha256"]
    assert env["script_sha256"] == "c11345887563330d06a90e9cb28ebbfd713ad0142c50c6277266fe0d7cf11eee"
    results = [json.loads(line) for line in raw.splitlines()]
    assert [r["n"] for r in results] == [8, 16, 32, 64]
    assert [r["e"] for r in results] == pytest.approx([0.431798, 0.217536, 0.108975, 0.0545137], rel=5e-6)
    assert [r["gs90_smooth"] for r in results] == [9, 31, 121, 479]
    for r in results:
        assert r["L_bub"] <= r["e"] < r["U_opt"] <= r["U_loc"] < r["eta"]
        assert r["U_opt"] == pytest.approx(r["U_opt_direct"], rel=1e-5)
        assert max(r[key] for key in ("ident_res", "shift_res", "Ushift_res", "eq_res_opt")) < 1e-11


def test_optional_larger_run_identifies_its_own_input_and_output():
    records = EVIDENCE / "microcheck128"
    env = json.loads((records / "environment.json").read_bytes())
    raw = (records / "stdout.jsonl").read_bytes()
    assert env["returncode"] == 0 and env["command"][-1] == "128"
    assert hashlib.sha256(raw).hexdigest() == env["stdout_sha256"]
    assert hashlib.sha256((RESEARCH / "microcheck.py").read_bytes()).hexdigest() == env["script_sha256"]
    result = json.loads(raw)
    assert (result["n"], result["dofs"], result["gs90_smooth"]) == (128, 16129, 1912)
    assert result["e"] == pytest.approx(0.0272601)


def test_replay_exact_foundation_receipt():
    from research.cfac.foundation_evidence import measured_cases, serialise

    actual = json.loads(json.dumps(measured_cases(), default=serialise))
    assert actual == json.loads((EVIDENCE / "foundation.json").read_bytes())


def test_inexact_iterate_breaks_reviewed_patch_compatibility(tmp_path):
    target = tmp_path / "probe.json"
    subprocess.run([sys.executable, str(RESEARCH / "probe_inexact_patch.py"), str(target)],
                   env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
                   check=True, timeout=60)
    control, perturbed = json.loads(target.read_bytes())
    assert control["residual_inf"] < 1e-12
    assert control["relative_equilibrium_defect"] < 1e-12
    assert perturbed["residual_inf"] == pytest.approx(1 / 16, abs=1e-12)
    assert 0.05 < perturbed["relative_equilibrium_defect"] < 0.06
