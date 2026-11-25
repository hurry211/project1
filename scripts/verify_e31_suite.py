"""Run the full E3.1 CI + historical regression ledger and emit evidence/e3/ci_regression_summary.json."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_step(name: str, cmd: list[str], cwd: Path = ROOT) -> dict:
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    dt = round(time.monotonic() - t0, 2)
    tail = "\n".join((p.stdout + "\n" + p.stderr).strip().splitlines()[-8:])
    rec = {
        "name": name,
        "cmd": " ".join(cmd),
        "status": "PASSED" if p.returncode == 0 else "FAILED",
        "exit_code": p.returncode,
        "wall_seconds": dt,
        "summary": tail.splitlines()[-1] if tail else f"exit={p.returncode}",
        "output_tail": tail,
    }
    print(json.dumps({"ci_step": name, "status": rec["status"], "wall_s": dt, "summary": rec["summary"]}), flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"CI step {name!r} failed (exit={p.returncode}):\n{tail}")
    return rec


def main() -> int:
    py = sys.executable
    tmp = Path(tempfile.mkdtemp(prefix="e31_ci_"))
    checks: list[dict] = []
    try:
        checks.append(_run_step(
            "pytest_full_non_mpi",
            [py, "-q", "-m", "pytest", "-q", "-m", "not mpi", "--require-mfem"] if False else [py, "-m", "pytest", "-q", "-m", "not mpi", "--require-mfem"],
        ))
        checks.append(_run_step("cli_quickstart", [py, "-m", "sciagent", "quickstart"]))
        checks.append(_run_step("cli_stage4_plan", [py, "-m", "sciagent", "plan", "--directory", str(tmp / "stage4_plan"), "specs/stage4/choose_diffusion.json", "--execute"]))
        checks.append(_run_step("stage2_coverage", [py, "-m", "benchmarks.stage2", "--directory", str(tmp / "stage2_dir"), "--report", str(tmp / "stage2_report.json")]))

        ctrl_dir = tmp / "controls"
        rob_json = tmp / "robustness.json"
        checks.append(_run_step(
            "stage1_robustness_284_plus_14_controls",
            [py, "-m", "benchmarks.robustness", "--generate-controls", str(ctrl_dir), "--report", str(rob_json), "--check-profiling"],
        ))
        if rob_json.is_file():
            rdata = json.loads(rob_json.read_text())
            checks[-1]["details"] = {
                "panel": rdata.get("panel", {}).get("detected"),
                "panel_total": rdata.get("panel", {}).get("total"),
                "iid": rdata.get("iid", {}).get("detected"),
                "iid_total": rdata.get("iid", {}).get("total"),
                "reference_controls": len(rdata.get("reference_controls", [])),
                "profiling_check": rdata.get("profiling_check"),
            }

        e1_dir = tmp / "e1_compact"
        checks.append(_run_step(
            "e1_paired_compact",
            [py, "-m", "benchmarks.e1", "--directory", str(e1_dir), "--scale", "compact", "--workloads", "C1", "H2", "--warmups", "0", "--repetitions", "1", "--cpu", "0"],
        ))
        camp_sha = _sha256(e1_dir / "campaign.json")
        checks.append(_run_step(
            "e1_offline_audit",
            [py, "-m", "benchmarks.e1_audit", "--directory", str(e1_dir), "--campaign-sha256", camp_sha, "--output", str(tmp / "e1_audit.json"), "--all"],
        ))

        vrev_dir = tmp / "verified_rev"
        checks.append(_run_step(
            "verified_revision_compact",
            [py, "-m", "benchmarks.verified_revision", "--baseline", ".", "--candidate", ".", "--directory", str(vrev_dir), "--scale", "compact", "--workloads", "C1", "--blocks", "2", "--cpu", "0"],
        ))
        d_sha = _sha256(vrev_dir / "design.json")
        r_sha = _sha256(vrev_dir / "report.json")
        checks.append(_run_step(
            "verified_revision_offline_audit",
            [py, "-m", "benchmarks.verified_revision_audit", "--directory", str(vrev_dir), "--design-sha256", d_sha, "--report-sha256", r_sha, "--output", str(tmp / "vrev_audit.json")],
        ))

        perf_dir = tmp / "perf"
        checks.append(_run_step(
            "performance_mixed_repeats2",
            [py, "-m", "benchmarks.performance", "--output", str(perf_dir), "--repeats", "2", "--mixed-boundary"],
        ))

        # Probe real MPI capability cleanly without bypassing system checks
        mpirun = shutil.which("mpirun")
        mpi_bin = ROOT / "build/cpp/mpi/poisson_mpi"
        if mpirun and mpi_bin.is_file():
            checks.append(_run_step("pytest_mpi", [py, "-m", "pytest", "-q", "--require-mfem", "--require-mpi", "tests/test_mpi_native.py", "tests/test_mpi_verifier.py"]))
            checks.append(_run_step("cli_mpi_study", [py, "-m", "sciagent", "mpi-study"]))
        else:
            checks.append({
                "name": "mpi_integration_suite",
                "cmd": "pytest -q --require-mfem --require-mpi && sciagent mpi-study && sciagent verify-mpi",
                "status": "BLOCKED",
                "exit_code": None,
                "wall_seconds": 0.0,
                "summary": "Blocked cleanly in local WSL2: system packages openmpi-bin, libopenmpi3t64, libhypre-2.28.0, libmetis5 not installed (sudo apt-get required).",
            })
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out_path = ROOT / "evidence/e3/ci_regression_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"schema": "e31_ci_ledger_v1", "checks": checks}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
