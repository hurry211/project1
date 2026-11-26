"""E3.1 Task 3: Strict paired scalability study from ~4k to >1M DOFs.

Compares four verifier/reference-solver configurations under identical MFEM binary
(`POISSON_MMS_BIN`), identical mathematical problem (`sin_sin_variable` with mixed
non-zero Neumann boundary flux and `kappa = 1 + x`), identical strict acceptance
tolerances (`self_report_rel_tol=1e-6`, `reference_solve_rel_tol=1e-7`,
`dirichlet_nodal_abs_tol=1e-10`), identical `RLIMIT_AS` memory ceiling, identical
single-core CPU pinning, and identical end-to-end `verify_attempt` timing scope:

  1. `e2_frozen_direct`: Real frozen E2 source tree (`ceb485c`), SuperLU direct
  2. `e3_streaming_direct`: Real E3 source tree, streaming quadrature + SuperLU direct
  3. `e3_sa_amg_pcg`: Real E3 source tree, streaming quadrature + self-written SA-AMG-PCG
  4. `e3_pyamg`: Real E3 source tree (`pyamg==5.4.1`), streaming quadrature + PyAMG-PCG

Also executes an end-to-end `controller.run_study` paired comparison between real
frozen E2 (`ceb485c`) and E3 on multi-level `sin_sin_variable` studies, retains full
replayable raw solution archives under `runs/e3_scaling_archives/`, and logs per-run
errors, residuals, backward errors, CG iterations, peak RSS, and failure reasons.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from contextlib import nullcontext
from pathlib import Path


def _set_single_thread_and_affinity() -> None:
    try:
        os.sched_setaffinity(0, {0})
    except (AttributeError, OSError):
        pass


def _peak_rss_kib() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def _run_worker_cli(args: list[str]) -> int:
    """Isolated child worker applying RLIMIT_AS and sys.path BEFORE importing any project/SciPy modules."""
    mode = args[0]
    repo_root = Path(args[1]).resolve()
    mem_mb = int(args[2])
    limit = mem_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    _set_single_thread_and_affinity()
    # Purge any non-venv repo paths inherited from editable installs so repo_root wins strictly
    sys.path[:] = [str(repo_root)] + [
        p for p in sys.path
        if p and (
            "site-packages" in p
            or "dist-packages" in p
            or not any(tag in p for tag in ("project1-e3", "project1-stage", "project1-e2"))
        )
    ]

    try:
        import sciagent
        from verifier import verifier
        assert Path(sciagent.__file__).resolve().is_relative_to(repo_root), (sciagent.__file__, repo_root)
        assert Path(verifier.__file__).resolve().is_relative_to(repo_root), (verifier.__file__, repo_root)

        if mode == "verify":
            attempt_dir = Path(args[3]).resolve()
            try:
                from verifier import reference_solver
                cap_ref = getattr(reference_solver, "capture_metrics", None)
            except Exception:
                cap_ref = None
            try:
                from sciagent import profiling
                cap_prof = getattr(profiling, "capture", None)
            except Exception:
                cap_prof = None

            t0 = time.perf_counter()
            with (cap_prof() if cap_prof else nullcontext()) as prof, (cap_ref() if cap_ref else nullcontext()) as refm:
                verdict = verifier.verify_attempt(attempt_dir, require_run_records=True)
            wall = time.perf_counter() - t0
            metrics_out = []
            if refm is not None:
                for m in refm:
                    metrics_out.append({
                        "method": m.method,
                        "dofs": m.dofs,
                        "nnz": m.nnz,
                        "iterations": m.iterations,
                        "amg_levels": list(m.amg_levels),
                        "setup_seconds": round(m.setup_seconds, 5),
                        "solve_seconds": round(m.solve_seconds, 5),
                        "residual_norm": m.residual_norm,
                        "relative_residual": m.relative_residual,
                        "backward_error": m.backward_error,
                    })
            payload = {
                "ok": True,
                "verdict": verdict,
                "wall_seconds": round(wall, 5),
                "peak_rss_kib": _peak_rss_kib(),
                "phase_seconds": prof.summary()["phase_seconds"] if prof is not None else None,
                "solver_metrics": metrics_out,
                "module_paths": {"sciagent": sciagent.__file__, "verifier": verifier.__file__},
            }
            print(json.dumps(payload))
            return 0

        if mode == "run_study":
            spec_path = Path(args[3]).resolve()
            runs_dir = Path(args[4]).resolve()
            from sciagent import controller
            t0 = time.perf_counter()
            manifest = controller.run_study(spec_path, runs_dir)
            wall = time.perf_counter() - t0
            rss_self = _peak_rss_kib() or 0
            rss_candidates = [rss_self]
            for att in manifest.get("attempts", []):
                ver_kib = (att.get("verification") or {}).get("peak_rss_kib")
                if isinstance(ver_kib, int):
                    rss_candidates.append(ver_kib)
            print(json.dumps({
                "ok": True,
                "manifest": manifest,
                "wall_seconds": round(wall, 5),
                "peak_rss_kib": max(rss_candidates) if rss_candidates else None,
                "parent_rss_kib": rss_self,
                "module_paths": {"sciagent": sciagent.__file__, "verifier": verifier.__file__},
            }))
            return 0
    except MemoryError:
        print(json.dumps({"ok": False, "reason": "MemoryError", "peak_rss_kib": _peak_rss_kib()}))
        return 2
    raise ValueError(f"unknown worker mode: {mode}")


if __name__ == "__main__" and len(sys.argv) >= 2 and sys.argv[1] == "--worker":
    raise SystemExit(_run_worker_cli(sys.argv[2:]))

import argparse
import math
import random
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks.e3_plots import render_bar_comparison, render_line_chart
from sciagent.evidence import sha256_file
from sciagent.spec import StudySpec, canonical_json, sha256_of

DEFAULT_E2_ROOT = Path(os.environ.get("E2_SOURCE_ROOT", "/home/fy/project1-e2-ceb485c"))
DEFAULT_PYAMG_PYTHON = Path(os.environ.get("PYAMG_PYTHON", "/home/fy/.venvs/e3-amg/bin/python"))
DEFAULT_MMS_BIN = Path(os.environ.get("POISSON_MMS_BIN", ROOT / "build/cpp/poisson_mms"))
DEFAULT_ARCHIVE_ROOT = ROOT / "runs/e3_scaling_archives"
DEFAULT_MEM_LIMIT_MB = 4096
DEFAULT_TIMEOUT_S = 240.0

METHODS = (
    "e2_frozen_direct",
    "e3_streaming_direct",
    "e3_sa_amg_pcg",
    "e3_pyamg",
)


def _var_problem() -> dict:
    sides = [(1, "bottom"), (2, "right"), (3, "top"), (4, "left")]
    return {
        "id": "sin_sin_variable",
        "equation": "-div(kappa*grad(u)) = f",
        "domain": "unit_square_2d",
        "kappa": "1+x",
        "exact_solution": "sin(pi*x)*sin(2*pi*y)",
        "source": "5*pi**2*(1+x)*sin(pi*x)*sin(2*pi*y)-pi*cos(pi*x)*sin(2*pi*y)",
        "boundaries": [
            {"attribute": a, "label": lbl, "kind": "neumann" if lbl == "top" else "dirichlet", "value": "from_exact_solution"}
            for a, lbl in sides
        ],
    }


def _git_head(repo: Path) -> str:
    if not (repo / ".git").exists():
        # Frozen git-archive snapshot without .git metadata
        return "ceb485cb9836c73ce846f62d49ca9591fcb83bfc" if "ceb485c" in repo.name else "unknown"
    try:
        p = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return p.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _worker_env(repo_root: Path, mms_bin: Path) -> dict[str, str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root),
        "POISSON_MMS_BIN": str(mms_bin),
        "MFEM_INSTALL": str(ROOT / "third_party/mfem-install"),
        "THIRD_PARTY": str(ROOT / "third_party"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
    }
    return env


def _build_base_study_spec(levels: list[int], mem_mb: int) -> dict:
    return {
        "schema_version": "0.3",
        "problem": _var_problem(),
        "execution": {
            "mesh_type": "quad",
            "polynomial_order": 1,
            "mesh_levels": levels,
            "assembly": "full",
            "solver": "cg_gs",
            "rel_tol": 1e-12,
        },
        "validation": {
            "reference_solve_max_dofs": 2_500_000,
        },
        "policy": {
            "max_attempts": 1,
            "max_dofs_per_level": 2_500_000,
            "max_total_wall_seconds": 1800.0,
            "per_level_timeout_seconds": 600.0,
            "per_level_memory_mb": mem_mb,
        },
    }


def _link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _create_canonical_archive(
    archive_root: Path,
    n_fine: int,
    mms_bin: Path,
    mem_mb: int,
) -> Path:
    """Create or reuse one complete controller archive per n_fine using the frozen poisson_mms binary."""
    levels = [n_fine // 4, n_fine // 2, n_fine]
    base_dir = archive_root / f"base_n{n_fine:04d}"
    ready_marker = base_dir / ".ready.json"
    if ready_marker.is_file():
        info = json.loads(ready_marker.read_text())
        att = Path(info["attempt_dir"])
        if att.is_dir() and info.get("mms_sha256") == sha256_file(mms_bin):
            return att

    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    spec = _build_base_study_spec(levels, mem_mb)
    spec_path = base_dir / "study_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    # Generate raw MFEM levels + request/frozen JSON via E3 mfem_runner without timing the generation step here
    study = StudySpec.model_validate(spec)
    from sciagent import mfem_runner
    from sciagent.evidence import EvidenceStore

    run_dir = base_dir / "canonical_run"
    store = EvidenceStore(run_dir)
    store.write_bytes_once("request/study_spec.json", (canonical_json(study) + "\n").encode("utf-8"))
    store.write_once("frozen/problem.json", canonical_json(study.problem))
    store.write_once("frozen/validation.json", canonical_json(study.validation))
    store.write_once("frozen/policy.json", canonical_json(study.policy))
    att_rel = "attempts/attempt_01"
    store.write_once(f"{att_rel}/execution.json", canonical_json(study.execution))
    att_dir = store.path(att_rel)
    for lvl in levels:
        lr = mfem_runner.run_level(
            study.problem, study.execution, "sin_sin_variable", lvl,
            att_dir / "levels" / f"n{lvl:04d}",
            timeout_s=600.0, mem_mb=mem_mb, binary=mms_bin,
        )
        if lr.exit_code != 0:
            raise RuntimeError(f"poisson_mms failed on level n={lvl}: {lr.exit_meaning}")

    ready_marker.write_text(json.dumps({
        "attempt_dir": str(att_dir),
        "levels": levels,
        "mms_sha256": sha256_file(mms_bin),
        "problem_sha256": sha256_of(study.problem),
        "execution_sha256": sha256_of(study.execution),
    }, indent=2) + "\n")
    return att_dir


def _prepare_method_view(archive_root: Path, base_attempt: Path, method: str, n_fine: int, mem_mb: int) -> Path:
    """Create a hardlinked attempt view differing only in validation reference_solve_method."""
    if method == "e2_frozen_direct":
        return base_attempt

    levels = [n_fine // 4, n_fine // 2, n_fine]
    spec = _build_base_study_spec(levels, mem_mb)
    if method == "e3_streaming_direct":
        spec["validation"]["reference_solve_method"] = "direct"
    elif method == "e3_sa_amg_pcg":
        spec["validation"]["reference_solve_method"] = "amg_pcg"
        spec["validation"]["reference_solve_amg_threshold_dofs"] = 1
    elif method == "e3_pyamg":
        spec["validation"]["reference_solve_method"] = "pyamg"
        spec["validation"]["reference_solve_amg_threshold_dofs"] = 1
    else:
        raise ValueError(f"unknown method {method}")

    study = StudySpec.model_validate(spec)
    view_dir = archive_root / f"view_n{n_fine:04d}_{method}"
    if view_dir.exists():
        shutil.rmtree(view_dir)
    (view_dir / "request").mkdir(parents=True)
    (view_dir / "frozen").mkdir(parents=True)
    att_dir = view_dir / "attempts/attempt_01"
    att_dir.mkdir(parents=True)

    (view_dir / "request/study_spec.json").write_text(canonical_json(study) + "\n")
    (view_dir / "frozen/problem.json").write_text(canonical_json(study.problem))
    (view_dir / "frozen/validation.json").write_text(canonical_json(study.validation))
    (view_dir / "frozen/policy.json").write_text(canonical_json(study.policy))
    (att_dir / "execution.json").write_text(canonical_json(study.execution))

    base_levels = base_attempt / "levels"
    for fpath in base_levels.rglob("*"):
        if fpath.is_file():
            rel = fpath.relative_to(base_levels)
            _link_file(fpath, att_dir / "levels" / rel)
    return att_dir


def _run_verify_trial(
    method: str,
    attempt_dir: Path,
    e2_root: Path,
    pyamg_python: Path,
    mms_bin: Path,
    mem_mb: int,
    timeout_s: float,
) -> dict:
    if method == "e2_frozen_direct":
        py_exe = Path(sys.executable)
        repo_root = e2_root
    elif method == "e3_pyamg":
        py_exe = pyamg_python
        repo_root = ROOT
    else:
        py_exe = Path(sys.executable)
        repo_root = ROOT

    cmd = [
        str(py_exe),
        str((ROOT / "benchmarks/e3_scaling.py").resolve()),
        "--worker",
        "verify",
        str(repo_root),
        str(mem_mb),
        str(attempt_dir),
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=_worker_env(repo_root, mms_bin),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {
            "verdict": "FAILED",
            "failure_codes": ["VERIFIER_TIMEOUT"],
            "failure_reason": f"timeout after {timeout_s}s",
            "wall_seconds": round(time.perf_counter() - t0, 4),
            "peak_rss_mb": None,
            "solver_metrics": [],
            "errors": [],
        }

    ext_wall = time.perf_counter() - t0
    last_line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.strip().startswith("{")), "")
    combined_out = proc.stdout + proc.stderr
    if proc.returncode == 2 or (proc.returncode != 0 and ("MemoryError" in combined_out or "SUPERLU_MALLOC" in combined_out)):
        reason = (
            f"SuperLU C malloc failed under RLIMIT_AS={mem_mb} MiB"
            if "SUPERLU_MALLOC" in combined_out
            else f"RLIMIT_AS {mem_mb} MiB exceeded (exit={proc.returncode})"
        )
        return {
            "verdict": "FAILED",
            "failure_codes": ["VERIFIER_MEMORY_LIMIT"],
            "failure_reason": reason,
            "wall_seconds": round(ext_wall, 4),
            "peak_rss_mb": None,
            "solver_metrics": [],
            "errors": [],
        }
    if proc.returncode != 0 or not last_line:
        return {
            "verdict": "FAILED",
            "failure_codes": ["VERIFIER_PROCESS_FAILED"],
            "failure_reason": (proc.stderr or proc.stdout or f"exit={proc.returncode}")[-300:],
            "wall_seconds": round(ext_wall, 4),
            "peak_rss_mb": None,
            "solver_metrics": [],
            "errors": [],
        }

    data = json.loads(last_line)
    if not data.get("ok"):
        return {
            "verdict": "FAILED",
            "failure_codes": ["VERIFIER_MEMORY_LIMIT"],
            "failure_reason": data.get("reason", "worker failed"),
            "wall_seconds": round(ext_wall, 4),
            "peak_rss_mb": round(data["peak_rss_kib"] / 1024.0, 2) if data.get("peak_rss_kib") else None,
            "solver_metrics": [],
            "errors": [],
        }

    ver = data["verdict"]
    lvls = ver.get("levels", [])
    rss_kib = data.get("peak_rss_kib")
    return {
        "verdict": ver.get("verdict"),
        "failure_codes": ver.get("failure_codes", []),
        "failure_reason": None if ver.get("verdict") == "ACCEPTED" else (ver.get("statement") or str(ver.get("failure_codes"))),
        "wall_seconds": round(data.get("wall_seconds", ext_wall), 4),
        "peak_rss_mb": round(rss_kib / 1024.0, 2) if rss_kib else None,
        "phase_seconds": data.get("phase_seconds"),
        "module_paths": data.get("module_paths"),
        "solver_metrics": data.get("solver_metrics", []),
        "errors": [
            {
                "n": lv["n"],
                "num_dofs": lv.get("num_dofs"),
                "l2_error": lv.get("l2_error"),
                "h1_semi_error": lv.get("h1_semi_error"),
                "reference_rel_diff": lv.get("reference_rel_diff"),
            }
            for lv in lvls
        ],
    }


def _bootstrap_ci95(values: list[float], *, seed: int = 20261125, n_boot: int = 2000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    meds = sorted(statistics.median(rng.choices(values, k=n)) for _ in range(n_boot))
    return (meds[int(0.025 * (n_boot - 1))], meds[int(0.975 * (n_boot - 1))])


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    s = sorted(values)
    return float(np.percentile(s, 75) - np.percentile(s, 25))


def run_e2_vs_e3_end_to_end(
    e2_root: Path,
    mms_bin: Path,
    archive_root: Path,
    mem_mb: int,
    blocks: int,
) -> list[dict]:
    """Paired randomized end-to-end controller.run_study: frozen E2 (ceb485c) vs E3."""
    results = []
    e2e_root = archive_root / "e2_vs_e3_e2e"
    e2e_root.mkdir(parents=True, exist_ok=True)

    for levels in ([64, 128, 256], [128, 256, 512]):
        spec = _build_base_study_spec(levels, mem_mb)
        spec_file = e2e_root / f"spec_n{levels[-1]}.json"
        spec_file.write_text(json.dumps(spec, indent=2) + "\n")

        block_recs = []
        rng = random.Random(9001 + levels[-1])
        for blk in range(blocks):
            pair = ["e2_ceb485c", "e3"]
            rng.shuffle(pair)
            outcomes = {}
            for order_idx, arm in enumerate(pair):
                repo = e2_root if arm == "e2_ceb485c" else ROOT
                runs_sub = e2e_root / f"n{levels[-1]}_b{blk}_{arm}"
                if runs_sub.exists():
                    shutil.rmtree(runs_sub)
                cmd = [
                    sys.executable,
                    str((ROOT / "benchmarks/e3_scaling.py").resolve()),
                    "--worker", "run_study", str(repo), str(mem_mb), str(spec_file), str(runs_sub),
                ]
                proc = subprocess.run(
                    cmd, cwd=str(repo), env=_worker_env(repo, mms_bin),
                    capture_output=True, text=True, timeout=300.0,
                )
                line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.strip().startswith("{")), "")
                if proc.returncode != 0 or not line:
                    outcomes[arm] = {
                        "order": order_idx, "final_state": "FAILED", "wall_seconds": None,
                        "peak_rss_mb": None, "error": (proc.stderr or proc.stdout)[-240:],
                    }
                else:
                    d = json.loads(line)
                    man = d["manifest"]
                    rss = d.get("peak_rss_kib")
                    outcomes[arm] = {
                        "order": order_idx,
                        "final_state": man.get("final_state"),
                        "wall_seconds": d["wall_seconds"],
                        "peak_rss_mb": round(rss / 1024.0, 2) if rss else None,
                        "run_dir": man.get("run_dir"),
                    }
            block_recs.append({"block": blk, "outcomes": outcomes})

        e2_walls = [b["outcomes"]["e2_ceb485c"]["wall_seconds"] for b in block_recs if b["outcomes"]["e2_ceb485c"]["final_state"] == "ACCEPTED"]
        e3_walls = [b["outcomes"]["e3"]["wall_seconds"] for b in block_recs if b["outcomes"]["e3"]["final_state"] == "ACCEPTED"]
        paired_spd = [
            b["outcomes"]["e2_ceb485c"]["wall_seconds"] / b["outcomes"]["e3"]["wall_seconds"]
            for b in block_recs
            if b["outcomes"]["e2_ceb485c"]["final_state"] == "ACCEPTED" and b["outcomes"]["e3"]["final_state"] == "ACCEPTED"
        ]
        e2_rss = [b["outcomes"]["e2_ceb485c"]["peak_rss_mb"] for b in block_recs if b["outcomes"]["e2_ceb485c"]["peak_rss_mb"]]
        e3_rss = [b["outcomes"]["e3"]["peak_rss_mb"] for b in block_recs if b["outcomes"]["e3"]["peak_rss_mb"]]
        spd_ci = _bootstrap_ci95(paired_spd, seed=70 + levels[-1])
        rec = {
            "levels": levels,
            "finest_dofs": (levels[-1] + 1) ** 2,
            "blocks": blocks,
            "e2_accepted": len(e2_walls),
            "e3_accepted": len(e3_walls),
            "e2_wall_median_s": round(statistics.median(e2_walls), 4) if e2_walls else None,
            "e3_wall_median_s": round(statistics.median(e3_walls), 4) if e3_walls else None,
            "e2_rss_median_mb": round(statistics.median(e2_rss), 2) if e2_rss else None,
            "e3_rss_median_mb": round(statistics.median(e3_rss), 2) if e3_rss else None,
            "paired_speedup_median": round(statistics.median(paired_spd), 3) if paired_spd else None,
            "paired_speedup_ci95": [round(spd_ci[0], 3), round(spd_ci[1], 3)] if paired_spd else None,
            "block_records": block_recs,
        }
        results.append(rec)
        print(json.dumps({"stage": "e2_vs_e3_e2e", **{k: v for k, v in rec.items() if k != "block_records"}}), flush=True)
    return results


def run_scaling_suite(
    *,
    e2_root: Path = DEFAULT_E2_ROOT,
    pyamg_python: Path = DEFAULT_PYAMG_PYTHON,
    mms_bin: Path = DEFAULT_MMS_BIN,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    out_dir: Path = ROOT / "evidence/e3",
    scales_n: tuple[int, ...] = (64, 128, 256, 512, 768, 1024, 1280),
    blocks: int = 5,
    mem_mb: int = DEFAULT_MEM_LIMIT_MB,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    if not (e2_root / "sciagent/controller.py").is_file():
        raise FileNotFoundError(f"frozen E2 checkout not found at {e2_root}")
    if not pyamg_python.is_file():
        raise FileNotFoundError(f"isolated pyamg Python not found at {pyamg_python}")
    if not mms_bin.is_file():
        raise FileNotFoundError(f"MFEM binary not found at {mms_bin}")

    pyamg_ver = subprocess.run(
        [str(pyamg_python), "-c", "import pyamg; print(pyamg.__version__)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    archive_root.mkdir(parents=True, exist_ok=True)
    archive_index = []
    method_views: dict[tuple[int, str], Path] = {}
    for n in scales_n:
        base_att = _create_canonical_archive(archive_root, n, mms_bin, mem_mb)
        archive_index.append({
            "n_fine": n,
            "levels": [n // 4, n // 2, n],
            "true_dofs": (n + 1) ** 2,
            "base_attempt_dir": str(base_att),
            "finest_dof_values_sha256": sha256_file(base_att / f"levels/n{n:04d}/dof_values.txt"),
            "replay_command_e2": f"PYTHONPATH={e2_root} {sys.executable} benchmarks/e3_scaling.py --worker verify {e2_root} {mem_mb} {base_att}",
            "replay_command_e3": f"{sys.executable} benchmarks/e3_scaling.py --worker verify {ROOT} {mem_mb} {base_att}",
        })
        for method in METHODS:
            method_views[(n, method)] = _prepare_method_view(archive_root, base_att, method, n, mem_mb)

    trials = []
    for n in scales_n:
        rng = random.Random(20261125 + n * 17)
        for blk in range(blocks):
            order = list(METHODS)
            rng.shuffle(order)
            for pos, method in enumerate(order):
                att_dir = method_views[(n, method)]
                res = _run_verify_trial(method, att_dir, e2_root, pyamg_python, mms_bin, mem_mb, timeout_s)
                fin_err = res["errors"][-1] if res.get("errors") else {}
                fin_met = res["solver_metrics"][-1] if res.get("solver_metrics") else {}
                trial = {
                    "n_fine": n,
                    "levels": [n // 4, n // 2, n],
                    "true_dofs": (n + 1) ** 2,
                    "block": blk,
                    "order_in_block": pos,
                    "method": method,
                    "code_version": "e2_ceb485c" if method == "e2_frozen_direct" else "e3",
                    "verdict": res["verdict"],
                    "failure_codes": res["failure_codes"],
                    "failure_reason": res["failure_reason"],
                    "wall_seconds": res["wall_seconds"],
                    "peak_rss_mb": res["peak_rss_mb"],
                    "l2_error": fin_err.get("l2_error"),
                    "h1_semi_error": fin_err.get("h1_semi_error"),
                    "reference_rel_diff": fin_err.get("reference_rel_diff"),
                    "solver_iterations": fin_met.get("iterations"),
                    "residual_norm": fin_met.get("residual_norm"),
                    "relative_residual": fin_met.get("relative_residual"),
                    "backward_error": fin_met.get("backward_error"),
                    "setup_seconds": fin_met.get("setup_seconds"),
                    "solve_seconds": fin_met.get("solve_seconds"),
                    "amg_levels": fin_met.get("amg_levels"),
                    "phase_seconds": res.get("phase_seconds"),
                    "module_paths": res.get("module_paths"),
                }
                trials.append(trial)
                print(json.dumps({
                    "stage": "trial", "n": n, "blk": blk, "method": method,
                    "verdict": trial["verdict"], "wall_s": trial["wall_seconds"],
                    "rss_mb": trial["peak_rss_mb"], "iters": trial["solver_iterations"],
                }), flush=True)

    summary = []
    for n in scales_n:
        # Build block lookups for strict mutually-accepted paired speedups
        by_blk_method = {
            (t["block"], t["method"]): t
            for t in trials if t["n_fine"] == n
        }
        for method in METHODS:
            sub = [t for t in trials if t["n_fine"] == n and t["method"] == method]
            acc = [t for t in sub if t["verdict"] == "ACCEPTED"]
            walls = [t["wall_seconds"] for t in acc]
            rss_vals = [t["peak_rss_mb"] for t in acc if t["peak_rss_mb"] is not None]
            iters = [t["solver_iterations"] for t in acc if t["solver_iterations"] is not None]
            rels = [t["relative_residual"] for t in acc if t["relative_residual"] is not None]
            res_norms = [t["residual_norm"] for t in acc if t["residual_norm"] is not None]
            bwds = [t["backward_error"] for t in acc if t["backward_error"] is not None]

            spd_vs_e2 = [
                by_blk_method[(blk, "e2_frozen_direct")]["wall_seconds"] / by_blk_method[(blk, method)]["wall_seconds"]
                for blk in range(blocks)
                if by_blk_method[(blk, "e2_frozen_direct")]["verdict"] == "ACCEPTED"
                and by_blk_method[(blk, method)]["verdict"] == "ACCEPTED"
            ] if method != "e2_frozen_direct" else []
            spd_vs_e3d = [
                by_blk_method[(blk, "e3_streaming_direct")]["wall_seconds"] / by_blk_method[(blk, method)]["wall_seconds"]
                for blk in range(blocks)
                if by_blk_method[(blk, "e3_streaming_direct")]["verdict"] == "ACCEPTED"
                and by_blk_method[(blk, method)]["verdict"] == "ACCEPTED"
            ] if method not in ("e2_frozen_direct", "e3_streaming_direct") else []

            w_ci = _bootstrap_ci95(walls, seed=n + hash(method) % 1000)
            e2_ci = _bootstrap_ci95(spd_vs_e2, seed=n + 101)
            e3d_ci = _bootstrap_ci95(spd_vs_e3d, seed=n + 202)
            row = {
                "n_fine": n,
                "true_dofs": (n + 1) ** 2,
                "method": method,
                "blocks": blocks,
                "n_accepted": len(acc),
                "n_failed": len(sub) - len(acc),
                "failure_reasons": sorted({t["failure_reason"] for t in sub if t["failure_reason"]}),
                "wall_median_s": round(statistics.median(walls), 4) if walls else None,
                "wall_iqr_s": round(_iqr(walls), 4) if walls else None,
                "wall_ci95_s": [round(w_ci[0], 4), round(w_ci[1], 4)] if walls else None,
                "rss_median_mb": round(statistics.median(rss_vals), 2) if rss_vals else None,
                "rss_max_mb": round(max(rss_vals), 2) if rss_vals else None,
                "iterations_median": int(round(statistics.median(iters))) if iters else None,
                "residual_norm_median": float(statistics.median(res_norms)) if res_norms else None,
                "rel_residual_median": float(statistics.median(rels)) if rels else None,
                "backward_error_median": float(statistics.median(bwds)) if bwds else None,
                "l2_error": acc[0]["l2_error"] if acc else None,
                "h1_semi_error": acc[0]["h1_semi_error"] if acc else None,
                "reference_rel_diff": acc[0]["reference_rel_diff"] if acc else None,
                "paired_speedup_vs_e2": round(statistics.median(spd_vs_e2), 3) if spd_vs_e2 else None,
                "paired_speedup_vs_e2_ci95": [round(e2_ci[0], 3), round(e2_ci[1], 3)] if spd_vs_e2 else None,
                "paired_speedup_vs_e3_direct": round(statistics.median(spd_vs_e3d), 3) if spd_vs_e3d else None,
                "paired_speedup_vs_e3_direct_ci95": [round(e3d_ci[0], 3), round(e3d_ci[1], 3)] if spd_vs_e3d else None,
            }
            summary.append(row)

    e2e = run_e2_vs_e3_end_to_end(e2_root, mms_bin, archive_root, mem_mb, blocks)

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "e3_scaling_v2",
        "config": {
            "e2_source_root": str(e2_root),
            "e2_commit": _git_head(e2_root),
            "e3_source_root": str(ROOT),
            "e3_commit": _git_head(ROOT),
            "poisson_mms_bin": str(mms_bin),
            "poisson_mms_sha256": sha256_file(mms_bin),
            "archive_root": str(archive_root),
            "pyamg_python": str(pyamg_python),
            "pyamg_version": pyamg_ver,
            "case_id": "sin_sin_variable",
            "kappa": "1+x",
            "neumann_sides": ["top"],
            "mesh_type": "quad",
            "order": 1,
            "mem_limit_mb": mem_mb,
            "timeout_s": timeout_s,
            "blocks": blocks,
            "scales_n": list(scales_n),
            "tolerances": {"self_report_rel_tol": 1e-6, "reference_solve_rel_tol": 1e-7, "dirichlet_nodal_abs_tol": 1e-10},
        },
        "archives": archive_index,
        "summary": summary,
        "trials": trials,
        "e2_vs_e3_end_to_end": e2e,
    }
    (out_dir / "scaling_4k_to_1m.json").write_text(json.dumps(report, indent=2) + "\n")
    render_scaling_plots(report, out_dir)
    return report


def render_scaling_plots(report: dict, out_dir: Path) -> None:
    """Render scaling SVG plots strictly from scaling_4k_to_1m.json data."""
    summary = report["summary"]
    meta = {
        "e2_frozen_direct": ("E2 Frozen Direct LU (ceb485c)", "#d62728", True),
        "e3_streaming_direct": ("E3 Streaming Direct LU", "#ff7f0e", True),
        "e3_sa_amg_pcg": ("E3 Self-Written SA-AMG-PCG", "#1f77b4", False),
        "e3_pyamg": ("E3 Mature PyAMG-PCG (v5.4.1)", "#2ca02c", False),
    }
    time_series = []
    mem_series = []
    for method, (lbl, col, dash) in meta.items():
        pts = [r for r in summary if r["method"] == method and r["wall_median_s"] is not None]
        if not pts:
            continue
        time_series.append({
            "label": lbl, "color": col, "dashed": dash,
            "x": [r["true_dofs"] for r in pts],
            "y": [r["wall_median_s"] for r in pts],
        })
        mpts = [r for r in pts if r["rss_median_mb"] is not None]
        if mpts:
            mem_series.append({
                "label": lbl, "color": col, "dashed": dash,
                "x": [r["true_dofs"] for r in mpts],
                "y": [r["rss_median_mb"] for r in mpts],
            })

    render_line_chart(
        out_dir / "fig_scaling_wall_time.svg",
        title="End-to-End Verify_Attempt Scaling: Frozen E2 vs E3 Streaming Direct, SA-AMG-PCG, PyAMG",
        xlabel="Finest Mesh Degrees of Freedom (N)",
        ylabel="Median Wall Time per 3-Level Verify_Attempt (s)",
        series=time_series,
    )
    render_line_chart(
        out_dir / "fig_scaling_memory.svg",
        title="Peak Verifier Memory Scaling (VmHWM under RLIMIT_AS=4096 MiB)",
        xlabel="Finest Mesh Degrees of Freedom (N)",
        ylabel="Median Peak RSS Memory (MiB)",
        series=mem_series,
    )


def run_verify_worker_main() -> int:
    ap = argparse.ArgumentParser(description="E3.1 strict paired scaling benchmark (frozen E2 vs E3 vs AMG vs PyAMG).")
    ap.add_argument("--e2-root", type=Path, default=DEFAULT_E2_ROOT)
    ap.add_argument("--pyamg-python", type=Path, default=DEFAULT_PYAMG_PYTHON)
    ap.add_argument("--mms-bin", type=Path, default=DEFAULT_MMS_BIN)
    ap.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evidence/e3")
    ap.add_argument("--scales", type=int, nargs="+", default=[64, 128, 256, 512, 768, 1024, 1280])
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--mem-mb", type=int, default=DEFAULT_MEM_LIMIT_MB)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = ap.parse_args()

    run_scaling_suite(
        e2_root=args.e2_root,
        pyamg_python=args.pyamg_python,
        mms_bin=args.mms_bin,
        archive_root=args.archive_root,
        out_dir=args.out_dir,
        scales_n=tuple(args.scales),
        blocks=args.blocks,
        mem_mb=args.mem_mb,
        timeout_s=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_verify_worker_main())
