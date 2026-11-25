"""E3 Task 1 benchmark: 4k to 1M-DOF verifier scalability, component profiling, and E2 vs E3 plots.

Generates each level's native MFEM artifact once per n, then runs isolated
RLIMIT_AS-bounded verifier subprocesses for:
  - e2_batched_direct: E2 batched element_quadrature + _integrated_errors + SuperLU spsolve
  - e3_streaming_direct: E3 chunked I/O + streaming error/assembly + SuperLU spsolve
  - e3_streaming_amg: E3 chunked I/O + streaming error/assembly + SA-AMG-PCG reference solve
Capturing per-stage times (load, basis, geometry, quadrature, sparse assembly,
AMG setup, AMG/LU solve, evidence archiving), empirical scaling exponents,
and publication SVG plots in evidence/e3/.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks.e3_plots import render_bar_comparison, render_line_chart
from sciagent import profiling
from sciagent.evidence import artifact_hashes
from sciagent.spec import StudySpec
from verifier import expr, verifier
from verifier.fem import build_local_basis, cartesian_partition, element_quadrature, lattice_indices, load_mesh_data


SCALING_NS = (64, 128, 256, 512, 768, 1024)


def _peak_rss_mib() -> float:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024.0, 2)
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)


def _fit_power_law(dofs: list[float], values: list[float]) -> float | None:
    pts = [(math.log(d), math.log(v)) for d, v in zip(dofs, values) if d > 0 and v > 0]
    if len(pts) < 2:
        return None
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    slope, _ = np.polyfit(xs, ys, 1)
    return round(float(slope), 3)


def _artifact_bytes(level_dir: Path) -> int:
    return sum(p.stat().st_size for p in level_dir.iterdir() if p.is_file())


def _run_native_level(n: int, level_dir: Path, spec_path: Path, bin_path: Path) -> dict:
    level_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cmd = [
        str(bin_path),
        "--case", "sin_sin", "--mesh-type", "tri", "--n", str(n),
        "--order", "1", "--dirichlet", "1,2,3,4", "--neumann", "none",
        "--assembly", "full", "--solver", "cg_gs", "--rel-tol", "1e-12",
        "--max-iter", "20000", "--out", str(level_dir),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    native_s = time.perf_counter() - t0
    res = json.loads((level_dir / "result.json").read_text())
    t_arch0 = time.perf_counter()
    hashes = artifact_hashes(level_dir)
    arch_s = time.perf_counter() - t_arch0
    return {
        "n": n,
        "dofs": int(res["num_dofs"]),
        "elements": int(res["num_elements"]),
        "native_s": round(native_s, 4),
        "native_setup_ms": round(float(res.get("time_setup_ms", 0.0)), 2),
        "native_solve_ms": round(float(res.get("time_solve_ms", 0.0)), 2),
        "native_export_ms": round(float(res.get("time_export_ms", 0.0)), 2),
        "native_rss_mib": round(float(res.get("peak_rss_kib", 0.0)) / 1024.0, 2),
        "artifact_mib": round(_artifact_bytes(level_dir) / (1024 * 1024), 2),
        "archive_hash_s": round(arch_s, 4),
        "archive_files": len(hashes),
    }


def _verify_worker(n: int, mode: str, mem_limit_mb: int, level_dir: Path, spec_path: Path) -> dict:
    limit_bytes = mem_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    raw = json.loads(spec_path.read_text())
    raw["execution"]["mesh_levels"] = [n]
    raw["validation"].update({
        "min_levels": 3,
        "reference_solve_max_dofs": 2_000_000,
        "reference_solve_method": "amg_pcg" if mode == "e3_streaming_amg" else "direct",
    })
    spec = StudySpec.model_validate(raw)

    # Basis-build micro-timing without keeping arrays alive before main verifier measurement
    t_b0 = time.perf_counter()
    mesh_probe = load_mesh_data(level_dir)
    t_load_probe = time.perf_counter() - t_b0
    t_b1 = time.perf_counter()
    basis_probe = build_local_basis(mesh_probe, spec.execution.polynomial_order)
    t_basis_probe = time.perf_counter() - t_b1
    del mesh_probe, basis_probe

    if mode == "e2_batched_direct":
        t_v0 = time.perf_counter()
        t_l0 = time.perf_counter()
        mesh = load_mesh_data(level_dir)
        t_l1 = time.perf_counter()
        u_fn = expr.compile_expr(spec.problem.exact_solution)
        f_fn = expr.compile_expr(spec.problem.source)
        m = 2 * spec.execution.polynomial_order + 4
        basis = build_local_basis(mesh, spec.execution.polynomial_order)
        cartesian_partition(mesh, n, verifier.GEOM_TOL)
        lattice_indices(mesh.verts, n, verifier.GEOM_TOL)
        t_g1 = time.perf_counter()
        q = element_quadrature(mesh, basis, m)
        l2, h1 = verifier._integrated_errors(mesh, q, u_fn, spec.problem.equation)
        t_i1 = time.perf_counter()
        coords, _ = verifier._unique_dof_coords(mesh, mesh.values.size)
        on_dir = np.zeros(coords.shape[0], dtype=bool)
        for b in spec.problem.boundaries:
            on_dir |= verifier._side_of_points(coords, b.label)
        u_ref = verifier.reference_solve(
            mesh, basis, q, spec.problem, coords, on_dir, u_fn, f_fn, m, method="direct"
        )
        rel_diff = float(np.max(np.abs(u_ref - mesh.values)) / max(np.max(np.abs(u_ref)), 1e-300))
        t_r1 = time.perf_counter()
        return {
            "n": n,
            "mode": mode,
            "ok": rel_diff <= spec.validation.reference_solve_rel_tol,
            "failed_codes": [] if rel_diff <= spec.validation.reference_solve_rel_tol else ["REFERENCE_MISMATCH"],
            "l2_error": l2,
            "h1_semi_error": h1,
            "reference_rel_diff": rel_diff,
            "verify_s": round(t_r1 - t_v0, 4),
            "phases": {
                "verify.field_load": round(t_l1 - t_l0, 4),
                "verify.geometry_boundary": round(t_g1 - t_l1, 4),
                "verify.integration": round(t_i1 - t_g1, 4),
                "verify.reference": round(t_r1 - t_i1, 4),
            },
            "components": {"mesh_load_s": round(t_load_probe, 4), "basis_build_s": round(t_basis_probe, 4)},
            "peak_rss_mib": _peak_rss_mib(),
        }

    t_v0 = time.perf_counter()
    with profiling.capture() as prof:
        rep = verifier.verify_level(
            level_dir, spec.problem, spec.execution, spec.validation, n, require_run_record=False
        )
    verify_s = time.perf_counter() - t_v0
    summary = prof.summary()
    return {
        "n": n,
        "mode": mode,
        "ok": rep.ok,
        "failed_codes": [c.code for c in rep.checks if not c.passed],
        "l2_error": rep.l2_error,
        "h1_semi_error": rep.h1_semi_error,
        "reference_rel_diff": rep.reference_rel_diff,
        "verify_s": round(verify_s, 4),
        "phases": {k: round(v, 4) for k, v in summary["phase_seconds"].items()},
        "components": {"mesh_load_s": round(t_load_probe, 4), "basis_build_s": round(t_basis_probe, 4)},
        "peak_rss_mib": _peak_rss_mib(),
    }


def run_scaling_suite(
    ns: tuple[int, ...] = SCALING_NS,
    mem_limit_mb: int = 3072,
    out_dir: Path = ROOT / "evidence/e3",
    bin_path: Path = ROOT / "build/cpp/poisson_mms",
    spec_path: Path = ROOT / "specs/poisson_sin_sin_p1_tri.json",
) -> dict:
    modes = ("e2_batched_direct", "e3_streaming_direct", "e3_streaming_amg")
    records: list[dict] = []
    for n in ns:
        tmp = Path(tempfile.mkdtemp(prefix=f"e3_lvl_n{n}_"))
        try:
            lvl = tmp / f"n{n:04d}"
            nat = _run_native_level(n, lvl, spec_path, bin_path)
            for mode in modes:
                proc = subprocess.run(
                    [
                        sys.executable, __file__, "--worker",
                        str(n), mode, str(mem_limit_mb), str(lvl), str(spec_path),
                    ],
                    capture_output=True, text=True, timeout=360,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    vrec = json.loads(proc.stdout.strip().splitlines()[-1])
                else:
                    vrec = {
                        "n": n, "mode": mode, "ok": False,
                        "failed_codes": ["OOM_OR_SOLVER_CRASH"],
                        "error_tail": (proc.stderr or proc.stdout)[-400:],
                        "returncode": proc.returncode,
                    }
                rec = {**nat, **vrec}
                records.append(rec)
                print(json.dumps(rec), flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    by_mode = {m: [r for r in records if r["mode"] == m] for m in modes}
    exponents = {}
    for m, rows in by_mode.items():
        ok_rows = [r for r in rows if r.get("ok")]
        exponents[m] = {
            "verify_time_alpha": _fit_power_law([r["dofs"] for r in ok_rows], [r["verify_s"] for r in ok_rows]),
            "peak_rss_alpha": _fit_power_law([r["dofs"] for r in ok_rows], [r["peak_rss_mib"] for r in ok_rows]),
            "max_verified_dofs": max((r["dofs"] for r in ok_rows), default=0),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "e3_scaling_v1",
        "mem_limit_mb": mem_limit_mb,
        "ns": list(ns),
        "records": records,
        "empirical_exponents": exponents,
    }
    (out_dir / "scaling_4k_to_1m.json").write_text(json.dumps(report, indent=2) + "\n")

    labels = {
        "e2_batched_direct": "E2 Batched + Direct LU",
        "e3_streaming_direct": "E3 Streaming + Direct LU",
        "e3_streaming_amg": "E3 Streaming + SA-AMG-PCG",
    }
    colors = {"e2_batched_direct": "#d62728", "e3_streaming_direct": "#ff7f0e", "e3_streaming_amg": "#1f77b4"}

    time_series, rss_series = [], []
    for m in modes:
        ok_rows = [r for r in by_mode[m] if r.get("ok")]
        if ok_rows:
            time_series.append({
                "label": labels[m], "color": colors[m],
                "x": [r["dofs"] for r in ok_rows], "y": [r["verify_s"] for r in ok_rows],
            })
            rss_series.append({
                "label": labels[m], "color": colors[m],
                "x": [r["dofs"] for r in ok_rows], "y": [r["peak_rss_mib"] for r in ok_rows],
            })

    render_line_chart(
        out_dir / "fig_scaling_time.svg",
        title="Verifier Wall Time Scaling (4k to 1.05M DOFs)",
        xlabel="Degrees of Freedom (N)",
        ylabel="Verifier Wall Time (s)",
        series=time_series,
    )
    render_line_chart(
        out_dir / "fig_scaling_memory.svg",
        title=f"Verifier Peak RSS Scaling (RLIMIT_AS={mem_limit_mb} MiB)",
        xlabel="Degrees of Freedom (N)",
        ylabel="Peak RSS (MiB)",
        series=rss_series,
    )

    amg_max = next((r for r in reversed(by_mode["e3_streaming_amg"]) if r.get("ok")), None)
    if amg_max:
        ph = amg_max["phases"]
        bars = [
            {"label": "Mesh/Field Load", "value": ph.get("verify.field_load", 0.0), "color": "#1f77b4"},
            {"label": "Geom/Boundary", "value": ph.get("verify.geometry_boundary", 0.0), "color": "#2ca02c"},
            {"label": "Quadrature/Error", "value": ph.get("verify.integration", 0.0), "color": "#ff7f0e"},
            {"label": "Assembly+AMG-PCG", "value": ph.get("verify.reference", 0.0), "color": "#9467bd"},
            {"label": "Archive SHA-256", "value": amg_max.get("archive_hash_s", 0.0), "color": "#17becf"},
        ]
        render_bar_comparison(
            out_dir / "fig_scaling_breakdown.svg",
            title=f"E3 Verifier Stage Time Breakdown at {amg_max['dofs']:,} DOFs",
            ylabel="Seconds",
            bars=bars,
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=5, metavar=("N", "MODE", "MEM_MB", "LEVEL_DIR", "SPEC"))
    parser.add_argument("--ns", type=int, nargs="+", default=list(SCALING_NS))
    parser.add_argument("--mem-limit-mb", type=int, default=3072)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evidence/e3")
    parser.add_argument("--bin", type=Path, default=ROOT / "build/cpp/poisson_mms")
    parser.add_argument("--spec", type=Path, default=ROOT / "specs/poisson_sin_sin_p1_tri.json")
    args = parser.parse_args(argv)
    if args.worker:
        n_s, mode, mem_s, lvl_s, spec_s = args.worker
        print(json.dumps(_verify_worker(int(n_s), mode, int(mem_s), Path(lvl_s), Path(spec_s))), flush=True)
        return 0
    run_scaling_suite(tuple(args.ns), args.mem_limit_mb, args.out_dir, args.bin, args.spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
