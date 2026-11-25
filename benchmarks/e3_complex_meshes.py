"""E3 Task 2 benchmark: Unstructured Gmsh, variable kappa, P1-P4/Q1-Q4, Example 6 AMR, and period degradation.

Runs real MFEM C++ binaries (poisson_mms and poisson_amr) + independent Python verifiers:
  1. Unstructured Gmsh MSH 2.2 (Delaunay tri + curved/distorted quad) with variable kappa and P1..P4 / Q1..Q4.
  2. MFEM Example 6 AMR (ZienkiewiczZhuEstimator + ThresholdRefiner) on gaussian_steep and lshape_singularity
     vs uniform refinement, verified independently by verifier/amr.py (T-junction constraints + C^T K C).
  3. Applicability & degradation of repeated_element_period across Cartesian constant-kappa (period 1..2),
     variable-kappa, and unstructured/graded meshes (period == E), including time/memory penalties.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
from sciagent import mfem_runner
from sciagent.gmsh import generate_unstructured_msh, inspect_gmsh_unstructured
from sciagent.spec import ProblemSpec, StudySpec, ValidationSpec
from verifier import expr, verifier
from verifier.amr import verify_amr_run
from verifier.fem import build_local_basis, eval_monomials, evaluate_quadrature_slice, load_mesh_data, reference_quadrature, repeated_element_period


def _find_lshape_mesh() -> Path:
    candidates = [
        ROOT / "third_party/mfem-src/data/l-shape.mesh",
    ]
    for env_key in ("MFEM_SRC", "THIRD_PARTY"):
        env_val = os.environ.get(env_key)
        if not env_val:
            continue
        base = Path(env_val)
        candidates.append(base / "data/l-shape.mesh")
        candidates.append(base / "mfem-src/data/l-shape.mesh")
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("cannot locate MFEM l-shape.mesh (expected under third_party/mfem-src/data/l-shape.mesh)")


def _problem_var(neumann: tuple[str, ...] = ("top",)) -> ProblemSpec:
    sides = [(1, "bottom"), (2, "right"), (3, "top"), (4, "left")]
    return ProblemSpec.model_validate({
        "id": "sin_sin_variable",
        "equation": "-div(kappa*grad(u)) = f",
        "domain": "unit_square_2d",
        "kappa": "1+x",
        "exact_solution": "sin(pi*x)*sin(2*pi*y)",
        "source": "5*pi**2*(1+x)*sin(pi*x)*sin(2*pi*y)-pi*cos(pi*x)*sin(2*pi*y)",
        "boundaries": [
            {"attribute": a, "label": lbl, "kind": "neumann" if lbl in neumann else "dirichlet", "value": "from_exact_solution"}
            for a, lbl in sides
        ],
    })


def _problem_steep() -> ProblemSpec:
    sides = [(1, "bottom"), (2, "right"), (3, "top"), (4, "left")]
    return ProblemSpec.model_validate({
        "id": "gaussian_steep",
        "equation": "-laplacian(u) = f",
        "domain": "unit_square_2d",
        "exact_solution": "exp(-120*((x-0.375)**2+(y-0.625)**2))",
        "source": "(480-57600*((x-0.375)**2+(y-0.625)**2))*exp(-120*((x-0.375)**2+(y-0.625)**2))",
        "boundaries": [{"attribute": a, "label": lbl, "kind": "dirichlet", "value": "from_exact_solution"} for a, lbl in sides],
    })


def run_unstructured_and_high_order(work_root: Path, mms_bin: Path) -> list[dict]:
    """Benchmark unstructured Delaunay tri and distorted quad across orders p=1..4 with variable kappa."""
    results = []
    prob = _problem_var(("top",))
    for geom in ("tri", "quad"):
        msh_bytes = generate_unstructured_msh(geom, nx=8, seed=42 if geom == "tri" else 77, graded_focus=(0.375, 0.625))
        msh_sha = hashlib.sha256(msh_bytes).hexdigest()
        msh_path = work_root / f"unstructured_{geom}.msh"
        msh_path.write_bytes(msh_bytes)
        for p in (1, 2, 3, 4):
            # Fewer refinements for high p so DOFs span a comparable practical envelope
            refs = (0, 1, 2) if p <= 2 else (0, 1)
            mesh_levels = tuple(8 * (2 ** r) for r in refs)
            spec = StudySpec.model_validate({
                "schema_version": "0.3",
                "problem": prob.model_dump(mode="json"),
                "execution": {
                    "mesh_type": geom,
                    "mesh_topology": "unstructured",
                    "polynomial_order": p,
                    "mesh_levels": list(mesh_levels),
                    "assembly": "full",
                    "solver": "cg_gs",
                    "rel_tol": 1e-12,
                },
                "external_mesh": {"file": msh_path.name, "sha256": msh_sha},
            })
            grid = inspect_gmsh_unstructured(msh_bytes, spec.problem, spec.execution)
            lvl_records = []
            for r, n in zip(refs, mesh_levels):
                lvl_dir = work_root / f"unstruct_{geom}_p{p}_r{r}"
                lr = mfem_runner.run_level(
                    spec.problem, spec.execution, "sin_sin_variable", n, lvl_dir,
                    timeout_s=120.0, mem_mb=2048, binary=mms_bin, mesh_path=msh_path, refinements=r,
                )
                rep = verifier.verify_level(
                    lvl_dir, spec.problem, spec.execution, spec.validation, n,
                    require_run_record=True, external_grid=grid,
                )
                lvl_records.append({
                    "refinement": r, "n": n, "dofs": rep.num_dofs,
                    "ok": rep.ok, "failed_codes": [c.code for c in rep.checks if not c.passed],
                    "l2_error": rep.l2_error, "h1_semi_error": rep.h1_semi_error,
                    "reference_rel_diff": rep.reference_rel_diff,
                    "wall_seconds": lr.wall_seconds,
                })
            # Empirical DOF convergence order between last two levels
            a, b = lvl_records[-2], lvl_records[-1]
            l2_h_order = math.log(a["l2_error"] / b["l2_error"]) / math.log(2.0)
            h1_h_order = math.log(a["h1_semi_error"] / b["h1_semi_error"]) / math.log(2.0)
            results.append({
                "geom": geom, "order": p, "all_ok": all(r["ok"] for r in lvl_records),
                "observed_l2_order": round(l2_h_order, 3),
                "observed_h1_order": round(h1_h_order, 3),
                "levels": lvl_records,
            })
            print(json.dumps({"stage": "unstructured", "geom": geom, "p": p,
                              "ok": all(r["ok"] for r in lvl_records),
                              "l2_order": round(l2_h_order, 3), "h1_order": round(h1_h_order, 3)}), flush=True)
    return results


def run_ex6_amr_vs_uniform(work_root: Path, mms_bin: Path, amr_bin: Path) -> dict:
    """Compare MFEM Example 6 AMR (verified via verifier/amr.py) against uniform refinement."""
    steep_prob = _problem_steep()
    records: dict[str, dict] = {}

    # 1. Steep Gaussian: quad NCMesh AMR vs quad Uniform (orders 1 & 2) + tri conforming AMR (order 2)
    for label, mt, p, init_n, max_steps, max_dofs in (
        ("steep_amr_quad_q1", "quad", 1, 4, 8, 20000),
        ("steep_amr_quad_q2", "quad", 2, 4, 7, 20000),
        ("steep_amr_tri_p2", "tri", 2, 8, 6, 20000),
    ):
        amr_dir = work_root / label
        run_info = mfem_runner.run_amr(
            "gaussian_steep", mt, p, amr_dir, n=init_n, assembly="full", solver="cg_gs",
            max_steps=max_steps, max_dofs=max_dofs, theta=0.7, binary=amr_bin,
        )
        ver = verify_amr_run(amr_dir, steep_prob, ValidationSpec())
        assert run_info["exit_code"] == 0, f"{label} exited with {run_info['exit_code']}"
        assert ver["verdict"] == "ACCEPTED", f"{label} verification failed: {ver['failure_codes']}"
        records[label] = {
            "case": "gaussian_steep", "mesh_type": mt, "order": p,
            "wall_seconds": run_info["wall_seconds"], "verdict": ver["verdict"],
            "steps": [
                {
                    "step": s["step"],
                    "true_dofs": s["num_true_dofs_inferred"],
                    "full_dofs": s["num_dofs"],
                    "hanging_dofs": s["hanging_dof_count"],
                    "max_hanging_err": s["max_hanging_continuity_error"],
                    "elements": s["num_elements"],
                    "l2_error": s["l2_error"],
                    "h1_semi_error": s["h1_semi_error"],
                    "reference_rel_diff": s["reference_rel_diff"],
                }
                for s in ver["steps"]
            ],
        }
        print(json.dumps({"stage": "amr", "name": label, "steps": len(ver["steps"]),
                          "final_tdofs": ver["steps"][-1]["num_true_dofs_inferred"],
                          "final_h1": ver["steps"][-1]["h1_semi_error"],
                          "verdict": ver["verdict"]}), flush=True)

    # Uniform Q1 and Q2 baselines on gaussian_steep
    for uname, p, ns in (("steep_uniform_quad_q1", 1, (8, 16, 32, 64, 128)),
                         ("steep_uniform_quad_q2", 2, (4, 8, 16, 32, 64))):
        u_steps = []
        ex = StudySpec.model_validate({
            "schema_version": "0.4", "problem": steep_prob.model_dump(mode="json"),
            "execution": {"mesh_type": "quad", "polynomial_order": p, "mesh_levels": list(ns),
                          "assembly": "full", "solver": "cg_gs"},
        }).execution
        for n in ns:
            ld = work_root / f"{uname}_n{n}"
            lr = mfem_runner.run_level(steep_prob, ex, "gaussian_steep", n, ld, 120.0, 2048, mms_bin)
            rep = verifier.verify_level(ld, steep_prob, ex, ValidationSpec(), n)
            u_steps.append({
                "n": n, "true_dofs": rep.num_dofs, "l2_error": rep.l2_error,
                "h1_semi_error": rep.h1_semi_error, "wall_seconds": lr.wall_seconds, "ok": rep.ok,
            })
        records[uname] = {"case": "gaussian_steep", "mesh_type": "quad", "order": p, "steps": u_steps}

    # 2. Classic Example 6 L-shape re-entrant corner singularity: Q1 & Q2 AMR vs Uniform refinement
    lshape_msh = _find_lshape_mesh()
    for lname, p, steps in (("lshape_amr_q1", 1, 9), ("lshape_amr_q2", 2, 8)):
        amr_dir = work_root / lname
        run_info = mfem_runner.run_amr(
            "lshape_singularity", "quad", p, amr_dir, dirichlet="1", neumann="none",
            mesh_path=lshape_msh, assembly="full", solver="cg_gs", max_steps=steps,
            max_dofs=20000, theta=0.7, binary=amr_bin,
        )
        ver = verify_amr_run(amr_dir, None, ValidationSpec())
        assert run_info["exit_code"] == 0 and ver["verdict"] == "ACCEPTED", f"{lname}: {ver}"
        records[lname] = {
            "case": "lshape_singularity", "mesh_type": "quad", "order": p,
            "wall_seconds": run_info["wall_seconds"], "verdict": ver["verdict"],
            "steps": [
                {
                    "step": s["step"],
                    "true_dofs": s["num_true_dofs_inferred"],
                    "hanging_dofs": s["hanging_dof_count"],
                    "max_hanging_err": s["max_hanging_continuity_error"],
                    "elements": s["num_elements"],
                    "l2_error": s["l2_error"],
                    "h1_semi_error": s["h1_semi_error"],
                    "reference_rel_diff": s["reference_rel_diff"],
                }
                for s in ver["steps"]
            ],
        }
        print(json.dumps({"stage": "amr", "name": lname, "steps": len(ver["steps"]),
                          "final_tdofs": ver["steps"][-1]["num_true_dofs_inferred"],
                          "final_h1": ver["steps"][-1]["h1_semi_error"],
                          "verdict": ver["verdict"]}), flush=True)

    # Uniform Q1 & Q2 on l-shape using poisson_amr with theta -> impossible or uniform initial refinements:
    # Run coarse-to-fine uniform l-shape steps via theta=0.99999 (refines all elements at step 0..k)
    for ulname, p, steps in (("lshape_uniform_q1", 1, 6), ("lshape_uniform_q2", 2, 5)):
        u_dir = work_root / ulname
        run_info = mfem_runner.run_amr(
            "lshape_singularity", "quad", p, u_dir, dirichlet="1", neumann="none",
            mesh_path=lshape_msh, assembly="full", solver="cg_gs", max_steps=steps,
            max_dofs=60000, theta=0.999999, binary=amr_bin,
        )
        ver = verify_amr_run(u_dir, None, ValidationSpec())
        records[ulname] = {
            "case": "lshape_singularity", "mesh_type": "quad", "order": p,
            "verdict": ver["verdict"],
            "steps": [
                {
                    "step": s["step"], "true_dofs": s["num_true_dofs_inferred"],
                    "elements": s["num_elements"], "l2_error": s["l2_error"], "h1_semi_error": s["h1_semi_error"],
                }
                for s in ver["steps"]
            ],
        }
    return records


def run_repeated_period_degradation(work_root: Path, mms_bin: Path) -> list[dict]:
    """Measure repeated_element_period hit vs degradation across Cartesian, variable-kappa, and unstructured meshes."""
    scenarios = []
    # 1. Cartesian P1/Q1 constant kappa (`sin_sin`), 2. Cartesian P1 variable kappa (`sin_sin_variable`),
    # 3. Unstructured Delaunay P1 (`sin_sin`), 4. Distorted curved Q1 (`sin_sin`)
    var_prob = _problem_var(())
    const_prob = StudySpec.model_validate(json.loads((ROOT / "specs/poisson_sin_sin_p1_tri.json").read_text())).problem

    configs = [
        ("cartesian_tri_p1_const", const_prob, "sin_sin", "tri", "cartesian", 128, None),
        ("cartesian_quad_q1_const", const_prob, "sin_sin", "quad", "cartesian", 128, None),
        ("cartesian_tri_p1_var_kappa", var_prob, "sin_sin_variable", "tri", "cartesian", 128, None),
        ("unstructured_tri_p1_const", const_prob, "sin_sin", "tri", "unstructured", 64, 0.22),
        ("unstructured_quad_q1_const", const_prob, "sin_sin", "quad", "unstructured", 64, 0.22),
    ]
    for name, prob, cid, geom, topo, nx, jitter in configs:
        lvl_dir = work_root / f"period_{name}"
        ex = StudySpec.model_validate({
            "schema_version": "0.3", "problem": prob.model_dump(mode="json"),
            "execution": {"mesh_type": geom, "mesh_topology": topo, "polynomial_order": 1, "mesh_levels": [nx],
                          "assembly": "full", "solver": "cg_gs"},
            **({"external_mesh": {"file": f"m_{name}.msh", "sha256": "0" * 64}} if topo == "unstructured" else {}),
        }).execution
        mesh_path = None
        grid = None
        if topo == "unstructured":
            mbytes = generate_unstructured_msh(geom, nx=nx, seed=99, jitter=jitter or 0.22)
            mesh_path = work_root / f"m_{name}.msh"
            mesh_path.write_bytes(mbytes)
            grid = inspect_gmsh_unstructured(mbytes, prob, ex)
        mfem_runner.run_level(prob, ex, cid, nx, lvl_dir, 120.0, 2048, mms_bin, mesh_path=mesh_path, refinements=0)
        mesh = load_mesh_data(lvl_dir)
        basis = build_local_basis(mesh, 1)
        E = mesh.verts.shape[0]
        m = 6
        xi, eta, w = reference_quadrature(mesh.geom, m)
        monom = eval_monomials(basis.exps, xi, eta)

        t_det0 = time.perf_counter()
        period_edges = repeated_element_period(mesh.verts[:, 1:] - mesh.verts[:, :1])
        t_det = time.perf_counter() - t_det0

        # Compare assembly time WITH Ke cache check vs full element-by-element stiffness integration
        kappa_fn = expr.compile_expr(prob.kappa)
        var_kappa = None if prob.kappa == "1" else kappa_fn
        can_reuse = (var_kappa is None and basis.coeffs.strides[0] == 0 and period_edges <= 8)

        t_reuse0 = time.perf_counter()
        if can_reuse:
            qp = evaluate_quadrature_slice(mesh, basis, xi, eta, w, 0, period_edges, monom_cache=monom)
            Ke_sh = verifier._stiffness_block(qp.gphi, qp.wdet, qp.X[..., 0], qp.X[..., 1], None, prob.equation)
            # Replicate stiffness indexing workload
            _ = Ke_sh[np.arange(E) % period_edges]
        else:
            for s0 in range(0, E, max(1, 32768 // xi.size)):
                s1 = min(s0 + max(1, 32768 // xi.size), E)
                qp = evaluate_quadrature_slice(mesh, basis, xi, eta, w, s0, s1, monom_cache=monom)
                _ = verifier._stiffness_block(qp.gphi, qp.wdet, qp.X[..., 0], qp.X[..., 1], var_kappa, prob.equation)
        t_eval = time.perf_counter() - t_reuse0

        # Also measure naive full no-cache baseline for comparison on same mesh
        t_naive0 = time.perf_counter()
        for s0 in range(0, E, max(1, 32768 // xi.size)):
            s1 = min(s0 + max(1, 32768 // xi.size), E)
            qp = evaluate_quadrature_slice(mesh, basis, xi, eta, w, s0, s1, monom_cache=monom)
            _ = verifier._stiffness_block(qp.gphi, qp.wdet, qp.X[..., 0], qp.X[..., 1], var_kappa, prob.equation)
        t_naive = time.perf_counter() - t_naive0

        rec = {
            "scenario": name, "geom": geom, "topology": topo, "kappa": prob.kappa,
            "elements": E, "detected_period": period_edges, "cache_hit": can_reuse,
            "basis_shared_stride0": basis.coeffs.strides[0] == 0,
            "period_scan_s": round(t_det, 5),
            "ke_assembly_with_policy_s": round(t_eval, 5),
            "ke_assembly_no_cache_s": round(t_naive, 5),
            "ke_speedup_vs_nocache": round(t_naive / max(t_eval, 1e-9), 2),
        }
        scenarios.append(rec)
        print(json.dumps({"stage": "period", **rec}), flush=True)
    return scenarios


def run_complex_suite(
    out_dir: Path = ROOT / "evidence/e3",
    mms_bin: Path = ROOT / "build/cpp/poisson_mms",
    amr_bin: Path = ROOT / "build/cpp/poisson_amr",
) -> dict:
    work_root = Path(tempfile.mkdtemp(prefix="e3_complex_"))
    try:
        unstruct = run_unstructured_and_high_order(work_root, mms_bin)
        amr = run_ex6_amr_vs_uniform(work_root, mms_bin, amr_bin)
        degradation = run_repeated_period_degradation(work_root, mms_bin)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "e3_complex_meshes_v1",
        "unstructured_high_order": unstruct,
        "ex6_amr_vs_uniform": amr,
        "repeated_element_period_degradation": degradation,
    }
    (out_dir / "complex_meshes_and_amr.json").write_text(json.dumps(report, indent=2) + "\n")

    # SVG 1: Ex6 AMR vs Uniform H1-seminorm error vs DOFs
    amr_series = []
    style = {
        "steep_uniform_quad_q1": ("Steep Gauss Uniform Q1", "#d62728", True),
        "steep_amr_quad_q1": ("Steep Gauss Ex6 AMR Q1", "#ff7f0e", False),
        "steep_uniform_quad_q2": ("Steep Gauss Uniform Q2", "#9467bd", True),
        "steep_amr_quad_q2": ("Steep Gauss Ex6 AMR Q2", "#1f77b4", False),
        "lshape_uniform_q2": ("L-Shape Uniform Q2", "#8c564b", True),
        "lshape_amr_q2": ("L-Shape Ex6 AMR Q2", "#2ca02c", False),
    }
    for key, (lbl, col, dash) in style.items():
        steps = amr[key]["steps"]
        amr_series.append({
            "label": lbl, "color": col, "dashed": dash,
            "x": [s["true_dofs"] for s in steps], "y": [s["h1_semi_error"] for s in steps],
        })
    render_line_chart(
        out_dir / "fig_amr_vs_uniform.svg",
        title="MFEM Example 6 AMR vs Uniform Refinement (Verified H1-Seminorm Error)",
        xlabel="True Unconstrained DOFs (N_true)",
        ylabel="Verified H1-Seminorm Error",
        series=amr_series,
    )

    # SVG 2: Unstructured Gmsh variable-kappa P1-P4 / Q1-Q4 convergence
    ho_series = []
    for rec in unstruct:
        ho_series.append({
            "label": f"Unstruct {rec['geom'].upper()} P{rec['order']} (L2 order~{rec['observed_l2_order']})",
            "x": [lv["dofs"] for lv in rec["levels"]],
            "y": [lv["l2_error"] for lv in rec["levels"]],
            "dashed": rec["geom"] == "quad",
        })
    render_line_chart(
        out_dir / "fig_higher_order_convergence.svg",
        title="Unstructured Gmsh Variable-Kappa Convergence (Orders 1 to 4)",
        xlabel="Degrees of Freedom (N)",
        ylabel="Verified L2 Error",
        series=ho_series,
    )

    # SVG 3: repeated_element_period speedup / degradation bar chart
    bars = [
        {"label": s["scenario"].replace("_const", "").replace("_", " "),
         "sublabel": f"period={s['detected_period']}, hit={s['cache_hit']}",
         "value": s["ke_speedup_vs_nocache"]}
        for s in degradation
    ]
    render_bar_comparison(
        out_dir / "fig_period_degradation.svg",
        title="Element Stiffness Cache Speedup vs Complex-Mesh Degradation",
        ylabel="Speedup vs Per-Element Assembly (x)",
        bars=bars,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evidence/e3")
    parser.add_argument("--mms-bin", type=Path, default=ROOT / "build/cpp/poisson_mms")
    parser.add_argument("--amr-bin", type=Path, default=ROOT / "build/cpp/poisson_amr")
    args = parser.parse_args(argv)
    run_complex_suite(args.out_dir, args.mms_bin, args.amr_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
