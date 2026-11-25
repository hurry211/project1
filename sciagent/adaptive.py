"""Select and execute verified MFEM strategies using problem and hardware features."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from verifier import expr
from verifier.amr import verify_amr_run
from verifier.verifier import ACCEPTED, verify_attempt
from . import capability, environment, adaptive_advisor, mfem_runner
from .gmsh import generate_unstructured_msh, inspect_gmsh_unstructured
from .spec import ANISOTROPIC, ExecutionSpec, ProblemSpec, RetryPolicy, StudySpec, ValidationSpec


def _read_meminfo_mb() -> tuple[float, float]:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        vals = {}
        for line in meminfo.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].rstrip(":") in ("MemTotal", "MemAvailable"):
                vals[parts[0].rstrip(":")] = float(parts[1]) / 1024.0
        if "MemTotal" in vals:
            return round(vals["MemTotal"], 1), round(vals.get("MemAvailable", vals["MemTotal"]), 1)
    return 8192.0, 4096.0


@dataclass(frozen=True)
class HardwareFeatures:
    cpu_count: int
    mem_total_mb: float
    mem_available_mb: float
    mpi_enabled: bool
    suitesparse_enabled: bool
    max_dofs_budget: int
    per_level_memory_mb: int


@dataclass(frozen=True)
class ProblemFeatures:
    case_id: str
    kappa_contrast: float
    is_variable_kappa: bool
    is_anisotropic: bool
    neumann_fraction: float
    target_l2: float
    log10_target_l2: float
    u_max: float
    grad_max: float
    hessian_mean: float
    hessian_max: float
    localization_ratio: float
    is_unstructured: bool


@dataclass(frozen=True)
class StrategyCandidate:
    candidate_id: str
    mesh_type: Literal["tri", "quad"]
    mesh_topology: Literal["cartesian", "unstructured", "amr"]
    polynomial_order: int
    assembly: Literal["full", "partial"]
    solver: Literal["cg_gs", "cg_jacobi", "cg_none"]
    mesh_levels: tuple[int, ...]
    estimated_finest_dofs: int
    backend: Literal["serial_cpu", "mpi_cpu"] = "serial_cpu"
    amr_max_steps: int = 11
    amr_max_dofs: int = 12000


@dataclass(frozen=True)
class AdaptiveTask:
    task_id: str
    split: Literal["calibration", "held_out"]
    case_id: str
    problem: ProblemSpec
    validation: ValidationSpec
    policy: RetryPolicy
    base_mesh_type: Literal["tri", "quad"] = "quad"
    mesh_topology: Literal["cartesian", "unstructured"] = "cartesian"
    unstructured_nx: int = 6
    unstructured_seed: int = 42


def extract_hardware_features(policy: RetryPolicy, env: dict | None = None) -> HardwareFeatures:
    env = env or environment.probe()
    cfg = env.get("mfem", {}).get("config", {})
    tot_mb, avail_mb = _read_meminfo_mb()
    return HardwareFeatures(
        cpu_count=len(os.sched_getaffinity(0)),
        mem_total_mb=tot_mb,
        mem_available_mb=avail_mb,
        mpi_enabled=bool(cfg.get("MFEM_USE_MPI", False)),
        suitesparse_enabled=bool(cfg.get("MFEM_USE_SUITESPARSE", False)),
        max_dofs_budget=int(policy.max_dofs_per_level),
        per_level_memory_mb=int(policy.per_level_memory_mb),
    )


def extract_problem_features(
    problem: ProblemSpec,
    validation: ValidationSpec,
    mesh_topology: str = "cartesian",
    *,
    case_id: str = "",
) -> ProblemFeatures:
    """Compute dimensionless mathematical complexity indicators of the frozen PDE task."""
    xs = np.linspace(0.05, 0.95, 17)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    x, y = X.ravel(), Y.ravel()
    u_fn = expr.compile_expr(problem.exact_solution)
    k_fn = expr.compile_expr(problem.kappa)

    u_vals = u_fn(x, y)
    u_max = max(float(np.max(np.abs(u_vals))), 1e-12)
    gx, gy = expr.gradient(u_fn, x, y, h=1e-3)
    grad_max = max(float(np.max(np.hypot(gx, gy))), 1e-12)

    h = 2e-3
    uxx = (u_fn(x + h, y) - 2.0 * u_vals + u_fn(x - h, y)) / (h * h)
    uyy = (u_fn(x, y + h) - 2.0 * u_vals + u_fn(x, y - h)) / (h * h)
    uxy = (u_fn(x + h, y + h) - u_fn(x + h, y - h) - u_fn(x - h, y + h) + u_fn(x - h, y - h)) / (4.0 * h * h)
    hess_frob = np.sqrt(uxx * uxx + uyy * uyy + 2.0 * uxy * uxy) / u_max
    h_mean = max(float(np.mean(hess_frob)), 1e-12)
    h_max = max(float(np.max(hess_frob)), 1e-12)

    k_vals = np.abs(k_fn(x, y))
    k_contrast = float(np.max(k_vals) / max(float(np.min(k_vals)), 1e-12))
    neu_frac = sum(1 for b in problem.boundaries if b.kind == "neumann") / max(len(problem.boundaries), 1)
    target = validation.max_l2_error_finest if validation.max_l2_error_finest is not None else 1e-4

    return ProblemFeatures(
        case_id=case_id or problem.id,
        kappa_contrast=round(k_contrast, 4),
        is_variable_kappa=problem.kappa != "1",
        is_anisotropic=problem.equation == ANISOTROPIC,
        neumann_fraction=round(neu_frac, 4),
        target_l2=float(target),
        log10_target_l2=round(-math.log10(max(float(target), 1e-16)), 4),
        u_max=round(u_max, 6),
        grad_max=round(grad_max, 6),
        hessian_mean=round(h_mean, 4),
        hessian_max=round(h_max, 4),
        localization_ratio=round(h_max / h_mean, 4),
        is_unstructured=mesh_topology == "unstructured",
    )


def enumerate_candidates(
    task: AdaptiveTask,
    hw: HardwareFeatures,
    env: dict,
    *,
    include_amr: bool = True,
) -> tuple[list[StrategyCandidate], list[dict]]:
    """Build verified feasible strategy candidates + capability exclusion audit log."""
    feasible: list[StrategyCandidate] = []
    excluded: list[dict] = []

    dof_cap = min(hw.max_dofs_budget, task.validation.reference_solve_max_dofs)

    # Probe MPI candidate explicitly so parallel capability is audited without fake metrics
    mpi_exec = ExecutionSpec(
        mesh_type=task.base_mesh_type, mesh_topology=task.mesh_topology,
        polynomial_order=2, mesh_levels=(8, 16, 32), backend="mpi_cpu", assembly="full", solver="hypre_amg",
    )
    mpi_dec = capability.evaluate(mpi_exec, env, task.problem.equation, case_id=task.case_id)
    if mpi_dec.status != "tested":
        excluded.append({"candidate_id": "mpi-hypre-excluded", "status": mpi_dec.status, "code": mpi_dec.code, "reason": mpi_dec.reason})

    if task.mesh_topology == "unstructured":
        n0 = task.unstructured_nx
        min_lv = task.validation.min_levels
        schedules = ((n0, 2 * n0, 4 * n0),) if min_lv >= 3 else ((n0, 2 * n0), (n0, 2 * n0, 4 * n0))
        geoms = (task.base_mesh_type,)
        topos = ("unstructured",)
        mesh_src = "gmsh22"
    else:
        min_lv = task.validation.min_levels
        base_scheds = ((8, 16, 32), (16, 32, 64), (32, 64, 128), (64, 128, 256))
        schedules = base_scheds if min_lv >= 3 else ((8, 16),) + base_scheds
        geoms = ("quad", "tri")
        topos = ("cartesian",)
        mesh_src = "cartesian"

    for topo in topos:
        for mt in geoms:
            for p in (1, 2, 3, 4):
                for asm in ("full", "partial"):
                    for sol in ("cg_gs", "cg_jacobi", "cg_none"):
                        for sched in schedules:
                            ex = ExecutionSpec(
                                mesh_type=mt, mesh_topology=topo, polynomial_order=p,
                                mesh_levels=list(sched), assembly=asm, solver=sol,
                            )
                            dec = capability.evaluate(
                                ex, env, task.problem.equation,
                                exact_solution=task.problem.exact_solution,
                                case_id=task.case_id, mesh_source=mesh_src,
                                e3_verified=True,
                            )
                            dofs = ex.estimated_dofs(sched[-1])
                            cid = f"{topo}-{mt}-p{p}-{asm}-{sol}-n{sched[-1]}"
                            if dec.status != "tested":
                                if sched == schedules[0] and p in (1, 2):
                                    excluded.append({"candidate_id": cid, "status": dec.status, "code": dec.code})
                                continue
                            if dofs > dof_cap:
                                continue
                            # Prune unpreconditioned cg_none on fine meshes where iteration cost explodes needlessly
                            if sol == "cg_none" and dofs > 8000:
                                continue
                            # Prune ultra-huge high-order cartesian runs (> 100k DOFs) so benchmark remains responsive
                            if dofs > 90000:
                                continue
                            feasible.append(StrategyCandidate(
                                candidate_id=cid, mesh_type=mt, mesh_topology=topo, polynomial_order=p,
                                assembly=asm, solver=sol, mesh_levels=sched, estimated_finest_dofs=dofs,
                            ))

    # Add Ex6 AMR candidates on unit-square Poisson problems when requested
    if include_amr and task.mesh_topology != "unstructured" and task.problem.kappa == "1":
        for mt in ("quad",):
            for p in (1, 2):
                for max_dofs in (4000, 12000):
                    ex = ExecutionSpec(
                        mesh_type=mt, mesh_topology="amr", polynomial_order=p,
                        mesh_levels=(8, 16, 32), assembly="full", solver="cg_gs",
                    )
                    dec = capability.evaluate(
                        ex, env, task.problem.equation,
                        exact_solution=task.problem.exact_solution,
                        case_id=task.case_id, mesh_source="amr",
                    )
                    if dec.status != "tested" or max_dofs > dof_cap:
                        continue
                    feasible.append(StrategyCandidate(
                        candidate_id=f"amr-{mt}-p{p}-md{max_dofs}",
                        mesh_type=mt, mesh_topology="amr", polynomial_order=p,
                        assembly="full", solver="cg_gs", mesh_levels=(8,),
                        estimated_finest_dofs=max_dofs, amr_max_steps=10, amr_max_dofs=max_dofs,
                    ))

    return feasible, excluded


def _featurize_pair(pf: ProblemFeatures, cand: StrategyCandidate, hw: HardwareFeatures) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build physics-informed design rows for (error_model, runtime_model, memory_model)."""
    p = float(cand.polynomial_order)
    is_amr = 1.0 if cand.mesh_topology == "amr" else 0.0
    n_eff = math.sqrt(max(cand.estimated_finest_dofs, 4)) / p
    log_n = math.log10(max(n_eff, 2.0))
    log_dofs = math.log10(max(cand.estimated_finest_dofs, 4))
    log_hess = math.log10(max(pf.hessian_mean, 1e-6))
    log_loc = math.log10(max(pf.localization_ratio, 1.0))

    x_err = np.array([
        1.0,
        p,
        log_n,
        (p + 1.0) * log_n,
        p * is_amr,
        is_amr * log_loc,
        log_hess,
        log_loc,
        1.0 if pf.is_variable_kappa else 0.0,
        1.0 if cand.mesh_topology == "unstructured" else 0.0,
        1.0 if cand.mesh_type == "quad" else 0.0,
    ], dtype=np.float64)

    x_time = np.array([
        1.0,
        log_dofs,
        log_dofs * math.log10(p + 1.0),
        float(len(cand.mesh_levels)),
        1.0 if cand.assembly == "partial" else 0.0,
        1.0 if cand.solver == "cg_jacobi" else 0.0,
        1.0 if cand.solver == "cg_none" else 0.0,
        is_amr,
        1.0 if cand.mesh_topology == "unstructured" else 0.0,
        1.0 if pf.is_variable_kappa else 0.0,
        1.0 if cand.mesh_type == "quad" else 0.0,
        1.0 / max(hw.cpu_count, 1),
    ], dtype=np.float64)

    x_mem = np.array([
        1.0,
        log_dofs,
        math.log10(p + 1.0),
        is_amr,
        1.0 if cand.mesh_topology == "unstructured" else 0.0,
    ], dtype=np.float64)
    return x_err, x_time, x_mem


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> np.ndarray:
    d = X.shape[1]
    reg = alpha * np.eye(d, dtype=np.float64)
    reg[0, 0] = 1e-8  # do not shrink intercept
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


class PerformancePredictor:
    """Ridge-regularized log-linear surrogate predicting L2 accuracy, wall time, and peak RSS."""

    def __init__(self, w_err: np.ndarray, w_time: np.ndarray, w_mem: np.ndarray, train_stats: dict):
        self.w_err = w_err
        self.w_time = w_time
        self.w_mem = w_mem
        self.train_stats = train_stats

    @classmethod
    def fit(cls, records: list[dict], alpha: float = 5e-2) -> "PerformancePredictor":
        Xe, Xt, Xm, ye, yt, ym = [], [], [], [], [], []
        for r in records:
            if not r.get("ok") or r.get("l2_error") is None or r["l2_error"] <= 0:
                continue
            xe, xt, xm = _featurize_pair(r["pf"], r["cand"], r["hw"])
            Xe.append(xe); Xt.append(xt); Xm.append(xm)
            ye.append(math.log10(max(r["l2_error"], 1e-16)))
            yt.append(math.log10(max(r["wall_seconds"], 1e-4)))
            ym.append(math.log10(max(r["peak_rss_mb"], 10.0)))
        if len(Xe) < 6:
            raise ValueError(f"insufficient calibration points ({len(Xe)}) to fit predictor")
        Xe_a, Xt_a, Xm_a = np.vstack(Xe), np.vstack(Xt), np.vstack(Xm)
        ye_a, yt_a, ym_a = np.asarray(ye), np.asarray(yt), np.asarray(ym)
        w_err = _fit_ridge(Xe_a, ye_a, alpha=alpha)
        w_time = _fit_ridge(Xt_a, yt_a, alpha=alpha)
        w_mem = _fit_ridge(Xm_a, ym_a, alpha=alpha)
        stats = {
            "n_samples": len(Xe),
            "log10_l2_rmse": round(float(np.sqrt(np.mean((Xe_a @ w_err - ye_a) ** 2))), 4),
            "log10_time_rmse": round(float(np.sqrt(np.mean((Xt_a @ w_time - yt_a) ** 2))), 4),
            "log10_mem_rmse": round(float(np.sqrt(np.mean((Xm_a @ w_mem - ym_a) ** 2))), 4),
        }
        return cls(w_err, w_time, w_mem, stats)

    def predict(self, pf: ProblemFeatures, cand: StrategyCandidate, hw: HardwareFeatures) -> dict[str, float]:
        xe, xt, xm = _featurize_pair(pf, cand, hw)
        l2 = 10.0 ** float(np.dot(xe, self.w_err))
        sec = 10.0 ** float(np.dot(xt, self.w_time))
        rss = 10.0 ** float(np.dot(xm, self.w_mem))
        return {"pred_l2_error": l2, "pred_wall_seconds": sec, "pred_peak_rss_mb": rss}

    def to_dict(self) -> dict:
        return {
            "w_err": self.w_err.tolist(),
            "w_time": self.w_time.tolist(),
            "w_mem": self.w_mem.tolist(),
            "train_stats": self.train_stats,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerformancePredictor":
        return cls(np.asarray(data["w_err"]), np.asarray(data["w_time"]), np.asarray(data["w_mem"]), data["train_stats"])


def execute_candidate(
    task: AdaptiveTask,
    cand: StrategyCandidate,
    work_dir: Path,
    *,
    mms_bin: Path | None = None,
    amr_bin: Path | None = None,
) -> dict:
    """Execute one strategy candidate through REAL MFEM binary + REAL independent verifier."""
    mms_bin = mms_bin or mfem_runner.solver_binary()
    amr_bin = amr_bin or mfem_runner.amr_binary()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=False)
    (work_dir / "task.json").write_text(json.dumps({
        "task_id": task.task_id, "case_id": task.case_id, "split": task.split,
        "problem": task.problem.model_dump(mode="json"), "policy": task.policy.model_dump(mode="json"),
        "validation": task.validation.model_dump(mode="json"), "candidate": asdict(cand)
    }, sort_keys=True))
    t0 = time.perf_counter()

    if cand.mesh_topology == "amr":
        amr_out = work_dir / "amr_run"
        run_res = mfem_runner.run_amr(
            task.case_id, cand.mesh_type, cand.polynomial_order, amr_out,
            n=cand.mesh_levels[0], assembly=cand.assembly, solver=cand.solver,
            dirichlet=",".join(map(str, task.problem.dirichlet_attributes())) or "none",
            neumann=",".join(map(str, task.problem.neumann_attributes())) or "none",
            max_steps=cand.amr_max_steps, max_dofs=cand.amr_max_dofs,
            theta=0.6, timeout_s=task.policy.per_level_timeout_seconds, mem_mb=task.policy.per_level_memory_mb, binary=amr_bin,
        )
        ver = verify_amr_run(amr_out, task.problem, task.validation)
        (work_dir / "verdict.json").write_text(json.dumps(ver, allow_nan=False))
        wall = time.perf_counter() - t0
        last_step = ver["steps"][-1] if ver["steps"] else {}
        l2 = last_step.get("l2_error")
        accuracy_ok = (l2 is not None) and (task.validation.max_l2_error_finest is None or l2 <= task.validation.max_l2_error_finest)
        codes = list(ver["failure_codes"])
        if run_res["exit_code"] != 0:
            codes.append("SOLVER_EXIT")
        for step, result_file in zip(ver["steps"], sorted(amr_out.glob("step_*/result.json")), strict=True):
            native = json.loads(result_file.read_bytes())
            if (native["case"] != task.case_id or native["order"] != cand.polynomial_order
                    or native["assembly"] != cand.assembly or native["solver"] != cand.solver
                    or native["mesh_type"] != cand.mesh_type or not native["solver_converged"]
                    or native["dirichlet_attributes"] != task.problem.dirichlet_attributes()
                    or native["neumann_attributes"] != task.problem.neumann_attributes()):
                codes.append("CLAIM_SPEC_MISMATCH")
            if (step.get("num_true_dofs_inferred") or 0) > min(cand.amr_max_dofs, task.policy.max_dofs_per_level):
                codes.append("DOF_BUDGET")
        if not accuracy_ok:
            codes.append("ACCURACY_NOT_MET")
        accepted = ver["verdict"] == ACCEPTED and not codes
        rss_kib = run_res.get("summary", {}).get("peak_rss_kib") or 0
        return {
            "candidate_id": cand.candidate_id,
            "ok": accepted,
            "verdict": ACCEPTED if accepted else ("RETRY_ELIGIBLE" if codes == ["ACCURACY_NOT_MET"] else "REJECTED"),
            "failure_codes": sorted(set(codes)),
            "l2_error": l2,
            "h1_semi_error": last_step.get("h1_semi_error"),
            "finest_dofs": last_step.get("num_true_dofs_inferred") or cand.amr_max_dofs,
            "wall_seconds": round(wall, 4),
            "peak_rss_mb": round(max(float(rss_kib), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0, 2),
        }

    # Standard uniform/unstructured study via frozen verifier
    run_root = work_dir / "run"
    (run_root / "frozen").mkdir(parents=True, exist_ok=True)
    (run_root / "request").mkdir(parents=True, exist_ok=True)
    att = run_root / "attempts" / "attempt_01"
    att.mkdir(parents=True, exist_ok=True)

    mesh_path = None
    ext_spec = None
    if cand.mesh_topology == "unstructured":
        msh_bytes = generate_unstructured_msh(
            cand.mesh_type, nx=task.unstructured_nx, seed=task.unstructured_seed, graded_focus=(0.375, 0.625)
        )
        mesh_path = run_root / "frozen/mesh.msh"
        mesh_path.write_bytes(msh_bytes)
        ext_spec = {"file": "mesh.msh", "sha256": hashlib.sha256(msh_bytes).hexdigest()}

    ex = ExecutionSpec(
        mesh_type=cand.mesh_type, mesh_topology=cand.mesh_topology,
        polynomial_order=cand.polynomial_order, mesh_levels=list(cand.mesh_levels),
        assembly=cand.assembly, solver=cand.solver,
    )
    study = StudySpec.model_validate({
        "schema_version": "0.4",
        "problem": task.problem.model_dump(mode="json"),
        "execution": ex.model_dump(mode="json"),
        "validation": task.validation.model_dump(mode="json"),
        "policy": task.policy.model_dump(mode="json"),
        **({"external_mesh": ext_spec} if ext_spec else {}),
    })
    (run_root / "frozen/problem.json").write_text(study.problem.model_dump_json())
    (run_root / "frozen/validation.json").write_text(study.validation.model_dump_json())
    (run_root / "frozen/policy.json").write_text(study.policy.model_dump_json())
    (run_root / "request/study_spec.json").write_text(study.model_dump_json())
    (att / "execution.json").write_text(study.execution.model_dump_json())

    ext_grid = inspect_gmsh_unstructured(mesh_path.read_bytes(), study.problem, study.execution) if mesh_path else None
    max_rss_kib = 0.0
    finest_dofs = cand.estimated_finest_dofs
    for n in cand.mesh_levels:
        lvl_dir = att / "levels" / f"n{n:04d}"
        ref = ext_grid.refinements(n) if ext_grid else 0
        lr = mfem_runner.run_level(
            study.problem, study.execution, task.case_id, n, lvl_dir,
            timeout_s=task.policy.per_level_timeout_seconds,
            mem_mb=task.policy.per_level_memory_mb,
            binary=mms_bin, mesh_path=mesh_path, refinements=ref,
        )
        rjson = lvl_dir / "result.json"
        if rjson.is_file():
            rdata = json.loads(rjson.read_text())
            max_rss_kib = max(max_rss_kib, float(rdata.get("peak_rss_kib") or 0.0))
            finest_dofs = int(rdata.get("num_dofs") or finest_dofs)
        if lr.exit_code != 0:
            break

    ver = verify_attempt(att, require_run_records=True)
    (work_dir / "verdict.json").write_text(json.dumps(ver, allow_nan=False))
    wall = time.perf_counter() - t0
    last_lv = ver["levels"][-1] if ver["levels"] else {}
    return {
        "candidate_id": cand.candidate_id,
        "ok": ver["verdict"] == ACCEPTED,
        "verdict": ver["verdict"],
        "failure_codes": ver["failure_codes"],
        "l2_error": last_lv.get("l2_error"),
        "h1_semi_error": last_lv.get("h1_semi_error"),
        "finest_dofs": last_lv.get("num_dofs") or finest_dofs,
        "wall_seconds": round(wall, 4),
        "peak_rss_mb": round(max(max_rss_kib, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0, 2),
    }


def choose_expert_candidate(pf: ProblemFeatures, cands: list[StrategyCandidate]) -> StrategyCandidate:
    """Deterministic numerical-analyst rule baseline."""
    if pf.is_unstructured:
        want_p = 3 if pf.log10_target_l2 >= 4.2 else 2
        pool = [c for c in cands if c.polynomial_order == want_p and c.assembly == "full" and c.solver == "cg_gs"]
        return sorted(pool or cands, key=lambda c: c.estimated_finest_dofs)[0]

    # Steep localized fields benefit strongly from AMR when available
    if pf.localization_ratio >= 6.0:
        amr_pool = [c for c in cands if c.mesh_topology == "amr" and c.polynomial_order == 2]
        if amr_pool:
            return sorted(amr_pool, key=lambda c: c.amr_max_dofs)[0]

    if pf.log10_target_l2 >= 6.8:
        want_p = 4
    elif pf.log10_target_l2 >= 5.0:
        want_p = 3
    else:
        want_p = 2
    # Choose middle-resolution schedule for p=3/4 or finer for lower p
    pool = [c for c in cands if c.mesh_topology == "cartesian" and c.mesh_type == "quad"
            and c.polynomial_order == want_p and c.assembly == "full" and c.solver == "cg_gs"]
    if not pool:
        pool = cands
    pool_sorted = sorted(pool, key=lambda c: c.estimated_finest_dofs)
    idx = min(1, len(pool_sorted) - 1) if pf.log10_target_l2 >= 5.0 and len(pool_sorted) > 1 else 0
    return pool_sorted[idx]


def choose_predictive_candidate(
    pf: ProblemFeatures,
    hw: HardwareFeatures,
    cands: list[StrategyCandidate],
    predictor: PerformancePredictor,
    safety_margin: float = 0.55,
) -> tuple[StrategyCandidate, list[dict]]:
    """Select cheapest strategy predicted to meet both L2 accuracy target and memory cap."""
    scored = []
    mem_cap = 0.85 * hw.per_level_memory_mb
    for c in cands:
        pred = predictor.predict(pf, c, hw)
        scored.append({"cand": c, **pred})
    feasible = [
        row for row in scored
        if row["pred_l2_error"] <= safety_margin * pf.target_l2 and row["pred_peak_rss_mb"] <= mem_cap
    ]
    if not feasible:
        # Fallback: pick candidate with smallest predicted L2 within memory cap
        within_mem = [r for r in scored if r["pred_peak_rss_mb"] <= mem_cap] or scored
        chosen = min(within_mem, key=lambda r: (r["pred_l2_error"], r["pred_wall_seconds"]))
    else:
        chosen = min(feasible, key=lambda r: (r["pred_wall_seconds"], r["cand"].estimated_finest_dofs))
    top_preview = [
        {"candidate_id": r["cand"].candidate_id, "pred_l2": r["pred_l2_error"], "pred_s": round(r["pred_wall_seconds"], 3)}
        for r in sorted(scored, key=lambda r: r["pred_wall_seconds"])[:5]
    ]
    return chosen["cand"], top_preview


def run_strategy_on_task(
    strategy_name: Literal["fixed_baseline", "expert_rules", "predictive_model", "pilot_autotuner", "advisor_rules", "ai_assisted"],
    task: AdaptiveTask,
    work_dir: Path,
    *,
    predictor: PerformancePredictor | None = None,
    mms_bin: Path | None = None,
    amr_bin: Path | None = None,
    ollama_model: str | None = None,
    ollama_digest: str | None = None,
) -> dict:
    """Execute one complete strategy (decision + pilots + retries fully charged) on a task."""
    t_total0 = time.perf_counter()
    env = environment.probe()
    hw = extract_hardware_features(task.policy, env)
    pf = extract_problem_features(task.problem, task.validation, task.mesh_topology, case_id=task.case_id)
    cands, excluded = enumerate_candidates(task, hw, env, include_amr=True)

    t_dec0 = time.perf_counter()
    pilot_seconds = 0.0
    pilot_details = []
    ai_meta: dict[str, Any] = {}
    ordered_queue: list[StrategyCandidate] = []

    # A fixed, task-independent full-assembly ladder provides a verified comparator.
    default_pool = [c for c in cands if c.mesh_topology == task.mesh_topology
                    and c.mesh_type == task.base_mesh_type and c.assembly == "full" and c.solver == "cg_gs"]
    fixed = []
    for order, size in ((3, 32), (4, 32), (3, 64)):
        pool = [c for c in default_pool if c.polynomial_order == order]
        if pool:
            fixed.append(min(pool, key=lambda c: abs(c.mesh_levels[-1] - size)))
    if strategy_name == "fixed_baseline":
        ordered_queue = fixed or default_pool

    elif strategy_name == "expert_rules":
        first = choose_expert_candidate(pf, cands)
        rest = sorted([c for c in cands if c.candidate_id != first.candidate_id],
                      key=lambda c: (c.mesh_topology != "amr", -c.polynomial_order, c.estimated_finest_dofs))
        ordered_queue = [first] + rest[:2]

    elif strategy_name == "predictive_model":
        assert predictor is not None, "predictive_model requires a calibrated PerformancePredictor"
        first, preview = choose_predictive_candidate(pf, hw, cands, predictor)
        rest = sorted([c for c in cands if c.candidate_id != first.candidate_id],
                      key=lambda c: (predictor.predict(pf, c, hw)["pred_l2_error"],
                                     predictor.predict(pf, c, hw)["pred_wall_seconds"]))
        ordered_queue = [first] + rest[:2]
        ai_meta["predictor_preview"] = preview

    elif strategy_name == "pilot_autotuner":
        # Measure real coarse pilot (n0 or n=8) across 4 structural archetypes (Q1, Q2, Q3, Q4 or AMR)
        if task.mesh_topology == "unstructured":
            pilot_specs = [(task.base_mesh_type, p, "full", "cg_gs") for p in (1, 2, 3, 4)]
        else:
            pilot_specs = [("quad", 1, "full", "cg_gs"), ("quad", 2, "full", "cg_gs"),
                           ("quad", 3, "full", "cg_gs"), ("quad", 4, "partial", "cg_jacobi")]
        scored_pilots = []
        for idx, (mt, p, asm, sol) in enumerate(pilot_specs):
            n_pilot = task.unstructured_nx if task.mesh_topology == "unstructured" else 4
            pilot_sched = (n_pilot, 2 * n_pilot, 4 * n_pilot)
            p_task = AdaptiveTask(
                task_id=f"{task.task_id}_pilot_{idx}", split=task.split, case_id=task.case_id,
                problem=task.problem,
                validation=task.validation.model_copy(update={"max_l2_error_finest": None,
                                                            "require_monotone_decrease": False}),
                policy=task.policy, base_mesh_type=mt, mesh_topology=task.mesh_topology,
                unstructured_nx=task.unstructured_nx, unstructured_seed=task.unstructured_seed,
            )
            p_cand = StrategyCandidate(
                candidate_id=f"pilot-{mt}-p{p}", mesh_type=mt, mesh_topology=task.mesh_topology,
                polynomial_order=p, assembly=asm, solver=sol, mesh_levels=pilot_sched,
                estimated_finest_dofs=((pilot_sched[-1] * p + 1) ** 2),
            )
            res_p = execute_candidate(p_task, p_cand, work_dir / f"pilot_{idx}", mms_bin=mms_bin, amr_bin=amr_bin)
            pilot_seconds += res_p["wall_seconds"]
            pilot_details.append({"candidate": p_cand.candidate_id, "l2": res_p["l2_error"], "s": res_p["wall_seconds"]})
            if res_p["ok"] and res_p["l2_error"] and res_p["l2_error"] > 0:
                matching = [c for c in cands if c.mesh_type == mt and c.polynomial_order == p
                            and c.mesh_topology == task.mesh_topology
                            and c.assembly == asm and c.solver == sol]
                for c in matching:
                    ratio = max(c.mesh_levels[-1] / max(pilot_sched[-1], 1), 1.0)
                    proj_l2 = res_p["l2_error"] / (ratio ** (p + 1))
                    proj_s = res_p["wall_seconds"] * ((c.estimated_finest_dofs / max(res_p["finest_dofs"], 1)) ** 1.05)
                    scored_pilots.append((proj_l2 <= 0.65 * pf.target_l2, proj_s, proj_l2, c))
        feasible_pilots = [x for x in scored_pilots if x[0]]
        if feasible_pilots:
            best_c = min(feasible_pilots, key=lambda x: (x[1], x[2]))[3]
        elif scored_pilots:
            best_c = min(scored_pilots, key=lambda x: x[2])[3]
        else:
            best_c = cands[0]
        rest = sorted([c for c in cands if c.candidate_id != best_c.candidate_id],
                      key=lambda c: (-c.polynomial_order, c.estimated_finest_dofs))
        ordered_queue = [best_c] + rest[:2]

    elif strategy_name in ("ai_assisted", "advisor_rules"):
        ranked = []
        for candidate in cands:
            predicted = predictor.predict(pf, candidate, hw)
            if (predicted["pred_l2_error"] <= 0.55 * pf.target_l2 and
                    predicted["pred_peak_rss_mb"] <= 0.85 * hw.per_level_memory_mb):
                ranked.append((predicted["pred_wall_seconds"], candidate.candidate_id, candidate, predicted))
        ranked.sort(key=lambda row: row[:2])
        shortlist = ranked[:5]
        first = shortlist[0][2] if shortlist else choose_expert_candidate(pf, cands)
        ai_meta["used_live_llm"] = False
        if strategy_name == "ai_assisted" and ollama_model and ollama_digest and shortlist:
            previews = [{"candidate_id": item.candidate_id, **pred} for _, _, item, pred in shortlist]
            choice, audit = adaptive_advisor.choose(asdict(pf), asdict(hw), previews,
                                                    model=ollama_model, digest=ollama_digest)
            ai_meta.update(advisor=audit, used_live_llm=audit["used_live_llm"])
            if choice:
                first = next(item for _, _, item, _ in shortlist if item.candidate_id == choice)
        ordered_queue = [first]
    if ordered_queue:
        # On failed qualification, move to a different reviewed full-assembly discretization.
        ordered_queue = list(dict.fromkeys([ordered_queue[0], *fixed, *ordered_queue[1:]]))
    decision_seconds = (time.perf_counter() - t_dec0) - pilot_seconds

    attempts = []
    final_ok = False
    peak_rss_mb = 0.0
    final_l2 = None
    final_dofs = None
    max_tries = task.policy.max_attempts
    for idx, cand in enumerate(ordered_queue[:max_tries], start=1):
        res = execute_candidate(task, cand, work_dir / f"try_{idx:02d}_{cand.candidate_id}", mms_bin=mms_bin, amr_bin=amr_bin)
        attempts.append(res)
        peak_rss_mb = max(peak_rss_mb, res["peak_rss_mb"])
        if res["ok"]:
            final_ok = True
            final_l2 = res["l2_error"]
            final_dofs = res["finest_dofs"]
            break

    total_wall = time.perf_counter() - t_total0
    exec_seconds = sum(a["wall_seconds"] for a in attempts)
    return {
        "task_id": task.task_id,
        "split": task.split,
        "strategy": strategy_name,
        "accepted": final_ok,
        "selected_initial_candidate": ordered_queue[0].candidate_id if ordered_queue else None,
        "final_candidate": attempts[-1]["candidate_id"] if attempts else None,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "decision_seconds": round(max(decision_seconds, 0.0), 4),
        "pilot_seconds": round(pilot_seconds, 4),
        "pilot_details": pilot_details,
        "execution_seconds": round(exec_seconds, 4),
        "total_end_to_end_seconds": round(total_wall, 4),
        "finest_l2_error": final_l2,
        "target_l2": task.validation.max_l2_error_finest,
        "finest_dofs": final_dofs,
        "peak_rss_mb": round(peak_rss_mb, 2),
        "excluded_capabilities": excluded,
        "meta": ai_meta,
    }
