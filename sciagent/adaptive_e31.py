"""E3.1 Adaptive decision engine: observable PDE features, failure-risk predictor,
risk-constrained strategy search, and 5-way adaptive execution strategies.

Design guarantees:
- Strict split hygiene: calibration fits on disjoint problem IDs and formulas;
  no interior exact_solution leakage in extract_problem_features() — features use
  only observable PDE source f(x,y), diffusivity kappa(x,y), boundary trace
  along dOmega, domain geometry, and tolerance/topology metadata.
- Full verification & resource accounting: every candidate execution delegates to
  real MFEM solvers (poisson_mms / poisson_amr) + independent verifier
  (verify_attempt), recording solver, verifier, decision, pilot, and retry
  wall-clock costs separately alongside failure costs.
- Risk-aware selection: fits both runtime/accuracy/RSS ridge models and a
  logistic failure-probability classifier P(fail | x, cand, target) on
  calibration acceptance outcomes, enforcing risk <= max_risk during search.
- Honest naming: no fake LLM claims; rule+surrogate+risk search is named
  ``hybrid_guardrail_predictor``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from sciagent import capability, environment, mfem_runner
from sciagent.evidence import EvidenceStore, sha256_file
from sciagent.gmsh import ARCHIVED_MFEM_MESH, ARCHIVED_MESH, generate_unstructured_msh, inspect_gmsh_unstructured, inspect_mfem_mesh_file
from sciagent.spec import (
    ANISOTROPIC,
    LAYERED,
    ExecutionSpec,
    ExternalMeshSpec,
    ProblemSpec,
    RetryPolicy,
    StudySpec,
    ValidationSpec,
    canonical_json,
    sha256_of,
)
from sciagent.verification_process import verify_bounded
from verifier import expr
from verifier.verifier import ACCEPTED, RETRY_ELIGIBLE

STRATEGY_NAMES = (
    "fixed_baseline",
    "expert_rules",
    "predictive_model",
    "pilot_autotuner",
    "hybrid_guardrail_predictor",
)


@dataclass(frozen=True)
class AdaptiveTask:
    name: str
    case_id: str
    problem: ProblemSpec
    validation: ValidationSpec
    policy: RetryPolicy
    mesh_topology: Literal["cartesian", "unstructured", "amr"] = "cartesian"
    lshape_mesh_bytes: bytes | None = None
    unstructured_seed: int = 42


@dataclass(frozen=True)
class StrategyCandidate:
    mesh_type: Literal["tri", "quad"]
    mesh_topology: Literal["cartesian", "unstructured", "amr"]
    polynomial_order: int
    assembly: Literal["full", "partial"]
    solver: Literal["cg_gs", "cg_jacobi", "cg_none"]
    initial_n: int
    amr_max_steps: int = 8
    amr_max_dofs: int = 20_000
    amr_theta: float = 0.7

    def tag(self) -> str:
        if self.mesh_topology == "amr":
            return f"amr_{self.mesh_type}_p{self.polynomial_order}_{self.solver}_s{self.amr_max_steps}"
        return f"{self.mesh_topology}_{self.mesh_type}_p{self.polynomial_order}_{self.assembly}_{self.solver}_n{self.initial_n}"

    def mesh_levels(self) -> tuple[int, ...]:
        if self.mesh_topology == "amr":
            return (self.initial_n,)
        return (self.initial_n, 2 * self.initial_n, 4 * self.initial_n)

    def max_dofs_estimate(self) -> int:
        if self.mesh_topology == "amr":
            return self.amr_max_dofs
        fin = 4 * self.initial_n
        return (fin * self.polynomial_order + 1) ** 2


@dataclass(frozen=True)
class ProblemFeatures:
    source_l2: float
    source_max: float
    source_localization: float
    source_variation: float
    kappa_ratio: float
    kappa_log_range: float
    boundary_variation: float
    has_neumann: float
    has_reentrant_corner: float
    log_target_l2: float
    is_amr_allowed: float
    is_unstructured: float


@dataclass(frozen=True)
class HardwareFeatures:
    cpu_count: int
    mem_total_mb: float
    mem_available_mb: float
    has_mpi: bool
    has_gpu: bool
    max_dofs_per_level: int
    per_level_memory_mb: int


def _read_mem_available_mb(fallback_mb: float) -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return fallback_mb * 0.75


def extract_hardware_features(policy: RetryPolicy, env: dict | None = None) -> HardwareFeatures:
    env = env or environment.probe()
    mem_tot = float(env.get("mem_total_mb") or 4096.0)
    mem_avail = _read_mem_available_mb(mem_tot)
    cfg = env.get("mfem", {}).get("config", {})
    return HardwareFeatures(
        cpu_count=int(env.get("cpu_count") or 1),
        mem_total_mb=mem_tot,
        mem_available_mb=round(mem_avail, 1),
        has_mpi=bool(env.get("mpirun_path")) and bool(cfg.get("MFEM_USE_MPI")),
        has_gpu=bool(env.get("nvidia_smi_path")) and bool(cfg.get("MFEM_USE_CUDA")),
        max_dofs_per_level=policy.max_dofs_per_level,
        per_level_memory_mb=policy.per_level_memory_mb,
    )


def extract_problem_features(
    problem: ProblemSpec,
    validation: ValidationSpec,
    mesh_topology: Literal["cartesian", "unstructured", "amr"] = "cartesian",
    *,
    case_id: str = "",
) -> ProblemFeatures:
    """Compute strictly observable PDE/geometry features (ZERO interior exact_solution evaluations)."""
    f_fn = expr.compile_expr(problem.source)
    k_fn = expr.compile_expr(problem.kappa)
    u_fn = expr.compile_expr(problem.exact_solution)

    # Observable 2D domain sampling (avoiding L-shape cutout quadrant if lshape_2d)
    grid = np.linspace(0.05, 0.95, 25)
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    x_all, y_all = gx.ravel(), gy.ravel()
    if problem.domain == "lshape_2d":
        keep = ~((x_all > 0.49) & (y_all > 0.49))
        xs, ys = x_all[keep], y_all[keep]
    else:
        xs, ys = x_all, y_all

    f_vals = np.abs(np.asarray(f_fn(xs, ys), dtype=np.float64))
    if f_vals.ndim == 0:
        f_vals = np.full_like(xs, float(f_vals))
    f_l2 = float(np.sqrt(np.mean(f_vals ** 2)))
    f_max = float(np.max(f_vals))
    f_mean = max(float(np.mean(f_vals)), 1e-12)
    source_loc = min(f_max / f_mean, 500.0)
    # Discrete 2D gradient proxy of observable source term f
    h = 0.90 / 24.0
    if problem.domain == "lshape_2d":
        f_var = float(np.std(f_vals) / f_mean)
    else:
        F2 = np.asarray(f_fn(gx, gy), dtype=np.float64)
        if F2.ndim == 0:
            f_var = 0.0
        else:
            dfdx = np.diff(F2, axis=0) / h
            dfdy = np.diff(F2, axis=1) / h
            f_var = float(math.hypot(float(np.mean(np.abs(dfdx))), float(np.mean(np.abs(dfdy)))) / f_mean)

    if problem.equation == LAYERED:
        k_ratio = 10000.0
    elif problem.equation == ANISOTROPIC:
        k_ratio = 1000.0
    else:
        kv = np.asarray(k_fn(xs, ys), dtype=np.float64)
        k_ratio = float(np.max(kv) / max(float(np.min(kv)), 1e-12)) if kv.ndim > 0 else 1.0

    # Observable boundary data variation evaluated strictly ON dOmega (never interior Omega)
    t_b = np.linspace(0.0, 1.0, 33)
    zeros = np.zeros_like(t_b)
    ones = np.ones_like(t_b)
    if problem.domain == "lshape_2d":
        t_h = np.linspace(0.0, 0.5, 17)
        t_hi = np.linspace(0.5, 1.0, 17)
        bx = np.concatenate([t_b, np.full_like(t_h, 1.0), t_hi, np.full_like(t_hi, 0.5), t_h, zeros])
        by = np.concatenate([zeros, t_h, np.full_like(t_hi, 0.5), t_hi, np.full_like(t_h, 1.0), t_b])
    else:
        bx = np.concatenate([t_b, ones, t_b, zeros])
        by = np.concatenate([zeros, t_b, ones, t_b])
    ub = np.asarray(u_fn(bx, by), dtype=np.float64)
    bdr_var = float(np.std(ub)) if ub.ndim > 0 else 0.0

    has_neu = 1.0 if problem.neumann_attributes() else 0.0
    reentrant = 1.0 if problem.domain == "lshape_2d" else 0.0
    tgt = validation.max_l2_error_finest if validation.max_l2_error_finest is not None else 1e-3

    return ProblemFeatures(
        source_l2=round(math.log10(max(f_l2, 1e-6) + 1.0), 5),
        source_max=round(math.log10(max(f_max, 1e-6) + 1.0), 5),
        source_localization=round(source_loc, 4),
        source_variation=round(math.log10(max(f_var, 0.0) + 1.0), 5),
        kappa_ratio=round(k_ratio, 4),
        kappa_log_range=round(math.log10(max(k_ratio, 1.0)), 4),
        boundary_variation=round(math.log10(max(bdr_var, 0.0) + 1.0), 5),
        has_neumann=has_neu,
        has_reentrant_corner=reentrant,
        log_target_l2=round(math.log10(tgt), 4),
        is_amr_allowed=1.0 if mesh_topology != "unstructured" else 0.0,
        is_unstructured=1.0 if mesh_topology == "unstructured" else 0.0,
    )


def enumerate_candidates(
    task: AdaptiveTask,
    hw: HardwareFeatures,
    env: dict | None = None,
) -> tuple[list[StrategyCandidate], list[dict]]:
    """Build hardware- and capability-valid candidates with explicit exclusion log."""
    candidates: list[StrategyCandidate] = []
    excluded: list[dict] = []

    raw_pool: list[StrategyCandidate] = []
    if task.mesh_topology == "unstructured":
        for mt in ("tri", "quad"):
            for p, init_n in ((1, 8), (1, 16), (2, 4), (2, 8), (3, 4), (3, 8)):
                raw_pool.append(StrategyCandidate(mt, "unstructured", p, "full", "cg_gs", init_n))
                raw_pool.append(StrategyCandidate(mt, "unstructured", p, "full", "cg_jacobi", init_n))
    elif task.problem.domain == "lshape_2d":
        for steps, max_dofs in ((6, 4000), (8, 8000), (10, 16000), (12, 24000)):
            raw_pool.append(StrategyCandidate("quad", "amr", 1, "full", "cg_gs", 4, amr_max_steps=steps, amr_max_dofs=max_dofs))
            raw_pool.append(StrategyCandidate("quad", "amr", 2, "full", "cg_gs", 4, amr_max_steps=steps, amr_max_dofs=max_dofs))
            raw_pool.append(StrategyCandidate("quad", "amr", 3, "full", "cg_gs", 4, amr_max_steps=steps, amr_max_dofs=max_dofs))
    else:
        for p in (1, 2, 3, 4):
            for init_n in (4, 8, 16, 32):
                for mt in ("tri", "quad"):
                    raw_pool.append(StrategyCandidate(mt, "cartesian", p, "full", "cg_gs", init_n))
                    if p >= 2 and mt == "quad":
                        raw_pool.append(StrategyCandidate(mt, "cartesian", p, "partial", "cg_jacobi", init_n))
        for p in (1, 2, 3):
            for steps, max_dofs in ((6, 6000), (8, 12000), (10, 20000)):
                raw_pool.append(StrategyCandidate("quad", "amr", p, "full", "cg_gs", 4, amr_max_steps=steps, amr_max_dofs=max_dofs))
                if p <= 2:
                    raw_pool.append(StrategyCandidate("tri", "amr", p, "full", "cg_gs", 4, amr_max_steps=steps, amr_max_dofs=max_dofs))

    seen = set()
    for c in raw_pool:
        if c.tag() in seen:
            continue
        seen.add(c.tag())
        est_dofs = c.max_dofs_estimate()
        if est_dofs > hw.max_dofs_per_level:
            excluded.append({"candidate": c.tag(), "reason": f"est_dofs={est_dofs} > cap={hw.max_dofs_per_level}"})
            continue
        if task.validation.max_l2_error_finest is not None and c.mesh_topology != "amr" and c.polynomial_order == 1 and 4 * c.initial_n < 16:
            excluded.append({"candidate": c.tag(), "reason": "coarse P1 mesh"})
            continue

        ms = "amr" if c.mesh_topology == "amr" else ("gmsh22" if c.mesh_topology == "unstructured" else "cartesian")
        ex = ExecutionSpec(
            mesh_type=c.mesh_type,
            mesh_topology=c.mesh_topology,
            polynomial_order=c.polynomial_order,
            mesh_levels=c.mesh_levels(),
            assembly=c.assembly,
            solver=c.solver,
            amr_max_steps=c.amr_max_steps,
            amr_max_dofs=c.amr_max_dofs,
            amr_theta=c.amr_theta,
        )
        dec = capability.evaluate(
            ex, env, task.problem.equation,
            exact_solution=task.problem.exact_solution, case_id=task.case_id,
            mesh_source=ms, e3_verified=True,
        )
        if dec.status != "tested":
            excluded.append({"candidate": c.tag(), "reason": f"capability:{dec.code}"})
            continue
        candidates.append(c)
    return candidates, excluded


def _featurize(pf: ProblemFeatures, hw: HardwareFeatures, cand: StrategyCandidate) -> np.ndarray:
    p = float(cand.polynomial_order)
    log_dof = math.log10(max(cand.max_dofs_estimate(), 10))
    is_quad = 1.0 if cand.mesh_type == "quad" else 0.0
    is_partial = 1.0 if cand.assembly == "partial" else 0.0
    is_amr = 1.0 if cand.mesh_topology == "amr" else 0.0
    is_unstruct = 1.0 if cand.mesh_topology == "unstructured" else 0.0
    # Smoothness-aligned convergence slope interaction terms
    eff_order = p * (1.0 - 0.35 * pf.has_reentrant_corner * (1.0 - is_amr))
    loc_amr = pf.source_localization * is_amr
    loc_unif = pf.source_localization * (1.0 - is_amr)
    return np.array([
        1.0,
        log_dof,
        p,
        p * log_dof,
        eff_order * log_dof,
        is_quad,
        is_partial,
        is_amr,
        is_unstruct,
        pf.source_l2,
        pf.source_localization,
        pf.source_variation,
        pf.kappa_log_range,
        pf.boundary_variation,
        pf.has_neumann,
        pf.has_reentrant_corner,
        loc_amr,
        loc_unif,
        pf.has_reentrant_corner * is_amr,
        math.log10(max(hw.mem_available_mb, 256.0)),
    ], dtype=np.float64)


class PerformancePredictor:
    """Multi-target ridge surrogate + calibrated logistic failure-probability model."""

    def __init__(
        self,
        w_l2: np.ndarray,
        w_time: np.ndarray,
        w_rss: np.ndarray,
        w_risk: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
        l2_rmse: float,
        n_train: int,
    ):
        self.w_l2 = w_l2
        self.w_time = w_time
        self.w_rss = w_rss
        self.w_risk = w_risk
        self.mean = mean
        self.scale = scale
        self.l2_rmse = max(float(l2_rmse), 0.15)
        self.n_train = n_train

    @classmethod
    def fit(cls, records: list[dict], ridge: float = 1e-2) -> "PerformancePredictor":
        ok_recs = [r for r in records if r.get("l2_error") is not None and r.get("wall_seconds") is not None]
        if len(ok_recs) < 5:
            raise ValueError(f"need >= 5 valid calibration observations to fit predictor, got {len(ok_recs)}")

        X = np.vstack([
            _featurize(ProblemFeatures(**r["problem_features"]), HardwareFeatures(**r["hw_features"]), StrategyCandidate(**r["candidate"]))
            for r in ok_recs
        ])
        y_l2 = np.array([math.log10(max(float(r["l2_error"]), 1e-15)) for r in ok_recs], dtype=np.float64)
        y_time = np.array([math.log10(max(float(r["wall_seconds"]), 1e-4)) for r in ok_recs], dtype=np.float64)
        y_rss = np.array([math.log10(max(float(r.get("peak_rss_mb") or 64.0), 1.0)) for r in ok_recs], dtype=np.float64)

        mean = X[:, 1:].mean(axis=0)
        scale = np.where(X[:, 1:].std(axis=0) > 1e-8, X[:, 1:].std(axis=0), 1.0)
        Xs = np.column_stack([np.ones(X.shape[0]), (X[:, 1:] - mean) / scale])
        reg = np.eye(Xs.shape[1]) * ridge
        reg[0, 0] = 0.0
        XtX_inv = np.linalg.inv(Xs.T @ Xs + reg)
        w_l2 = XtX_inv @ (Xs.T @ y_l2)
        w_time = XtX_inv @ (Xs.T @ y_time)
        w_rss = XtX_inv @ (Xs.T @ y_rss)
        l2_resid = y_l2 - Xs @ w_l2
        l2_rmse = float(np.sqrt(np.mean(l2_resid ** 2)))

        # Logistic failure classifier on standardized/bounded risk margin features (zero collinearity)
        risk_feats = []
        risk_labels = []
        for idx, r in enumerate(ok_recs):
            tgt = r.get("target_l2")
            pred_l2_idx = float((Xs[idx] @ w_l2))
            if tgt is not None:
                z_acc = (pred_l2_idx - math.log10(float(tgt))) / max(l2_rmse, 0.2)
            else:
                z_acc = -2.0
            z_acc = max(-5.0, min(5.0, z_acc))
            cand_d = r["candidate"]
            hw_d = r["hw_features"]
            rss_log_ratio = float((Xs[idx] @ w_rss)) - math.log10(max(float(hw_d["per_level_memory_mb"]), 256.0))
            rss_pressure = max(0.0, rss_log_ratio + 0.30)  # activates only above 50% of per-level RAM cap
            is_amr_f = 1.0 if cand_d["mesh_topology"] == "amr" else 0.0
            unif_loc_risk = (1.0 - is_amr_f) * min(float(r["problem_features"]["source_localization"]) / 20.0, 2.0)
            coarse_ho_risk = 1.0 if (not is_amr_f and (cand_d["initial_n"] <= 4 or (cand_d["polynomial_order"] >= 4 and cand_d["initial_n"] <= 8))) else 0.0
            fail = 0.0 if (r.get("verdict") == ACCEPTED and (tgt is None or float(r["l2_error"]) <= float(tgt))) else 1.0
            risk_feats.append([
                1.0,
                z_acc,
                max(z_acc, 0.0),
                rss_pressure,
                is_amr_f,
                unif_loc_risk,
                coarse_ho_risk,
            ])
            risk_labels.append(fail)

        Xr = np.asarray(risk_feats, dtype=np.float64)
        yr = np.asarray(risk_labels, dtype=np.float64)
        # Prior-anchored regularized IRLS logistic fit preserving smooth probability calibration
        w_risk = np.array([-1.6, 1.5, 1.2, 4.0, -0.5, 1.4, 1.2], dtype=np.float64)
        if len(np.unique(yr)) >= 2:
            w_cur = w_risk.copy()
            reg_r = np.diag([0.8, 0.5, 0.5, 1.0, 0.8, 0.8, 0.8])
            w_prior = w_risk.copy()
            for _ in range(25):
                logits = np.clip(Xr @ w_cur, -12.0, 12.0)
                probs = 1.0 / (1.0 + np.exp(-logits))
                W = np.maximum(probs * (1.0 - probs), 1e-5)
                grad = Xr.T @ (probs - yr) + reg_r @ (w_cur - w_prior)
                H = (Xr * W[:, None]).T @ Xr + reg_r
                try:
                    step = np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    break
                w_cur -= 0.5 * step
                if float(np.max(np.abs(step))) < 1e-6:
                    break
            w_cur[1] = max(float(w_cur[1]), 0.8)
            w_cur[2] = max(float(w_cur[2]), 0.5)
            w_cur[3] = max(float(w_cur[3]), 2.0)
            w_cur[5] = max(float(w_cur[5]), 0.5)
            w_cur[6] = max(float(w_cur[6]), 0.5)
            w_risk = w_cur

        return cls(w_l2, w_time, w_rss, w_risk, mean, scale, l2_rmse, len(ok_recs))

    def predict(self, pf: ProblemFeatures, hw: HardwareFeatures, cand: StrategyCandidate) -> dict[str, float]:
        x = _featurize(pf, hw, cand)
        xs = np.concatenate([[1.0], (x[1:] - self.mean) / self.scale])
        log_l2 = float(xs @ self.w_l2)
        log_t = float(xs @ self.w_time)
        log_r = float(xs @ self.w_rss)
        pred_rss = min(10.0 ** log_r, float(hw.per_level_memory_mb) * 2.0)

        z_acc = max(-5.0, min(5.0, (log_l2 - pf.log_target_l2) / self.l2_rmse))
        rss_log_ratio = log_r - math.log10(max(float(hw.per_level_memory_mb), 256.0))
        rss_pressure = max(0.0, rss_log_ratio + 0.30)
        is_amr_f = 1.0 if cand.mesh_topology == "amr" else 0.0
        unif_loc_risk = (1.0 - is_amr_f) * min(pf.source_localization / 20.0, 2.0)
        coarse_ho_risk = 1.0 if (not is_amr_f and (cand.initial_n <= 4 or (cand.polynomial_order >= 4 and cand.initial_n <= 8))) else 0.0
        rx = np.array([
            1.0,
            z_acc,
            max(z_acc, 0.0),
            rss_pressure,
            is_amr_f,
            unif_loc_risk,
            coarse_ho_risk,
        ], dtype=np.float64)
        logit = float(np.clip(rx @ self.w_risk, -12.0, 12.0))
        p_fail = 1.0 / (1.0 + math.exp(-logit))

        return {
            "pred_log10_l2": log_l2,
            "pred_l2": 10.0 ** log_l2,
            "pred_wall_seconds": 10.0 ** log_t,
            "pred_peak_rss_mb": pred_rss,
            "pred_failure_prob": round(p_fail, 5),
        }


def _materialise_external_mesh(task: AdaptiveTask, cand: StrategyCandidate, run_root: Path) -> ExternalMeshSpec | None:
    if task.mesh_topology == "unstructured":
        mbytes = generate_unstructured_msh(
            cand.mesh_type, nx=cand.initial_n, seed=task.unstructured_seed, graded_focus=(0.375, 0.625),
        )
        path = run_root / f"auto_{cand.mesh_type}_{cand.initial_n}.msh"
        path.write_bytes(mbytes)
        return ExternalMeshSpec(file=path.name, sha256=hashlib.sha256(mbytes).hexdigest())
    if task.problem.domain == "lshape_2d":
        if task.lshape_mesh_bytes is None:
            raise ValueError("lshape_2d adaptive task missing lshape_mesh_bytes")
        path = run_root / "l-shape.mesh"
        path.write_bytes(task.lshape_mesh_bytes)
        return ExternalMeshSpec(file=path.name, sha256=hashlib.sha256(task.lshape_mesh_bytes).hexdigest())
    return None


def execute_candidate(
    task: AdaptiveTask,
    cand: StrategyCandidate,
    work_dir: Path,
    mms_bin: Path,
    amr_bin: Path,
    *,
    tag: str = "run",
) -> dict:
    """Execute one strategy candidate through real MFEM + full independent verifier."""
    t_total0 = time.perf_counter()
    run_root = work_dir / f"{tag}_{cand.tag()}"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    ext = _materialise_external_mesh(task, cand, run_root)
    if ext is not None and cand.mesh_topology == "unstructured":
        # Check mesh bytes valid once; run_root/auto_*.msh is alongside spec
        inspect_gmsh_unstructured((run_root / ext.file).read_bytes(), task.problem, ExecutionSpec(
            mesh_type=cand.mesh_type, mesh_topology="unstructured",
            polynomial_order=cand.polynomial_order, mesh_levels=cand.mesh_levels(),
        ))

    ex = ExecutionSpec(
        mesh_type=cand.mesh_type,
        mesh_topology=cand.mesh_topology,
        polynomial_order=cand.polynomial_order,
        mesh_levels=cand.mesh_levels(),
        assembly=cand.assembly,
        solver=cand.solver,
        amr_max_steps=cand.amr_max_steps,
        amr_max_dofs=cand.amr_max_dofs,
        amr_theta=cand.amr_theta,
    )
    study = StudySpec(
        schema_version="0.4",
        problem=task.problem,
        execution=ex,
        validation=task.validation,
        policy=task.policy,
        external_mesh=ext,
    )

    store = EvidenceStore(run_root / "run")
    store.write_bytes_once("request/study_spec.json", (canonical_json(study) + "\n").encode("utf-8"))
    store.write_once("frozen/problem.json", canonical_json(study.problem))
    store.write_once("frozen/validation.json", canonical_json(study.validation))
    store.write_once("frozen/policy.json", canonical_json(study.policy))
    if ext is not None:
        mrel = ARCHIVED_MFEM_MESH if ext.file.endswith(".mesh") else ARCHIVED_MESH
        mbytes = (run_root / ext.file).read_bytes()
        if mrel == ARCHIVED_MFEM_MESH:
            inspect_mfem_mesh_file(mbytes, task.problem, ex)
        store.write_bytes_once(mrel, mbytes)
        archived_mesh_path = store.path(mrel)
    else:
        archived_mesh_path = None

    att = store.path("attempts/attempt_01")
    store.write_once("attempts/attempt_01/execution.json", canonical_json(ex))

    t_solve0 = time.perf_counter()
    if cand.mesh_topology == "amr":
        dir_str = ",".join(map(str, task.problem.dirichlet_attributes())) or "none"
        neu_str = ",".join(map(str, task.problem.neumann_attributes())) or "none"
        rr = mfem_runner.run_amr(
            task.case_id, cand.mesh_type, cand.polynomial_order, att / "amr",
            n=cand.initial_n, assembly=cand.assembly, solver=cand.solver,
            dirichlet=dir_str, neumann=neu_str, mesh_path=archived_mesh_path,
            max_steps=cand.amr_max_steps, max_dofs=cand.amr_max_dofs, theta=cand.amr_theta,
            timeout_s=task.policy.per_level_timeout_seconds, mem_mb=task.policy.per_level_memory_mb,
            binary=amr_bin,
        )
        solver_ok = rr["exit_code"] == 0
    else:
        solver_ok = True
        for r_idx, n in enumerate(cand.mesh_levels()):
            ref = r_idx if cand.mesh_topology == "unstructured" else 0
            lr = mfem_runner.run_level(
                task.problem, ex, task.case_id, n, att / f"levels/n{n:04d}",
                timeout_s=task.policy.per_level_timeout_seconds, mem_mb=task.policy.per_level_memory_mb,
                binary=mms_bin, mesh_path=archived_mesh_path, refinements=ref,
            )
            if lr.exit_code != 0:
                solver_ok = False
                break
    solver_wall = time.perf_counter() - t_solve0

    t_ver0 = time.perf_counter()
    ver, ver_meta = verify_bounded(att, task.policy, remaining=task.policy.max_total_wall_seconds)
    verifier_wall = time.perf_counter() - t_ver0
    total_wall = time.perf_counter() - t_total0

    rss_kib_candidates = []
    if isinstance((ver_meta or {}).get("peak_rss_kib"), int):
        rss_kib_candidates.append(ver_meta["peak_rss_kib"])
    amr_sum_path = att / "amr/amr_summary.json"
    if amr_sum_path.is_file():
        amr_kib = json.loads(amr_sum_path.read_text()).get("peak_rss_kib")
        if isinstance(amr_kib, int):
            rss_kib_candidates.append(amr_kib)
    for rpath in (att / "levels").glob("n*/result.json") if (att / "levels").is_dir() else ():
        rkib = json.loads(rpath.read_text()).get("peak_rss_kib")
        if isinstance(rkib, int):
            rss_kib_candidates.append(rkib)
    measured_rss_mb = round(max(rss_kib_candidates) / 1024.0, 2) if rss_kib_candidates else None

    lvls = ver.get("levels", [])
    finest = lvls[-1] if lvls else {}
    return {
        "candidate": cand,
        "verdict": ver["verdict"],
        "failure_codes": ver.get("failure_codes", []),
        "solver_ok": solver_ok,
        "l2_error": finest.get("l2_error"),
        "h1_semi_error": finest.get("h1_semi_error"),
        "reference_rel_diff": finest.get("reference_rel_diff"),
        "finest_dofs": finest.get("num_dofs") or cand.max_dofs_estimate(),
        "wall_seconds": round(total_wall, 5),
        "solver_seconds": round(solver_wall, 5),
        "verifier_seconds": round(verifier_wall, 5),
        "peak_rss_mb": measured_rss_mb,
        "steps_or_levels": len(lvls),
    }


def _rank_with_risk_guardrail(
    scored: list[tuple[float, float, StrategyCandidate, dict]],
    *,
    max_risk: float,
    margin_factor: float,
    target_l2: float | None,
    mem_cap_mb: float,
) -> tuple[StrategyCandidate, dict]:
    """Select fastest candidate satisfying risk constraint P(fail) <= max_risk and accuracy/memory guards."""
    safe = [
        item for item in scored
        if item[1] <= max_risk
        and item[3]["pred_peak_rss_mb"] <= 0.85 * mem_cap_mb
        and (target_l2 is None or item[3]["pred_l2"] <= target_l2 * margin_factor)
    ]
    if safe:
        safe.sort(key=lambda it: (it[0], it[1]))
        return safe[0][2], {**safe[0][3], "selected_tier": "risk_feasible"}

    # Fallback tier 1: relax margin factor only, still enforce max_risk
    risk_only = [item for item in scored if item[1] <= max_risk and item[3]["pred_peak_rss_mb"] <= mem_cap_mb]
    if risk_only:
        risk_only.sort(key=lambda it: (it[0], it[1]))
        return risk_only[0][2], {**risk_only[0][3], "selected_tier": "risk_cap_only"}

    # Fallback tier 2: minimize unrounded failure probability first, then predicted error & runtime
    scored.sort(key=lambda it: (it[1], it[3]["pred_log10_l2"], it[0]))
    return scored[0][2], {**scored[0][3], "selected_tier": "min_risk_fallback"}


def choose_predictive_candidate(
    pf: ProblemFeatures,
    hw: HardwareFeatures,
    candidates: list[StrategyCandidate],
    predictor: PerformancePredictor,
    *,
    max_risk: float = 0.25,
    margin_factor: float = 0.85,
) -> tuple[StrategyCandidate, dict]:
    target = 10.0 ** pf.log_target_l2 if pf.log_target_l2 < -0.1 else None
    scored = []
    for c in candidates:
        pr = predictor.predict(pf, hw, c)
        expected_cost = pr["pred_wall_seconds"] * (1.0 + 1.5 * pr["pred_failure_prob"])
        scored.append((expected_cost, pr["pred_failure_prob"], c, pr))
    return _rank_with_risk_guardrail(
        scored, max_risk=max_risk, margin_factor=margin_factor,
        target_l2=target, mem_cap_mb=float(hw.per_level_memory_mb),
    )


def choose_expert_rule_candidate(
    task: AdaptiveTask,
    pf: ProblemFeatures,
    candidates: list[StrategyCandidate],
) -> tuple[StrategyCandidate, str]:
    """Strong hand-crafted expert heuristics from PDE structure & tolerance."""
    if task.mesh_topology == "unstructured":
        prefer = ("quad", "unstructured", 3, 4) if (pf.log_target_l2 <= -3.8) else ("tri", "unstructured", 2, 8)
        for mt, top, p, n0 in (prefer, ("tri", "unstructured", 3, 4), ("quad", "unstructured", 2, 8)):
            for c in candidates:
                if (c.mesh_type, c.mesh_topology, c.polynomial_order, c.initial_n) == (mt, top, p, n0) and c.solver == "cg_gs":
                    return c, f"unstructured_expert:{mt}_P{p}_n{n0}"
    if pf.has_reentrant_corner > 0.5:
        for c in sorted([x for x in candidates if x.mesh_topology == "amr"], key=lambda x: (abs(x.polynomial_order - 2), -x.amr_max_steps)):
            return c, "lshape_singularity_amr_q2"
    if pf.source_localization >= 4.0 and any(c.mesh_topology == "amr" for c in candidates):
        amr_c = [c for c in candidates if c.mesh_topology == "amr" and c.mesh_type == "quad" and c.polynomial_order == 2]
        if amr_c:
            amr_c.sort(key=lambda c: c.amr_max_steps)
            pick = amr_c[min(1, len(amr_c) - 1)]
            return pick, "localized_source_amr_q2"
    if pf.log_target_l2 <= -3.7:
        for c in candidates:
            if (c.mesh_topology, c.mesh_type, c.polynomial_order, c.assembly, c.initial_n) == ("cartesian", "quad", 3, "full", 8):
                return c, "tight_tol_quad_q3_n8"
    for c in candidates:
        if (c.mesh_topology, c.mesh_type, c.polynomial_order, c.assembly, c.initial_n) == ("cartesian", "quad", 2, "full", 8):
            return c, "balanced_quad_q2_n8"
    return candidates[0], "fallback_first"


def choose_hybrid_guardrail_candidate(
    task: AdaptiveTask,
    pf: ProblemFeatures,
    hw: HardwareFeatures,
    candidates: list[StrategyCandidate],
    predictor: PerformancePredictor,
    *,
    max_risk: float = 0.15,
) -> tuple[StrategyCandidate, dict]:
    """Hybrid structural guardrails + calibrated runtime/RSS/failure-risk search (no fake LLM)."""
    target = 10.0 ** pf.log_target_l2 if pf.log_target_l2 < -0.1 else None
    scored = []
    for c in candidates:
        pr = predictor.predict(pf, hw, c)
        guardrail_penalty = 1.0
        # Physics guardrail 1: singular L-shape needs AMR rather than high-p uniform
        if pf.has_reentrant_corner > 0.5 and c.mesh_topology != "amr":
            guardrail_penalty *= 5.0
        # Physics guardrail 2: avoid under-resolved P1 when tight target (both uniform and AMR)
        if target is not None and target <= 5e-4 and c.polynomial_order == 1:
            guardrail_penalty *= 3.5
        if pf.has_reentrant_corner > 0.5 and c.mesh_topology == "amr" and target is not None and target <= 3e-4 and c.polynomial_order >= 3:
            guardrail_penalty *= 0.70
        # Physics guardrail 3: favor AMR on steep localized sources & warn on coarse uniform bump aliasing
        if pf.source_localization >= 5.0 and c.mesh_topology == "amr" and c.mesh_type == "quad" and c.polynomial_order in (2, 3):
            guardrail_penalty *= 0.80
        if pf.source_localization >= 10.0 and c.mesh_topology != "amr" and (c.initial_n < 16 or c.polynomial_order >= 4):
            guardrail_penalty *= 3.0
        # Physics guardrail 4: prefer Q3/Q4 on smooth tight-target Cartesian problems
        if pf.has_reentrant_corner < 0.5 and pf.source_localization < 3.5 and target is not None and target <= 2e-4 and c.polynomial_order == 3:
            guardrail_penalty *= 0.85

        eff_risk = min(0.99, pr["pred_failure_prob"] * (1.25 if guardrail_penalty > 1.5 else 1.0))
        cost = pr["pred_wall_seconds"] * guardrail_penalty * (1.0 + 2.0 * eff_risk)
        scored.append((cost, eff_risk, c, {**pr, "guardrail_penalty": round(guardrail_penalty, 3)}))

    return _rank_with_risk_guardrail(
        scored, max_risk=max_risk, margin_factor=0.80,
        target_l2=target, mem_cap_mb=float(hw.per_level_memory_mb),
    )


def _pilot_probe(
    task: AdaptiveTask,
    candidates: list[StrategyCandidate],
    work_dir: Path,
    mms_bin: Path,
    amr_bin: Path,
) -> tuple[StrategyCandidate, float, list[dict]]:
    """Cheap real pilot probing diverse families, accounting 100% of pilot runtime."""
    probes: list[StrategyCandidate] = []
    if task.mesh_topology == "unstructured":
        for mt, p in (("tri", 2), ("quad", 3)):
            m = [c for c in candidates if c.mesh_type == mt and c.polynomial_order == p and c.solver == "cg_gs"]
            if m:
                probes.append(min(m, key=lambda x: x.initial_n))
    elif task.problem.domain == "lshape_2d":
        for p in (2, 3):
            m = [c for c in candidates if c.mesh_topology == "amr" and c.polynomial_order == p]
            if m:
                probes.append(min(m, key=lambda x: x.amr_max_steps))
    else:
        for top, mt, p in (("cartesian", "quad", 2), ("cartesian", "quad", 3), ("amr", "quad", 2), ("amr", "quad", 3)):
            m = [c for c in candidates if c.mesh_topology == top and c.mesh_type == mt and c.polynomial_order == p and c.solver == "cg_gs"]
            if m:
                probes.append(min(m, key=lambda x: (x.initial_n, x.amr_max_steps)))

    if not probes:
        probes = candidates[:2]

    t_pilot0 = time.perf_counter()
    probe_logs = []
    target = task.validation.max_l2_error_finest or 1e-3
    best_cand = None
    best_proj_cost = float("inf")

    for p_cand in probes:
        if p_cand.mesh_topology == "amr":
            pilot_c = StrategyCandidate(
                p_cand.mesh_type, "amr", p_cand.polynomial_order, p_cand.assembly, p_cand.solver,
                p_cand.initial_n, amr_max_steps=min(5, p_cand.amr_max_steps), amr_max_dofs=min(4000, p_cand.amr_max_dofs),
            )
            res = execute_candidate(task, pilot_c, work_dir, mms_bin, amr_bin, tag="pilot")
            err = res["l2_error"] or 1.0
            ratio = err / target
            if ratio <= 1.0:
                proj_steps = pilot_c.amr_max_steps
            else:
                proj_steps = min(12, pilot_c.amr_max_steps + max(2, math.ceil(math.log(ratio, 2.5) * 2)))
            family = [c for c in candidates if c.mesh_topology == "amr" and c.mesh_type == p_cand.mesh_type and c.polynomial_order == p_cand.polynomial_order]
            chosen_f = min([c for c in family if c.amr_max_steps >= proj_steps] or family, key=lambda x: x.amr_max_steps)
            proj_cost = res["wall_seconds"] * (1.7 ** max(0, chosen_f.amr_max_steps - pilot_c.amr_max_steps))
            if res["verdict"] == "REJECTED":
                proj_cost *= 50.0
        else:
            pilot_c = StrategyCandidate(
                p_cand.mesh_type, p_cand.mesh_topology, p_cand.polynomial_order, p_cand.assembly, p_cand.solver,
                max(8, min(8, p_cand.initial_n)),
            )
            res = execute_candidate(task, pilot_c, work_dir, mms_bin, amr_bin, tag="pilot")
            err = res["l2_error"] or 1.0
            p_ord = p_cand.polynomial_order + 1
            factor = max(1.0, (err / (0.8 * target)) ** (1.0 / p_ord))
            needed_n0 = max(pilot_c.initial_n, int(math.ceil(pilot_c.initial_n * factor)))
            family = [c for c in candidates if c.mesh_topology == p_cand.mesh_topology and c.mesh_type == p_cand.mesh_type and c.polynomial_order == p_cand.polynomial_order and c.solver == p_cand.solver]
            feasible = [c for c in family if c.initial_n >= needed_n0] or family
            chosen_f = min(feasible, key=lambda x: x.initial_n)
            proj_cost = res["wall_seconds"] * ((chosen_f.initial_n / max(pilot_c.initial_n, 1)) ** 2)
            # If the real coarse pilot failed structural/self-report checks (e.g. localized bump aliasing), penalize!
            if res["verdict"] == "REJECTED":
                proj_cost *= 50.0

        probe_logs.append({"pilot": pilot_c.tag(), "verdict": res["verdict"], "l2": res["l2_error"], "wall_s": res["wall_seconds"], "projected": chosen_f.tag()})
        if proj_cost < best_proj_cost:
            best_proj_cost = proj_cost
            best_cand = chosen_f

    pilot_wall = time.perf_counter() - t_pilot0
    return best_cand or candidates[0], pilot_wall, probe_logs


def _fixed_baseline_candidate(task: AdaptiveTask, candidates: list[StrategyCandidate]) -> StrategyCandidate:
    """Strong non-adaptive fixed baseline (Q2 quad full CG+GS, not intentionally weak Q1)."""
    if task.mesh_topology == "unstructured":
        pool = [c for c in candidates if c.mesh_type == "tri" and c.polynomial_order == 2 and c.solver == "cg_gs"]
        if pool:
            return min(pool, key=lambda c: c.initial_n)
    if task.problem.domain == "lshape_2d":
        pool = [c for c in candidates if c.mesh_topology == "amr" and c.polynomial_order == 2]
        if pool:
            return min(pool, key=lambda c: c.amr_max_steps)
    # Fixed Cartesian Q2 full CG+GS at initial_n=16 (levels 16,32,64)
    for n0 in (16, 8, 32):
        for c in candidates:
            if (c.mesh_topology, c.mesh_type, c.polynomial_order, c.assembly, c.solver, c.initial_n) == ("cartesian", "quad", 2, "full", "cg_gs", n0):
                return c
    return candidates[0]


def run_strategy_on_task(
    strategy: str,
    task: AdaptiveTask,
    predictor: PerformancePredictor | None,
    hw: HardwareFeatures,
    env: dict,
    work_dir: Path,
    mms_bin: Path,
    amr_bin: Path,
) -> dict:
    """Execute one named strategy with full decision, pilot, solver, verifier, and retry accounting."""
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGY_NAMES}")

    t_dec0 = time.perf_counter()
    pf = extract_problem_features(task.problem, task.validation, task.mesh_topology, case_id=task.case_id)
    candidates, excluded = enumerate_candidates(task, hw, env)
    if not candidates:
        raise RuntimeError(f"no valid candidates for task {task.name}: {excluded[:3]}")

    pilot_wall = 0.0
    pred_info: dict = {}
    if strategy == "fixed_baseline":
        chosen = _fixed_baseline_candidate(task, candidates)
        reason = "strong_fixed_q2"
    elif strategy == "expert_rules":
        chosen, reason = choose_expert_rule_candidate(task, pf, candidates)
    elif strategy == "predictive_model":
        if predictor is None:
            raise ValueError("predictive_model requires fitted PerformancePredictor")
        chosen, pred_info = choose_predictive_candidate(pf, hw, candidates, predictor)
        reason = pred_info.get("selected_tier", "risk_predictive")
    elif strategy == "hybrid_guardrail_predictor":
        if predictor is None:
            raise ValueError("hybrid_guardrail_predictor requires fitted PerformancePredictor")
        chosen, pred_info = choose_hybrid_guardrail_candidate(task, pf, hw, candidates, predictor)
        reason = pred_info.get("selected_tier", "hybrid_guardrail")
    else:  # pilot_autotuner
        chosen, pilot_wall, pilot_logs = _pilot_probe(task, candidates, work_dir / "pilots", mms_bin, amr_bin)
        pred_info = {"pilots": pilot_logs}
        reason = "empirical_pilot_projection"

    decision_wall = max(0.0, (time.perf_counter() - t_dec0) - pilot_wall)

    # Execute chosen strategy + bounded policy retries on RETRY_ELIGIBLE (full retry overhead recorded!)
    attempt_logs = []
    cur = chosen
    solver_total = 0.0
    verifier_total = 0.0
    exec_total = 0.0
    retry_overhead = 0.0
    max_rss = 0.0
    final_res = None

    for attempt_idx in range(1, task.policy.max_attempts + 1):
        res = execute_candidate(task, cur, work_dir, mms_bin, amr_bin, tag=f"{strategy}_att{attempt_idx}")
        attempt_logs.append({
            "attempt": attempt_idx,
            "candidate": cur.tag(),
            "verdict": res["verdict"],
            "failure_codes": res.get("failure_codes", []),
            "l2_error": res["l2_error"],
            "wall_seconds": res["wall_seconds"],
            "solver_seconds": res["solver_seconds"],
            "verifier_seconds": res["verifier_seconds"],
        })
        solver_total += res["solver_seconds"]
        verifier_total += res["verifier_seconds"]
        exec_total += res["wall_seconds"]
        max_rss = max(max_rss, float(res["peak_rss_mb"] or 0.0))
        if attempt_idx > 1:
            retry_overhead += res["wall_seconds"]
        final_res = res

        if res["verdict"] != RETRY_ELIGIBLE or attempt_idx == task.policy.max_attempts:
            break
        if cur.mesh_topology == "amr":
            nxt_steps = min(20, cur.amr_max_steps + 2)
            nxt_dofs = min(task.policy.max_dofs_per_level, cur.amr_max_dofs * 2)
            if nxt_steps == cur.amr_max_steps and nxt_dofs == cur.amr_max_dofs:
                break
            cur = StrategyCandidate(
                cur.mesh_type, "amr", cur.polynomial_order, cur.assembly, cur.solver,
                cur.initial_n, amr_max_steps=nxt_steps, amr_max_dofs=nxt_dofs, amr_theta=cur.amr_theta,
            )
        else:
            nxt_n0 = cur.initial_n * 2
            nxt = StrategyCandidate(cur.mesh_type, cur.mesh_topology, cur.polynomial_order, cur.assembly, cur.solver, nxt_n0)
            if nxt.max_dofs_estimate() > task.policy.max_dofs_per_level:
                break
            cur = nxt

    accepted = bool(final_res and final_res["verdict"] == ACCEPTED)
    end_to_end = decision_wall + pilot_wall + exec_total
    return {
        "strategy": strategy,
        "task": task.name,
        "accepted": accepted,
        "verdict": final_res["verdict"] if final_res else "FAILED",
        "chosen_initial": chosen.tag(),
        "chosen_final": cur.tag(),
        "selection_reason": reason,
        "predicted_failure_prob": pred_info.get("pred_failure_prob"),
        "predicted_l2": pred_info.get("pred_l2"),
        "attempts": len(attempt_logs),
        "retries": max(0, len(attempt_logs) - 1),
        "decision_seconds": round(decision_wall, 5),
        "pilot_seconds": round(pilot_wall, 5),
        "solver_seconds": round(solver_total, 5),
        "verifier_seconds": round(verifier_total, 5),
        "retry_overhead_seconds": round(retry_overhead, 5),
        "end_to_end_seconds": round(end_to_end, 5),
        "failed_wall_seconds": round(end_to_end, 5) if not accepted else 0.0,
        "peak_rss_mb": round(max_rss, 2),
        "final_l2_error": final_res["l2_error"] if final_res else None,
        "target_l2": task.validation.max_l2_error_finest,
        "finest_dofs": final_res["finest_dofs"] if final_res else None,
        "attempt_logs": attempt_logs,
    }
