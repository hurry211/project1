"""Independent, residual-verified sparse reference linear solvers.

Combines exact sparse direct factorization (for small/medium systems or strict
direct baselines) with a pure NumPy/SciPy Smoothed-Aggregation Algebraic
Multigrid preconditioned CG solver (AMG-PCG) designed for 200k-1M+ DOF
verification inside bounded memory.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


_CAPTURE_STACK: list[list["ReferenceSolveMetrics"]] = []


@contextmanager
def capture_metrics():
    """Capture ReferenceSolveMetrics records emitted by solve_reference_system in this scope."""
    records: list[ReferenceSolveMetrics] = []
    _CAPTURE_STACK.append(records)
    try:
        yield records
    finally:
        _CAPTURE_STACK.pop()


class ReferenceSolverError(RuntimeError):
    """Raised when the independent reference solver fails convergence or sanity checks."""


@dataclass(frozen=True)
class ReferenceSolveMetrics:
    method: str
    dofs: int
    nnz: int
    iterations: int
    amg_levels: tuple[int, ...]
    setup_seconds: float
    solve_seconds: float
    relative_residual: float
    backward_error: float = 0.0
    residual_norm: float = 0.0


def _pairwise_aggregates(A: sp.csr_matrix, theta: float = 0.25) -> sp.csr_matrix:
    """Vectorized mutual-heavy-edge pairwise aggregation using normalized off-diagonal strength."""
    n = A.shape[0]
    indptr, indices, data = A.indptr, A.indices, A.data
    diag = np.maximum(np.abs(A.diagonal()), 1e-300)
    row_ids = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    off = (row_ids != indices) & (data != 0.0)
    r = row_ids[off]
    c = indices[off]
    # Keep positive coupling weights (M-matrix style off-diagonals, fallback abs for high-order/AMR).
    neg_v = -data[off]
    pos_mask = neg_v > 0.0
    if not np.any(pos_mask):
        neg_v = np.abs(data[off])
    else:
        r, c, neg_v = r[pos_mask], c[pos_mask], neg_v[pos_mask]
    if r.size == 0:
        return sp.identity(n, dtype=np.float64, format="csr")

    denom = np.sqrt(diag[r] * diag[c])
    strength = np.maximum(neg_v / denom, 0.0)
    row_max = np.zeros(n, dtype=np.float64)
    np.maximum.at(row_max, r, strength)
    strong = (strength >= theta * row_max[r]) & (strength > 0.0)
    sr, sc, s_val = r[strong], c[strong], strength[strong]
    if sr.size == 0:
        return sp.identity(n, dtype=np.float64, format="csr")

    # Quantize normalized strength + deterministic symmetric unordered edge hash into int64.
    # Supports N up to 2^22 (~4.19M free DOFs) with exact neighbor index unpacking.
    if n >= (1 << 22):
        raise ReferenceSolverError(f"aggregate index width exceeded for N={n}")
    s_norm = np.clip(s_val / max(float(s_val.max()), 1e-12), 0.0, 1.0)
    q = (s_norm * ((1 << 20) - 1)).astype(np.int64)
    mn = np.minimum(sr, sc).astype(np.uint64)
    mx = np.maximum(sr, sc).astype(np.uint64)
    ehash = (((mn * np.uint64(2654435761)) ^ (mx * np.uint64(2246822519))) * np.uint64(3266489917)) >> np.uint64(44)
    score = (q << 42) | (ehash.astype(np.int64) << 22) | sc.astype(np.int64)

    agg = np.full(n, -1, dtype=np.int32)
    next_id = 0
    nodes_idx = np.arange(n, dtype=np.int64)
    for _ in range(4):
        act = (agg[sr] == -1) & (agg[sc] == -1)
        if not np.any(act):
            break
        best = np.full(n, -1, dtype=np.int64)
        np.maximum.at(best, sr[act], score[act])
        nbr = np.where((agg == -1) & (best >= 0), best & 0x3FFFFF, -1)
        mutual = (nbr > nodes_idx) & (nbr[np.maximum(nbr, 0)] == nodes_idx)
        u_nodes = np.flatnonzero(mutual)
        if u_nodes.size == 0:
            break
        v_nodes = nbr[u_nodes]
        ids = np.arange(next_id, next_id + u_nodes.size, dtype=np.int32)
        agg[u_nodes] = ids
        agg[v_nodes] = ids
        next_id += int(u_nodes.size)

    rem_mask = (agg[sr] == -1) & (agg[sc] != -1)
    if np.any(rem_mask):
        best = np.full(n, -1, dtype=np.int64)
        np.maximum.at(best, sr[rem_mask], score[rem_mask])
        sing = np.flatnonzero((agg == -1) & (best >= 0))
        agg[sing] = agg[best[sing] & 0x3FFFFF]

    iso = np.flatnonzero(agg == -1)
    if iso.size:
        agg[iso] = np.arange(next_id, next_id + iso.size, dtype=np.int32)
        next_id += int(iso.size)

    return sp.csr_matrix(
        (np.ones(n, dtype=np.float64), agg, np.arange(n + 1, dtype=np.int32)),
        shape=(n, next_id),
    )


def _double_pairwise_agg(A: sp.csr_matrix, theta: float = 0.25) -> sp.csr_matrix:
    P1 = _pairwise_aggregates(A, theta)
    if P1.shape[1] >= int(0.92 * A.shape[0]) or P1.shape[1] <= 400:
        return P1
    A1 = (P1.T @ A @ P1).tocsr()
    P2 = _pairwise_aggregates(A1, theta)
    return (P1 @ P2).tocsr()


def _estimate_spectral_radius(A: sp.csr_matrix, diag: np.ndarray, iters: int = 8) -> float:
    rng = np.random.default_rng(0)
    v = rng.standard_normal(A.shape[0])
    v /= max(float(np.linalg.norm(v)), 1e-300)
    invd = 1.0 / np.maximum(diag, 1e-300)
    lam = 2.0
    for _ in range(iters):
        w = invd * (A @ v)
        nrm = float(np.linalg.norm(w))
        if nrm == 0.0:
            return 1.0
        lam = float(np.dot(v, w))
        v = w / nrm
    return max(abs(lam) * 1.1, 1.0)


def _chebyshev_smooth(
    A: sp.csr_matrix,
    invd: np.ndarray,
    b: np.ndarray,
    x: np.ndarray | None,
    lam_max: float,
    order: int = 2,
    zero_init: bool = False,
) -> np.ndarray:
    lo, hi = 0.2 * lam_max, lam_max
    c = 0.5 * (hi - lo)
    d = 0.5 * (hi + lo)
    if zero_init or x is None:
        x = np.zeros_like(b)
        zero_init = True
    alpha_prev = 0.0
    p = np.zeros_like(b)
    for k in range(order):
        r = b if (zero_init and k == 0) else (b - A @ x)
        z = invd * r
        if k == 0:
            alpha = 1.0 / d
            p = z
        elif k == 1:
            beta = 0.5 * (c / d) ** 2
            alpha = 1.0 / (d - 0.5 * c * c / d)
            p = z + beta * alpha_prev * p
        else:
            beta = (0.5 * c * alpha_prev) ** 2
            alpha = 1.0 / (d - 0.25 * c * c * alpha_prev)
            p = z + beta * p
        x = x + alpha * p
        alpha_prev = alpha
    return x


def _build_amg_hierarchy(K: sp.csr_matrix, max_coarse: int = 800):
    levels = []
    A = K.tocsr()
    A.sum_duplicates()
    while A.shape[0] > max_coarse and len(levels) < 10:
        diag = np.maximum(np.abs(A.diagonal()), 1e-300)
        invd = 1.0 / diag
        lam_max = _estimate_spectral_radius(A, diag, iters=8)
        P0 = _double_pairwise_agg(A, theta=0.25)
        if P0.shape[1] >= int(0.88 * A.shape[0]):
            break
        omega = min(0.75, 4.0 / (3.0 * lam_max))
        P = (P0 - sp.diags(omega * invd, 0, format="csr") @ (A @ P0)).tocsr()
        Pt = P.T.tocsr()
        Ac = (Pt @ A @ P).tocsr()
        Ac.sum_duplicates()
        levels.append((A, invd, lam_max, P, Pt))
        A = Ac
    coarse_lu = spla.splu(A.tocsc())
    return levels, coarse_lu


def _vcycle(levels, coarse_lu, b: np.ndarray) -> np.ndarray:
    xs = []
    bs = [b]
    for A, invd, lam_max, P, Pt in levels:
        x = _chebyshev_smooth(A, invd, bs[-1], None, lam_max, order=2, zero_init=True)
        r = bs[-1] - A @ x
        xs.append(x)
        bs.append(Pt @ r)
    uc = coarse_lu.solve(bs[-1])
    for idx in range(len(levels) - 1, -1, -1):
        A, invd, lam_max, P, Pt = levels[idx]
        x = xs[idx] + P @ uc
        uc = _chebyshev_smooth(A, invd, bs[idx], x, lam_max, order=2, zero_init=False)
    return uc


def solve_reference_direct(K_free: sp.csr_matrix | sp.csc_matrix, rhs: np.ndarray) -> tuple[np.ndarray, ReferenceSolveMetrics]:
    """Exact sparse LU reference solve (SuperLU)."""
    N = rhs.size
    if N == 0:
        return np.empty(0, dtype=np.float64), ReferenceSolveMetrics("direct", 0, 0, 1, (), 0.0, 0.0, 0.0, 0.0)
    t0 = time.perf_counter()
    u = spla.spsolve(K_free.tocsc(), rhs)
    dt = time.perf_counter() - t0
    if not np.all(np.isfinite(u)):
        raise ReferenceSolverError("direct reference solve returned non-finite values")
    rhs_norm = float(np.linalg.norm(rhs))
    res = float(np.linalg.norm(K_free @ u - rhs))
    rel = res / max(rhs_norm, 1e-300)
    ku_abs = np.abs(K_free) @ np.abs(u)
    bwd = res / max(float(np.linalg.norm(ku_abs)) + rhs_norm, 1e-300)
    return u, ReferenceSolveMetrics("direct", N, int(K_free.nnz), 1, (N,), 0.0, dt, rel, bwd, res)


def solve_reference_amg_pcg(
    K_free: sp.csr_matrix,
    rhs: np.ndarray,
    *,
    rtol: float = 1e-14,
    verify_residual_tol: float = 1e-11,
    maxiter: int = 300,
) -> tuple[np.ndarray, ReferenceSolveMetrics]:
    """Solve K_free u = rhs via smoothed-aggregation AMG-PCG with backward-stability verification."""
    N = rhs.size
    if N == 0:
        return np.empty(0, dtype=np.float64), ReferenceSolveMetrics("amg_pcg", 0, 0, 0, (), 0.0, 0.0, 0.0)
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return np.zeros(N, dtype=np.float64), ReferenceSolveMetrics("amg_pcg", N, int(K_free.nnz), 0, (N,), 0.0, 0.0, 0.0)

    t_setup0 = time.perf_counter()
    levels, coarse_lu = _build_amg_hierarchy(K_free)
    t_setup = time.perf_counter() - t_setup0
    level_sizes = tuple([A.shape[0] for A, *_ in levels] + [coarse_lu.shape[0]])

    M = spla.LinearOperator((N, N), matvec=lambda v: _vcycle(levels, coarse_lu, v), dtype=np.float64)
    iters = 0

    def _count(_):
        nonlocal iters
        iters += 1

    t_solve0 = time.perf_counter()
    u, info = spla.cg(K_free, rhs, M=M, rtol=rtol, atol=0.0, maxiter=maxiter, callback=_count)
    res_vec = rhs - K_free @ u
    res_norm = float(np.linalg.norm(res_vec))
    rel_res = res_norm / rhs_norm
    # Iterative refinement step if preconditioned CG stopped slightly shy of residual target.
    refine_round = 0
    while (not math.isfinite(rel_res) or rel_res > verify_residual_tol) and refine_round < 2 and np.all(np.isfinite(res_vec)):
        du, info2 = spla.cg(K_free, res_vec, M=M, rtol=rtol, atol=0.0, maxiter=maxiter, callback=_count)
        u = u + du
        info = info2
        res_vec = rhs - K_free @ u
        res_norm = float(np.linalg.norm(res_vec))
        rel_res = res_norm / rhs_norm
        refine_round += 1
    t_solve = time.perf_counter() - t_solve0

    # Verify both relative residual and Oettli-Prager normwise backward stability:
    # For N~1e6 2nd-order PDE matrices (kappa ~ O(N)), float64 matvec K@u roundoff
    # induces an unavoidable residual floor proportional to eps * || |K| |u| ||_2 / ||rhs||_2.
    finite = bool(np.all(np.isfinite(u)) and math.isfinite(rel_res))
    if not finite or info != 0:
        raise ReferenceSolverError(
            f"amg_pcg failed convergence (info={info}, rel_res={rel_res:.3e})"
        )
    ku_abs = np.abs(K_free) @ np.abs(u)
    ku_abs_norm = float(np.linalg.norm(ku_abs))
    bwd_err = res_norm / max(ku_abs_norm + rhs_norm, 1e-300)
    fl_floor = 256.0 * float(np.finfo(np.float64).eps) * (ku_abs_norm + rhs_norm) / rhs_norm
    effective_tol = min(1e-10, max(verify_residual_tol, fl_floor))
    if rel_res > effective_tol or bwd_err > 1e-13:
        raise ReferenceSolverError(
            f"amg_pcg failed residual verification (info={info}, rel_res={rel_res:.3e} > {effective_tol:.3e}, "
            f"bwd_err={bwd_err:.3e})"
        )
    return u, ReferenceSolveMetrics("amg_pcg", N, int(K_free.nnz), iters, level_sizes, t_setup, t_solve, rel_res, bwd_err, res_norm)


def solve_reference_pyamg(
    K_free: sp.csr_matrix,
    rhs: np.ndarray,
    *,
    rtol: float = 1e-14,
    verify_residual_tol: float = 1e-11,
    maxiter: int = 300,
) -> tuple[np.ndarray, ReferenceSolveMetrics]:
    """Mature third-party Smoothed-Aggregation AMG (pyamg) + PCG with identical residual verification."""
    try:
        import pyamg
    except ImportError as exc:
        raise ReferenceSolverError(f"pyamg is not installed in the active Python environment: {exc}") from exc

    N = rhs.size
    if N == 0:
        return np.empty(0, dtype=np.float64), ReferenceSolveMetrics("pyamg", 0, 0, 0, (), 0.0, 0.0, 0.0, 0.0)
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return np.zeros(N, dtype=np.float64), ReferenceSolveMetrics("pyamg", N, int(K_free.nnz), 0, (N,), 0.0, 0.0, 0.0, 0.0)

    t_setup0 = time.perf_counter()
    ml = pyamg.smoothed_aggregation_solver(K_free)
    t_setup = time.perf_counter() - t_setup0
    level_sizes = tuple(int(lv.A.shape[0]) for lv in ml.levels)
    M = ml.aspreconditioner(cycle="V")

    iters = 0

    def _count(_):
        nonlocal iters
        iters += 1

    t_solve0 = time.perf_counter()
    u, info = spla.cg(K_free, rhs, M=M, rtol=rtol, atol=0.0, maxiter=maxiter, callback=_count)
    res_vec = rhs - K_free @ u
    res_norm = float(np.linalg.norm(res_vec))
    rel_res = res_norm / rhs_norm
    refine_round = 0
    while (not math.isfinite(rel_res) or rel_res > verify_residual_tol) and refine_round < 2 and np.all(np.isfinite(res_vec)):
        du, info2 = spla.cg(K_free, res_vec, M=M, rtol=rtol, atol=0.0, maxiter=maxiter, callback=_count)
        u = u + du
        info = info2
        res_vec = rhs - K_free @ u
        res_norm = float(np.linalg.norm(res_vec))
        rel_res = res_norm / rhs_norm
        refine_round += 1
    t_solve = time.perf_counter() - t_solve0

    finite = bool(np.all(np.isfinite(u)) and math.isfinite(rel_res))
    if not finite or info != 0:
        raise ReferenceSolverError(f"pyamg_pcg failed convergence (info={info}, rel_res={rel_res:.3e})")
    ku_abs = np.abs(K_free) @ np.abs(u)
    ku_abs_norm = float(np.linalg.norm(ku_abs))
    bwd_err = res_norm / max(ku_abs_norm + rhs_norm, 1e-300)
    fl_floor = 256.0 * float(np.finfo(np.float64).eps) * (ku_abs_norm + rhs_norm) / rhs_norm
    effective_tol = min(1e-10, max(verify_residual_tol, fl_floor))
    if rel_res > effective_tol or bwd_err > 1e-13:
        raise ReferenceSolverError(
            f"pyamg_pcg failed residual verification (info={info}, rel_res={rel_res:.3e} > {effective_tol:.3e}, "
            f"bwd_err={bwd_err:.3e})"
        )
    return u, ReferenceSolveMetrics("pyamg", N, int(K_free.nnz), iters, level_sizes, t_setup, t_solve, rel_res, bwd_err, res_norm)


def solve_reference_system(
    K_free: sp.csr_matrix,
    rhs: np.ndarray,
    *,
    method: str = "auto",
    amg_threshold_dofs: int = 300_000,
    rtol: float = 1e-14,
    verify_residual_tol: float = 1e-11,
) -> tuple[np.ndarray, ReferenceSolveMetrics]:
    """Dispatch to direct LU, residual-verified SA-AMG-PCG, or mature pyamg without relaxing accuracy."""
    if method not in ("auto", "direct", "amg_pcg", "pyamg"):
        raise ValueError(f"unknown reference solver method: {method!r}")
    if method == "pyamg":
        sol, metrics = solve_reference_pyamg(K_free, rhs, rtol=rtol, verify_residual_tol=verify_residual_tol)
    else:
        use_amg = method == "amg_pcg" or (method == "auto" and rhs.size > amg_threshold_dofs)
        if use_amg:
            sol, metrics = solve_reference_amg_pcg(K_free, rhs, rtol=rtol, verify_residual_tol=verify_residual_tol)
        else:
            sol, metrics = solve_reference_direct(K_free, rhs)
    if _CAPTURE_STACK:
        _CAPTURE_STACK[-1].append(metrics)
    return sol, metrics
