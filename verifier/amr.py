"""Independent verification for MFEM Example 6 Adaptive Mesh Refinement (AMR) runs.

Reconstructs T-junction hanging-node constraints purely from exported physical
mesh vertices, element DOF coordinates, and dof_values.txt without reading
MFEM's internal prolongation matrix. Verifies:
  1. Positive element Jacobians, boundary attribute & geometry coverage.
  2. H1 hanging-node edge continuity (rejects broken T-junction traces).
  3. Dirichlet nodal values.
  4. Independent quadrature L2 and H1-seminorm error computation + self-report check.
  5. Constrained variational reference solve C^T K C u_free = rhs + solution comparison.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp

from sciagent.spec import SIDE_GEOMETRY, SIDE_NORMAL, ProblemSpec, ValidationSpec, compact_expression
from . import expr
from .fem import (
    ArtifactError,
    LocalBasis,
    MeshData,
    build_local_basis,
    element_quadrature,
    eval_monomials,
    gauss_01,
    inverse_map,
    load_mesh_data,
)
from .reference_solver import ReferenceSolverError, solve_reference_system

GEOM_TOL = 1e-11
HANGING_TOL = 1e-10


@dataclass
class AmrCheck:
    name: str
    passed: bool
    code: str
    detail: str = ""
    value: float | None = None
    threshold: float | None = None


@dataclass
class AmrStepReport:
    step: int
    num_elements: int | None
    num_dofs: int | None
    num_true_dofs_inferred: int | None
    hanging_dof_count: int
    max_hanging_continuity_error: float | None
    l2_error: float | None
    h1_semi_error: float | None
    self_reported_l2_error: float | None
    self_reported_h1_semi_error: float | None
    reference_rel_diff: float | None
    conforming: bool | None
    checks: list[AmrCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)


def _lshape_exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.asarray(x, dtype=np.float64) - 0.5
    dy = np.asarray(y, dtype=np.float64) - 0.5
    r2 = dx * dx + dy * dy
    phi = np.arctan2(dy, dx)
    phi = np.where(phi < 0.5 * math.pi - 1e-12, phi + 2.0 * math.pi, phi)
    theta = phi - 0.5 * math.pi
    return np.where(r2 == 0.0, 0.0, np.power(r2, 1.0 / 3.0) * np.sin((2.0 / 3.0) * theta))


def _lshape_grad(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.asarray(x, dtype=np.float64) - 0.5
    dy = np.asarray(y, dtype=np.float64) - 0.5
    r2 = dx * dx + dy * dy
    safe_r2 = np.maximum(r2, 1e-30)
    r = np.sqrt(safe_r2)
    phi = np.arctan2(dy, dx)
    phi = np.where(phi < 0.5 * math.pi - 1e-12, phi + 2.0 * math.pi, phi)
    theta = phi - 0.5 * math.pi
    coef = (2.0 / 3.0) * np.power(safe_r2, -1.0 / 6.0)
    ur = coef * np.sin((2.0 / 3.0) * theta)
    uth = coef * np.cos((2.0 / 3.0) * theta)
    ux = np.where(r2 < 1e-30, 0.0, ur * (dx / r) - uth * (dy / r))
    uy = np.where(r2 < 1e-30, 0.0, ur * (dy / r) + uth * (dx / r))
    return ux, uy


def _resolve_case_eval(res: dict, problem: ProblemSpec | None) -> tuple[Callable, Callable, Callable, Callable, str]:
    case_id = res.get("case", "")
    if case_id == "lshape_singularity":
        zero = lambda x, y: np.zeros_like(np.asarray(x, dtype=np.float64))
        one = lambda x, y: np.ones_like(np.asarray(x, dtype=np.float64))
        return _lshape_exact, _lshape_grad, zero, one, "-laplacian(u) = f"
    if problem is not None:
        u_fn = expr.compile_expr(problem.exact_solution)
        f_fn = expr.compile_expr(problem.source)
        k_fn = expr.compile_expr(problem.kappa)
        return u_fn, (lambda x, y: expr.gradient(u_fn, x, y)), f_fn, k_fn, problem.equation
    u_fn = expr.compile_expr(res["case_exact_solution"])
    f_fn = expr.compile_expr(res["case_source"])
    k_fn = expr.compile_expr(res.get("case_diffusivity", "1"))
    return u_fn, (lambda x, y: expr.gradient(u_fn, x, y)), f_fn, k_fn, res.get("case_equation", "-laplacian(u) = f")


def _lagrange_1d_weights(nodes_t: np.ndarray, t_eval: float) -> np.ndarray:
    """Evaluate 1D Lagrange basis polynomials defined by nodes_t at t_eval."""
    n = nodes_t.size
    w = np.ones(n, dtype=np.float64)
    for j in range(n):
        denom = 1.0
        numer = 1.0
        for m in range(n):
            if m != j:
                numer *= (t_eval - nodes_t[m])
                denom *= (nodes_t[j] - nodes_t[m])
        w[j] = numer / denom
    return w


def detect_hanging_constraints(
    mesh: MeshData, coords: np.ndarray, order: int
) -> tuple[dict[int, list[tuple[int, float]]], list[int], sp.csr_matrix]:
    """Independently identify T-junction slave DOFs and build prolongation C (N_full x N_true).

    For each element e and each geometric edge (pA, pB) of e:
      1) Collect the (order + 1) parent DOFs of element e lying collinearly on closed segment [pA, pB].
      2) Search all exported global DOFs via spatial hashing / bounding box that lie strictly
         interior to open segment (pA, pB) and are NOT in element e's own edge DOFs.
      3) Any such DOF is constrained to equal the 1D Lagrange interpolant along parent edge [pA, pB].
    Handles multi-level irregular hanging nodes recursively via topological resolution.
    """
    N = coords.shape[0]
    E, nv = mesh.verts.shape[:2]

    # Collect per-element edge parent DOFs and record raw edge -> parent relation
    constraints: dict[int, list[tuple[int, float]]] = {}

    # Coarse spatial lookup grid so searching O(N) nodes per edge is fast even for 50k DOFs
    cell_scale = max(16, int(math.sqrt(max(N, 1))))
    cell_idx = np.floor(np.clip(coords, 0.0, 1.0) * cell_scale).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for did, (cx, cy) in enumerate(cell_idx):
        buckets.setdefault((int(cx), int(cy)), []).append(did)

    tol_perp = 1e-11
    # Map each dof -> list of (p0, p1, seg_len, udir) for all element edges containing that dof
    dof_edges: dict[int, list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]] = {}
    edge_records: list[tuple[float, np.ndarray, np.ndarray, list[int], np.ndarray]] = []

    for e in range(E):
        verts = mesh.verts[e]
        edofs = mesh.elem_dofs[e]
        exy = mesh.elem_node_xy[e]
        for k in range(nv):
            p0 = verts[k]
            p1 = verts[(k + 1) % nv]
            vec = p1 - p0
            seg_len2 = float(np.dot(vec, vec))
            if seg_len2 <= 1e-28:
                continue
            seg_len = math.sqrt(seg_len2)
            udir = vec / seg_len

            # Identify element e's own DOFs on closed edge [p0, p1]
            rel_e = exy - p0[None, :]
            t_e = (rel_e @ vec) / seg_len2
            perp_e = np.abs(rel_e[:, 0] * vec[1] - rel_e[:, 1] * vec[0]) / seg_len
            on_edge = (perp_e <= tol_perp * max(seg_len, 1.0)) & (t_e >= -1e-11) & (t_e <= 1.0 + 1e-11)
            parent_local = np.flatnonzero(on_edge)
            if parent_local.size != order + 1:
                continue
            sort_idx = np.argsort(t_e[parent_local])
            parent_local = parent_local[sort_idx]
            parent_dofs = [int(edofs[li]) for li in parent_local]
            parent_ts = np.clip(t_e[parent_local], 0.0, 1.0)
            for pd in parent_dofs:
                dof_edges.setdefault(pd, []).append((p0, p1, seg_len, udir))
            edge_records.append((seg_len, p0, p1, parent_dofs, parent_ts))

    # Process shorter edges last or longer edges first; never let a shorter edge constrain a collinear parent DOF.
    edge_records.sort(key=lambda item: item[0], reverse=True)
    for seg_len, p0, p1, parent_dofs, parent_ts in edge_records:
        vec = p1 - p0
        seg_len2 = float(np.dot(vec, vec))
        udir = vec / seg_len
        parent_set = set(parent_dofs)

        xmin, xmax = min(p0[0], p1[0]) - 1e-9, max(p0[0], p1[0]) + 1e-9
        ymin, ymax = min(p0[1], p1[1]) - 1e-9, max(p0[1], p1[1]) + 1e-9
        ix0 = max(0, int(math.floor(xmin * cell_scale)) - 1)
        ix1 = min(cell_scale, int(math.floor(xmax * cell_scale)) + 1)
        iy0 = max(0, int(math.floor(ymin * cell_scale)) - 1)
        iy1 = min(cell_scale, int(math.floor(ymax * cell_scale)) + 1)

        cands: list[int] = []
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                cands.extend(buckets.get((ix, iy), ()))
        if not cands:
            continue
        c_arr = np.asarray(cands, dtype=np.int64)
        c_rel = coords[c_arr] - p0[None, :]
        c_t = (c_rel @ vec) / seg_len2
        c_perp = np.abs(c_rel[:, 0] * vec[1] - c_rel[:, 1] * vec[0]) / seg_len
        hits = np.flatnonzero(
            (c_perp <= tol_perp * max(seg_len, 1.0)) & (c_t > 1e-9) & (c_t < 1.0 - 1e-9)
        )
        for h_idx in hits:
            cand_dof = int(c_arr[h_idx])
            if cand_dof in parent_set or cand_dof in constraints:
                continue
            # Ensure cand_dof does not belong to a collinear edge at least as long as [p0, p1]
            # (prevents high-order Q3/Q4 child edges from constraining coarse parent interior nodes).
            is_coarse_or_peer = any(
                abs(ed_len >= seg_len * (1.0 - 1e-6)) and abs(abs(float(np.dot(ed_dir, udir))) - 1.0) <= 1e-8
                and abs(float((ed_p0[0] - p0[0]) * udir[1] - (ed_p0[1] - p0[1]) * udir[0])) <= 1e-10
                for ed_p0, _, ed_len, ed_dir in dof_edges.get(cand_dof, ())
            )
            if is_coarse_or_peer:
                continue
            w = _lagrange_1d_weights(parent_ts, float(c_t[h_idx]))
            constraints[cand_dof] = [(pd, float(pw)) for pd, pw in zip(parent_dofs, w) if abs(pw) > 1e-15]

    # Resolve nested multi-level slave chains transitively until all RHS entries are true DOFs
    true_dofs = [d for d in range(N) if d not in constraints]
    true_map = {d: idx for idx, d in enumerate(true_dofs)}
    resolved: dict[int, list[tuple[int, float]]] = {}

    def _resolve(d: int, visiting: set[int]) -> list[tuple[int, float]]:
        if d in resolved:
            return resolved[d]
        if d not in constraints:
            return [(true_map[d], 1.0)]
        if d in visiting:
            raise ArtifactError(f"cyclic hanging-node constraint detected at dof {d}")
        visiting.add(d)
        acc: dict[int, float] = {}
        for pd, pw in constraints[d]:
            for t_idx, tw in _resolve(pd, visiting):
                acc[t_idx] = acc.get(t_idx, 0.0) + pw * tw
        visiting.remove(d)
        resolved[d] = [(k, v) for k, v in acc.items() if abs(v) > 1e-15]
        return resolved[d]

    rows, cols, vals = [], [], []
    for d in range(N):
        if d not in constraints:
            rows.append(d)
            cols.append(true_map[d])
            vals.append(1.0)
        else:
            terms = _resolve(d, set())
            for ti, tw in terms:
                rows.append(d)
                cols.append(ti)
                vals.append(tw)

    C = sp.csr_matrix((vals, (rows, cols)), shape=(N, len(true_dofs)), dtype=np.float64)
    return constraints, true_dofs, C


def _detect_domain_Dirichlet(
    mesh: MeshData, coords: np.ndarray, dir_attrs: set[int], bdr_points_only: bool = True
) -> np.ndarray:
    """Identify full DOFs located geometrically on Dirichlet boundary segments."""
    on_dir = np.zeros(coords.shape[0], dtype=bool)
    dir_segs = [seg for attr, seg in zip(mesh.bdr_attr, mesh.bdr_xy) if int(attr) in dir_attrs]
    if not dir_segs:
        return on_dir
    for seg in dir_segs:
        p0, p1 = seg[0], seg[1]
        v = p1 - p0
        L2 = float(np.dot(v, v))
        if L2 <= 1e-28:
            continue
        L = math.sqrt(L2)
        rel = coords - p0[None, :]
        t = (rel @ v) / L2
        perp = np.abs(rel[:, 0] * v[1] - rel[:, 1] * v[0]) / L
        on_dir |= (perp <= 1e-11) & (t >= -1e-11) & (t <= 1.0 + 1e-11)
    return on_dir


def verify_amr_step(
    step_dir: Path,
    problem: ProblemSpec | None = None,
    validation: ValidationSpec | None = None,
    *,
    require_reference_solve: bool = True,
) -> AmrStepReport:
    """Verify one step_XX directory from an Example 6 AMR run independently."""
    step_dir = Path(step_dir)
    val = validation or ValidationSpec()
    rpath = step_dir / "result.json"
    checks: list[AmrCheck] = []
    if not rpath.is_file():
        return AmrStepReport(0, None, None, None, 0, None, None, None, None, None, None, None,
                             [AmrCheck("result_json_present", False, "MISSING_ARTIFACT", "result.json missing")])
    res = json.loads(rpath.read_text())
    step = int(res.get("amr_step", 0))
    order = int(res["order"])
    mesh = load_mesh_data(step_dir)
    N = mesh.values.size
    E = mesh.verts.shape[0]
    finite = bool(np.all(np.isfinite(mesh.values)))
    checks.append(AmrCheck("finite_solution", finite, "NON_FINITE_SOLUTION", f"non-finite={int(np.sum(~np.isfinite(mesh.values)))}"))
    if not finite:
        return AmrStepReport(step, E, N, None, 0, None, None, None,
                             res.get("self_reported_l2_error"), res.get("self_reported_h1_semi_error"),
                             None, res.get("conforming"), checks)

    coords = np.full((N, 2), np.nan)
    flat_d = mesh.elem_dofs.ravel()
    coords[flat_d] = mesh.elem_node_xy.reshape(-1, 2)
    spread = float(np.max(np.abs(coords[flat_d] - mesh.elem_node_xy.reshape(-1, 2))))
    checks.append(AmrCheck("dof_coordinate_consistency", spread <= 1e-12, "MALFORMED_ARTIFACT",
                           "max coordinate mismatch across elements", spread, 1e-12))

    constraints, true_dofs, C = detect_hanging_constraints(mesh, coords, order)
    if constraints:
        u_interp = C @ mesh.values[true_dofs]
        slave_ids = sorted(constraints.keys())
        hang_err = float(np.max(np.abs(mesh.values[slave_ids] - u_interp[slave_ids])))
    else:
        hang_err = 0.0
    checks.append(AmrCheck("hanging_node_continuity", hang_err <= HANGING_TOL, "HANGING_NODE_DISCONTINUITY",
                           f"max T-junction trace jump across {len(constraints)} hanging DOFs",
                           hang_err, HANGING_TOL))

    u_fn, grad_fn, f_fn, k_fn, eq = _resolve_case_eval(res, problem)
    dir_attrs = set(int(a) for a in res.get("dirichlet_attributes", []))
    on_dir = _detect_domain_Dirichlet(mesh, coords, dir_attrs)
    u_dir_exact = u_fn(coords[on_dir, 0], coords[on_dir, 1]) if on_dir.any() else np.empty(0)
    dmax = float(np.max(np.abs(mesh.values[on_dir] - u_dir_exact))) if on_dir.any() else 0.0
    checks.append(AmrCheck("dirichlet_nodal_values", dmax <= val.dirichlet_nodal_abs_tol, "DIRICHLET_VIOLATION",
                           f"max |u_h - u_exact| over {int(on_dir.sum())} Dirichlet DOFs", dmax, val.dirichlet_nodal_abs_tol))

    singular_case = res.get("case", "") == "lshape_singularity"
    m = max(order + 6, 16 if singular_case else 12)
    basis = build_local_basis(mesh, order)
    q = element_quadrature(mesh, basis, m)
    min_detJ = float(np.min(q.detJ))
    checks.append(AmrCheck("mesh_jacobian_positive", min_detJ > 0.0, "INVALID_MESH",
                           "min detJ across AMR quadrature points", min_detJ, 0.0))

    U = mesh.values[mesh.elem_dofs]
    uh = np.einsum("eqk,ek->eq", q.phi, U)
    guh = np.einsum("eqki,ek->eqi", q.gphi, U)
    x, y = q.X[..., 0], q.X[..., 1]
    ue = u_fn(x, y)
    gx, gy = grad_fn(x, y)
    l2 = float(np.sqrt(np.sum(q.wdet * (uh - ue) ** 2)))
    h1 = float(np.sqrt(np.sum(q.wdet * ((guh[..., 0] - gx) ** 2 + (guh[..., 1] - gy) ** 2))))

    sr_tol = 5e-3 if singular_case else val.self_report_rel_tol
    for name, mine, theirs in (("l2", l2, res.get("self_reported_l2_error")),
                               ("h1_semi", h1, res.get("self_reported_h1_semi_error"))):
        ok_sr = isinstance(theirs, (int, float)) and math.isfinite(theirs)
        rel_sr = abs(mine - theirs) / max(abs(mine), 1e-300) if ok_sr else math.inf
        checks.append(AmrCheck(f"self_report_{name}", ok_sr and rel_sr <= sr_tol,
                               "SELF_REPORT_MISMATCH", f"verifier={mine:.10e}, solver={theirs}",
                               rel_sr if math.isfinite(rel_sr) else None, sr_tol))

    ref_rel_diff = None
    if require_reference_solve:
        nloc = mesh.elem_dofs.shape[1]
        kappa_vals = None if res.get("case_diffusivity", "1") == "1" else k_fn(x, y)
        weights = q.wdet if kappa_vals is None else (q.wdet * kappa_vals)
        Ke = np.einsum("eq,eqid,eqjd->eij", weights, q.gphi, q.gphi)
        Fe = np.einsum("eq,eq,eqi->ei", q.wdet, f_fn(x, y), q.phi)
        rows = np.repeat(mesh.elem_dofs, nloc, axis=1).ravel()
        cols = np.tile(mesh.elem_dofs, (1, nloc)).ravel()
        K_full = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(N, N)).tocsr()
        F_full = np.bincount(mesh.elem_dofs.ravel(), weights=Fe.ravel(), minlength=N)

        # Apply constrained Galerkin variational formulation C^T K_full C
        K_con = (C.T @ K_full @ C).tocsr()
        F_con = C.T @ F_full
        true_arr = np.asarray(true_dofs, dtype=np.int64)
        true_dir = on_dir[true_arr]
        true_free = ~true_dir
        u_true = np.zeros(true_arr.size, dtype=np.float64)
        u_true[true_dir] = u_fn(coords[true_arr[true_dir], 0], coords[true_arr[true_dir], 1])
        rhs = F_con[true_free] - K_con[true_free][:, true_dir] @ u_true[true_dir]
        K_free = K_con[true_free][:, true_free].tocsr()
        try:
            u_free_sol, _ = solve_reference_system(
                K_free, rhs, method=val.reference_solve_method, amg_threshold_dofs=val.reference_solve_amg_threshold_dofs
            )
            u_true[true_free] = u_free_sol
            u_ref = C @ u_true
            ref_rel_diff = float(np.max(np.abs(u_ref - mesh.values)) / max(np.max(np.abs(u_ref)), 1e-300))
            checks.append(AmrCheck("reference_solve_agreement", ref_rel_diff <= val.reference_solve_rel_tol,
                                   "REFERENCE_MISMATCH", "constrained C^T K C reference solve vs MFEM",
                                   ref_rel_diff, val.reference_solve_rel_tol))
        except ReferenceSolverError as exc:
            checks.append(AmrCheck("reference_solve", False, "REFERENCE_SOLVER_FAILED", str(exc)))

    return AmrStepReport(
        step=step,
        num_elements=E,
        num_dofs=N,
        num_true_dofs_inferred=len(true_dofs),
        hanging_dof_count=len(constraints),
        max_hanging_continuity_error=hang_err,
        l2_error=l2,
        h1_semi_error=h1,
        self_reported_l2_error=res.get("self_reported_l2_error"),
        self_reported_h1_semi_error=res.get("self_reported_h1_semi_error"),
        reference_rel_diff=ref_rel_diff,
        conforming=res.get("conforming"),
        checks=checks,
    )


def verify_amr_run(
    amr_dir: Path,
    problem: ProblemSpec | None = None,
    validation: ValidationSpec | None = None,
    *,
    require_reference_solve: bool = True,
) -> dict:
    """Verify an entire sequence of step_XX subdirectories and overall error decay."""
    amr_dir = Path(amr_dir)
    step_dirs = sorted(p for p in amr_dir.iterdir() if p.is_dir() and p.name.startswith("step_"))
    reports = [
        verify_amr_step(sd, problem, validation, require_reference_solve=require_reference_solve)
        for sd in step_dirs
    ]
    all_ok = bool(reports) and all(r.ok for r in reports)
    # Require net error decay between coarse step 0 and final refined step when >= 2 steps exist.
    net_decay = True
    if len(reports) >= 2 and reports[0].h1_semi_error and reports[-1].h1_semi_error:
        net_decay = reports[-1].h1_semi_error < reports[0].h1_semi_error
    codes = sorted({c.code for r in reports for c in r.checks if not c.passed})
    if not reports:
        codes.append("MISSING_ARTIFACT")
    if not net_decay:
        codes.append("NO_AMR_ERROR_REDUCTION")
    verdict = "ACCEPTED" if (all_ok and net_decay) else "REJECTED"
    return {
        "verifier": "verifier.amr",
        "verdict": verdict,
        "failure_codes": codes,
        "step_count": len(reports),
        "net_h1_error_decay": net_decay,
        "steps": [asdict(r) | {"ok": r.ok} for r in reports],
    }
