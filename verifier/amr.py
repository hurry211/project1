"""Independent verification for MFEM Example 6 Adaptive Mesh Refinement (AMR) runs.

Reconstructs T-junction hanging-node constraints purely from exported physical
mesh vertices, element DOF coordinates, and dof_values.txt without reading
MFEM's internal prolongation matrix, and reads physics, diffusivity, boundary
conditions, and tolerances strictly from frozen ProblemSpec / ExecutionSpec /
ValidationSpec.

Verifies on every AMR step:
  1. Frozen-spec claim consistency, quadrature claims, CG stopping & process exit.
  2. Domain bounds, positive Jacobians, exact domain area, and watertight
     oriented-edge balance (catching missing/overlapping elements across T-junctions).
  3. Boundary attribute coverage, segment geometry/side length, and interval partition.
  4. Multi-level H1 hanging-node trace continuity (<= 1e-10).
  5. Geometric Dirichlet nodal values from frozen ProblemSpec.
  6. Matched-order self-report consistency at strict ValidationSpec tolerance (<= 1e-6)
     plus independent higher-order quadrature L2 and H1-seminorm errors.
  7. Constrained Galerkin variational reference solve C^T K C u_true = C^T F
     including variable diffusivity and non-zero Neumann boundary flux integrals.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp

from sciagent.spec import (
    ANISOTROPIC,
    LAYERED,
    SIDE_GEOMETRY,
    SIDE_NORMAL,
    ExecutionSpec,
    ProblemSpec,
    ValidationSpec,
    compact_expression,
)
from sciagent.evidence import artifact_hashes, sha256_file
from sciagent.gmsh import ARCHIVED_MFEM_MESH
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

VERIFIER_VERSION = "0.6.0-amr"
GEOM_TOL = 1e-11
HANGING_TOL = 1e-10
ACCEPTED = "ACCEPTED"
RETRY_ELIGIBLE = "RETRY_ELIGIBLE"
REJECTED = "REJECTED"

_LSHAPE_EDGES = (
    # (axis_fixed, coord_fixed, range_min, range_max, outward_nx, outward_ny, ccw_tx, ccw_ty)
    (1, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0),
    (0, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0),
    (1, 0.5, 0.5, 1.0, 0.0, 1.0, -1.0, 0.0),
    (0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0),
    (1, 1.0, 0.0, 0.5, 0.0, 1.0, -1.0, 0.0),
    (0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0),
)


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


def _finite(v) -> bool:
    try:
        return type(v) in (int, float) and math.isfinite(v)
    except OverflowError:
        return False


def _attributes(value):
    if isinstance(value, list) and all(type(a) is int for a in value):
        return sorted(value)
    return None


def _lagrange_1d_weights(nodes_t: np.ndarray, t_eval: float) -> np.ndarray:
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
    """Independently identify T-junction slave DOFs and build prolongation C (N_full x N_true)."""
    N = coords.shape[0]
    E, nv = mesh.verts.shape[:2]
    constraints: dict[int, list[tuple[int, float]]] = {}

    cell_scale = max(16, int(math.sqrt(max(N, 1))))
    cell_idx = np.floor(np.clip(coords, 0.0, 1.0) * cell_scale).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for did, (cx, cy) in enumerate(cell_idx):
        buckets.setdefault((int(cx), int(cy)), []).append(did)

    tol_perp = 1e-11
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
            is_coarse_or_peer = any(
                ed_len >= seg_len * (1.0 - 1e-6)
                and abs(abs(float(np.dot(ed_dir, udir))) - 1.0) <= 1e-8
                and abs(float((ed_p0[0] - p0[0]) * udir[1] - (ed_p0[1] - p0[1]) * udir[0])) <= 1e-10
                for ed_p0, _, ed_len, ed_dir in dof_edges.get(cand_dof, ())
            )
            if is_coarse_or_peer:
                continue
            w = _lagrange_1d_weights(parent_ts, float(c_t[h_idx]))
            constraints[cand_dof] = [(pd, float(pw)) for pd, pw in zip(parent_dofs, w) if abs(pw) > 1e-15]

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
            for ti, tw in _resolve(d, set()):
                rows.append(d)
                cols.append(ti)
                vals.append(tw)

    C = sp.csr_matrix((vals, (rows, cols)), shape=(N, len(true_dofs)), dtype=np.float64)
    return constraints, true_dofs, C


def _match_lshape_side(seg: np.ndarray) -> tuple[int, tuple[float, float]] | None:
    """Identify which of the 6 L-shape outer perimeter edges contains seg, returning (edge_idx, outward_n)."""
    for idx, (ax, cval, rmin, rmax, nx, ny, _, _) in enumerate(_LSHAPE_EDGES):
        other = 1 - ax
        if (
            np.all(np.abs(seg[:, ax] - cval) <= GEOM_TOL)
            and np.all(seg[:, other] >= rmin - GEOM_TOL)
            and np.all(seg[:, other] <= rmax + GEOM_TOL)
        ):
            return idx, (nx, ny)
    return None


def _check_domain_and_partition(mesh: MeshData, problem: ProblemSpec, checks: list[AmrCheck]) -> None:
    """Verify domain bounds, positive areas, total area, and T-junction watertight edge balance."""
    v = mesh.verts
    x, y = v[..., 0], v[..., 1]
    areas = 0.5 * np.sum(np.roll(x, 1, axis=1) * y - x * np.roll(y, 1, axis=1), axis=1)
    min_area = float(np.min(areas)) if areas.size else 0.0
    total_area = float(np.sum(areas)) if areas.size else 0.0
    expected_area = 0.75 if problem.domain == "lshape_2d" else 1.0

    in_unit = bool(np.all((x >= -GEOM_TOL) & (x <= 1.0 + GEOM_TOL) & (y >= -GEOM_TOL) & (y <= 1.0 + GEOM_TOL)))
    cutout_ok = True
    if problem.domain == "lshape_2d":
        cx = x.mean(axis=1)
        cy = y.mean(axis=1)
        cutout_ok = not bool(
            np.any((x > 0.5 + 1e-10) & (y > 0.5 + 1e-10))
            or np.any((cx > 0.5 + 1e-10) & (cy > 0.5 + 1e-10))
        )
    area_ok = in_unit and cutout_ok and min_area > 1e-14 and abs(total_area - expected_area) <= 1e-10
    checks.append(AmrCheck(
        "amr_element_area_partition", area_ok, "MESH_MISMATCH",
        f"min_area={min_area:.3e}, total_area={total_area:.12f}, expected={expected_area}",
        total_area, expected_area,
    ))

    # Watertight oriented-edge cancellation check across arbitrary multi-level T-junctions:
    # Sum oriented active element edges (+1 CCW) and subtract CCW domain boundary segments.
    # Every sub-interval on every interior or boundary line must sum cleanly to zero.
    lines: dict[tuple[float, float, float], list[tuple[float, float, float]]] = {}

    def _add_edge(p0: np.ndarray, p1: np.ndarray, weight: float) -> None:
        vec = p1 - p0
        L = float(math.hypot(float(vec[0]), float(vec[1])))
        if L <= 1e-14:
            return
        u = vec / L
        # Canonical tangent sign along line
        if abs(u[0]) > 1e-9:
            sgn = 1.0 if u[0] > 0 else -1.0
        else:
            sgn = 1.0 if u[1] > 0 else -1.0
        tau = u * sgn
        c = float(tau[0] * p0[1] - tau[1] * p0[0])
        key = (round(float(tau[0]), 9), round(float(tau[1]), 9), round(c, 9))
        t0 = float(np.dot(p0, tau))
        t1 = float(np.dot(p1, tau))
        a, b = (t0, t1) if t0 <= t1 else (t1, t0)
        lines.setdefault(key, []).append((a, b, weight * sgn))

    nv = v.shape[1]
    for e_pts in v:
        for k in range(nv):
            _add_edge(e_pts[k], e_pts[(k + 1) % nv], +1.0)

    # Cancel exterior domain boundary edges using true domain CCW orientation
    bdr_ok = True
    for seg in mesh.bdr_xy:
        svec = seg[1] - seg[0]
        sL = float(math.hypot(float(svec[0]), float(svec[1])))
        if sL <= 1e-14:
            bdr_ok = False
            continue
        if problem.domain == "lshape_2d":
            hit = _match_lshape_side(seg)
            if hit is None:
                bdr_ok = False
                continue
            nx, ny = hit[1]
            ccw_t = np.array([-ny, nx], dtype=np.float64)
        else:
            # Find which unit-square side contains seg
            matched_lbl = None
            for lbl, (ax, val) in SIDE_GEOMETRY.items():
                if np.all(np.abs(seg[:, ax] - val) <= GEOM_TOL):
                    matched_lbl = lbl
                    break
            if matched_lbl is None:
                bdr_ok = False
                continue
            nx, ny = SIDE_NORMAL[matched_lbl]
            ccw_t = np.array([-ny, nx], dtype=np.float64)
        p0, p1 = (seg[0], seg[1]) if float(np.dot(svec, ccw_t)) >= 0.0 else (seg[1], seg[0])
        _add_edge(p0, p1, -1.0)

    max_unbalanced_len = 0.0
    for segs in lines.values():
        pts = sorted(pt for a, b, _ in segs for pt in (a, b))
        merged: list[float] = []
        for p in pts:
            if not merged or abs(p - merged[-1]) > 1e-10:
                merged.append(p)
        for lo, hi in zip(merged, merged[1:]):
            if hi - lo <= 1e-10:
                continue
            mid = 0.5 * (lo + hi)
            net = sum(w for a, b, w in segs if a <= mid <= b)
            if abs(net) > 1e-9:
                max_unbalanced_len += (hi - lo)
    watertight_ok = bdr_ok and (max_unbalanced_len <= 1e-9)
    checks.append(AmrCheck(
        "amr_watertight_edge_partition", watertight_ok, "MESH_MISMATCH",
        "sum of active element edges minus domain CCW boundary along all lines (catches holes/overlaps)",
        max_unbalanced_len, 1e-9,
    ))


def _check_boundary_coverage(mesh: MeshData, problem: ProblemSpec, checks: list[AmrCheck]) -> None:
    """Gate A boundary checks using frozen ProblemSpec attributes and domain geometry."""
    by_attr = {b.attribute: b for b in problem.boundaries}
    mesh_attrs = set(int(a) for a in np.unique(mesh.bdr_attr))
    undeclared = sorted(mesh_attrs - set(by_attr))
    missing = sorted(set(by_attr) - mesh_attrs)
    checks.append(AmrCheck(
        "boundary_attribute_coverage", not undeclared and not missing, "BOUNDARY_COVERAGE",
        f"undeclared mesh attributes {undeclared}; declared but absent {missing}",
    ))

    if problem.domain == "lshape_2d":
        edge_len = {i: 0.0 for i in range(len(_LSHAPE_EDGES))}
        edge_ints: dict[int, list[list[float]]] = {i: [] for i in range(len(_LSHAPE_EDGES))}
        bad_geom = []
        for attr, seg in zip(mesh.bdr_attr, mesh.bdr_xy):
            if int(attr) not in by_attr:
                bad_geom.append(int(attr))
                continue
            hit = _match_lshape_side(seg)
            if hit is None:
                bad_geom.append(int(attr))
                continue
            idx, _ = hit
            axis = 1 - _LSHAPE_EDGES[idx][0]
            edge_len[idx] += float(np.linalg.norm(seg[1] - seg[0]))
            edge_ints[idx].append(sorted(seg[:, axis]))
        checks.append(AmrCheck(
            "boundary_label_geometry", not bad_geom, "BOUNDARY_LABEL_MISMATCH",
            f"segments outside L-shape perimeter: {sorted(set(bad_geom))}",
        ))
        expected_total = 4.0
        total_len = sum(edge_len.values())
        len_err = abs(total_len - expected_total)
        checks.append(AmrCheck(
            "boundary_side_length", len_err <= 1e-10, "BOUNDARY_COVERAGE",
            "max |L-shape perimeter - 4|", len_err, 1e-10,
        ))
        bad_edges = []
        for idx, (_, _, rmin, rmax, *_rest) in enumerate(_LSHAPE_EDGES):
            cur = rmin
            for a0, a1 in sorted(edge_ints[idx]):
                if abs(a0 - cur) > 1e-10:
                    bad_edges.append(idx)
                    break
                cur = a1
            else:
                if abs(cur - rmax) > 1e-10:
                    bad_edges.append(idx)
        checks.append(AmrCheck(
            "boundary_partition", not bad_edges, "BOUNDARY_COVERAGE",
            f"L-shape edges with gaps or overlaps: {bad_edges}",
        ))
        return

    # Standard unit_square_2d 4-side checks
    label_bad = []
    side_len = {lbl: 0.0 for lbl in SIDE_GEOMETRY}
    intervals: dict[str, list[list[float]]] = {lbl: [] for lbl in SIDE_GEOMETRY}
    for a, seg in zip(mesh.bdr_attr, mesh.bdr_xy):
        bc = by_attr.get(int(a))
        if bc is None:
            continue
        ax, val = SIDE_GEOMETRY[bc.label]
        if not np.all(np.abs(seg[:, ax] - val) < GEOM_TOL):
            label_bad.append(int(a))
        else:
            side_len[bc.label] += float(np.linalg.norm(seg[1] - seg[0]))
            other = 1 - ax
            intervals[bc.label].append(sorted(seg[:, other]))
    checks.append(AmrCheck(
        "boundary_label_geometry", not label_bad, "BOUNDARY_LABEL_MISMATCH",
        f"attributes whose segments are not on their declared side: {sorted(set(label_bad))}",
    ))
    len_err = max(abs(v - 1.0) for v in side_len.values())
    checks.append(AmrCheck(
        "boundary_side_length", len_err <= 1e-10, "BOUNDARY_COVERAGE",
        "max |declared side length - 1|", len_err, 1e-10,
    ))
    bad_sides = []
    for label, segs in intervals.items():
        cur = 0.0
        for a0, a1 in sorted(segs):
            if abs(a0 - cur) > 1e-10:
                bad_sides.append(label)
                break
            cur = a1
        else:
            if abs(cur - 1.0) > 1e-10:
                bad_sides.append(label)
    checks.append(AmrCheck(
        "boundary_partition", not bad_sides, "BOUNDARY_COVERAGE",
        f"sides with missing or overlapping segments: {bad_sides}",
    ))


def _detect_domain_dirichlet(mesh: MeshData, coords: np.ndarray, problem: ProblemSpec) -> np.ndarray:
    """Identify full DOFs lying geometrically on Dirichlet boundaries declared in frozen ProblemSpec."""
    on_dir = np.zeros(coords.shape[0], dtype=bool)
    if problem.domain == "lshape_2d":
        dir_attrs = set(problem.dirichlet_attributes())
        for attr, seg in zip(mesh.bdr_attr, mesh.bdr_xy):
            if int(attr) not in dir_attrs:
                continue
            p0, p1 = seg[0], seg[1]
            v = p1 - p0
            L2 = float(np.dot(v, v))
            if L2 <= 1e-28:
                continue
            L = math.sqrt(L2)
            rel = coords - p0[None, :]
            t = (rel @ v) / L2
            perp = np.abs(rel[:, 0] * v[1] - rel[:, 1] * v[0]) / L
            on_dir |= (perp <= GEOM_TOL) & (t >= -GEOM_TOL) & (t <= 1.0 + GEOM_TOL)
        return on_dir

    for b in problem.boundaries:
        if b.kind == "dirichlet":
            ax, val = SIDE_GEOMETRY[b.label]
            on_dir |= np.abs(coords[:, ax] - val) < GEOM_TOL
    return on_dir


def _assemble_amr_neumann_load(
    mesh: MeshData,
    basis: LocalBasis,
    problem: ProblemSpec,
    grad_fn: Callable,
    k_fn: Callable,
    m: int,
    F_full: np.ndarray,
) -> None:
    """Assemble exact outward conormal boundary flux integral int_{Gamma_N} (kappa grad(u).n) phi ds."""
    neu_map = {b.attribute: b for b in problem.boundaries if b.kind == "neumann"}
    if not neu_map:
        return

    # Index active element edges -> owning element
    verts = np.round(mesh.verts, 11)
    edge_owner: dict[frozenset, int] = {}
    nv = verts.shape[1]
    for e in range(verts.shape[0]):
        for k in range(nv):
            a = tuple(map(float, verts[e, k]))
            b = tuple(map(float, verts[e, (k + 1) % nv]))
            edge_owner[frozenset((a, b))] = e

    g1, w1 = gauss_01(m)
    for attr, seg in zip(mesh.bdr_attr, mesh.bdr_xy):
        bc = neu_map.get(int(attr))
        if bc is None:
            continue
        key = frozenset((tuple(map(float, np.round(seg[0], 11))), tuple(map(float, np.round(seg[1], 11)))))
        e = edge_owner.get(key)
        if e is None:
            raise ArtifactError("AMR Neumann boundary segment does not match any active element edge")
        pts = seg[0][None, :] + g1[:, None] * (seg[1] - seg[0])[None, :]
        L = float(np.linalg.norm(seg[1] - seg[0]))
        ref = inverse_map(mesh.geom, mesh.verts[e:e + 1], pts[None, :, :])[0]
        M, _, _ = eval_monomials(basis.exps, ref[:, 0], ref[:, 1])
        phi = M @ basis.coeffs[e]
        gx, gy = grad_fn(pts[:, 0], pts[:, 1])
        if problem.domain == "lshape_2d":
            hit = _match_lshape_side(seg)
            if hit is None:
                raise ArtifactError("L-shape Neumann segment outside outer perimeter")
            nx, ny = hit[1]
        else:
            nx, ny = SIDE_NORMAL[bc.label]
        if problem.equation == ANISOTROPIC:
            gn = 1000.0 * gx * nx + gy * ny
        else:
            gn = k_fn(pts[:, 0], pts[:, 1]) * (gx * nx + gy * ny)
        F_full[mesh.elem_dofs[e]] += (w1 * L * gn) @ phi


def verify_amr_step(
    step_dir: Path,
    problem: ProblemSpec,
    validation: ValidationSpec | None = None,
    execution: ExecutionSpec | None = None,
    *,
    expected_step: int | None = None,
    require_reference_solve: bool = True,
) -> AmrStepReport:
    """Verify one step_XX directory strictly against frozen ProblemSpec/ExecutionSpec/ValidationSpec."""
    step_dir = Path(step_dir)
    val = validation or ValidationSpec()
    checks: list[AmrCheck] = []
    rpath = step_dir / "result.json"
    if not rpath.is_file():
        return AmrStepReport(0, None, None, None, 0, None, None, None, None, None, None, None,
                             [AmrCheck("result_json_present", False, "MISSING_ARTIFACT", "result.json missing")])
    try:
        res = json.loads(rpath.read_text())
        if not isinstance(res, dict):
            raise ValueError("expected a JSON object")
    except (OSError, ValueError) as exc:
        return AmrStepReport(0, None, None, None, 0, None, None, None, None, None, None, None,
                             [AmrCheck("result_json_parse", False, "MALFORMED_ARTIFACT", str(exc))])

    step = int(res.get("amr_step", 0))
    order = execution.polynomial_order if execution is not None else int(res.get("order", 1))
    exp_step = expected_step if expected_step is not None else step

    # Verify solver claims against frozen ProblemSpec + ExecutionSpec (never trust solver claims blindly)
    claimed = {
        "amr_step": (res.get("amr_step"), exp_step),
        "case_exact_solution": (compact_expression(res.get("case_exact_solution")), compact_expression(problem.exact_solution)),
        "case_source": (compact_expression(res.get("case_source")), compact_expression(problem.source)),
        "case_equation": (compact_expression(res.get("case_equation")), compact_expression(problem.equation)),
        "case_diffusivity": (compact_expression(res.get("case_diffusivity", "1")), compact_expression(problem.kappa)),
        "dirichlet_attributes": (_attributes(res.get("dirichlet_attributes")), problem.dirichlet_attributes()),
        "neumann_attributes": (_attributes(res.get("neumann_attributes")), problem.neumann_attributes()),
        "order": (res.get("order"), order),
    }
    if "case_domain" in res:
        claimed["case_domain"] = (res.get("case_domain"), problem.domain)
    if execution is not None:
        claimed.update({
            "mesh_type": (res.get("mesh_type"), execution.mesh_type),
            "mesh_topology": (res.get("mesh_topology"), "amr"),
            "assembly": (res.get("assembly"), execution.assembly),
            "solver": (res.get("solver"), execution.solver),
            "rel_tol": (res.get("rel_tol"), execution.rel_tol),
            "max_iter": (res.get("max_iter"), execution.max_iter),
        })
    bad_claims = [k for k, (got, want) in claimed.items() if got != want or isinstance(got, bool)]
    checks.append(AmrCheck(
        "solver_claims_match_spec", not bad_claims, "CLAIM_SPEC_MISMATCH",
        f"mismatched fields: {bad_claims}" if bad_claims else "solver claims match frozen spec",
    ))

    # Quadrature orders and CG stopping invariants
    expected_rhs_q = max(2 * order + 4, 20)
    q_claims = [res.get("rhs_quadrature_order"), res.get("error_quadrature_order")]
    q_ok = all(type(q_) is int and q_ == expected_rhs_q for q_ in q_claims)
    checks.append(AmrCheck(
        "quadrature_orders", q_ok, "QUADRATURE_CLAIM_MISMATCH",
        f"reported RHS/error orders {q_claims}, expected {expected_rhs_q}",
    ))
    if problem.kappa != "1":
        sq = res.get("stiffness_quadrature_order")
        checks.append(AmrCheck(
            "variable_stiffness_quadrature_order", type(sq) is int and sq == 2 * order + 3,
            "QUADRATURE_CLAIM_MISMATCH", f"reported weighted stiffness order {sq}, expected {2 * order + 3}",
        ))

    conv = res.get("solver_converged")
    checks.append(AmrCheck(
        "solver_reported_convergence", type(conv) is bool and (conv or not val.require_solver_converged),
        "SOLVER_NOT_CONVERGED", f"solver_converged={conv}",
    ))
    iters = res.get("solver_iterations")
    init_n, fin_n, abs_t = res.get("solver_initial_norm"), res.get("solver_final_norm"), res.get("abs_tol")
    max_it = execution.max_iter if execution is not None else int(res.get("max_iter", 20000))
    rtol_val = execution.rel_tol if execution is not None else float(res.get("rel_tol", 1e-12))
    cg_ok = (
        type(iters) is int and 0 <= iters <= max_it
        and _finite(init_n) and init_n >= 0 and _finite(fin_n) and fin_n >= 0
        and _finite(abs_t) and abs_t == 0
    )
    if cg_ok and conv is True:
        cg_ok = fin_n <= math.nextafter(rtol_val * init_n, math.inf) and (iters != 0 or fin_n == init_n)
    checks.append(AmrCheck(
        "cg_stopping_record", cg_ok, "CG_RECORD_INCONSISTENT",
        f"iterations={iters}, initial={init_n}, final={fin_n}, abs_tol={abs_t}",
    ))

    try:
        mesh = load_mesh_data(step_dir)
    except (ArtifactError, OSError, UnicodeError) as exc:
        checks.append(AmrCheck("artifacts_load", False, "MALFORMED_ARTIFACT", str(exc)))
        return AmrStepReport(step, None, None, None, 0, None, None, None,
                             res.get("self_reported_l2_error"), res.get("self_reported_h1_semi_error"),
                             None, res.get("conforming"), checks)

    N = mesh.values.size
    E = mesh.verts.shape[0]
    finite = bool(np.all(np.isfinite(mesh.values)))
    checks.append(AmrCheck(
        "finite_solution", finite, "NON_FINITE_SOLUTION",
        f"non-finite={int(np.sum(~np.isfinite(mesh.values)))}",
    ))
    if not finite:
        return AmrStepReport(step, E, N, None, 0, None, None, None,
                             res.get("self_reported_l2_error"), res.get("self_reported_h1_semi_error"),
                             None, res.get("conforming"), checks)

    coords = np.full((N, 2), np.nan)
    flat_d = mesh.elem_dofs.ravel()
    coords[flat_d] = mesh.elem_node_xy.reshape(-1, 2)
    if np.isnan(coords).any():
        checks.append(AmrCheck("dof_coordinates", False, "MALFORMED_ARTIFACT", "unreferenced global DOFs"))
        return AmrStepReport(step, E, N, None, 0, None, None, None,
                             res.get("self_reported_l2_error"), res.get("self_reported_h1_semi_error"),
                             None, res.get("conforming"), checks)
    spread = float(np.max(np.abs(coords[flat_d] - mesh.elem_node_xy.reshape(-1, 2))))
    checks.append(AmrCheck("dof_coordinate_consistency", spread <= 1e-12, "MALFORMED_ARTIFACT",
                           "max coordinate mismatch across elements", spread, 1e-12))

    # Mesh geometry & watertight domain partition before hanging-node detection
    geom_ok = (execution is None or mesh.geom == execution.mesh_type)
    elem_claim_ok = geom_ok and type(res.get("num_elements")) is int and res.get("num_elements") == E
    bdr_claim_ok = type(res.get("num_boundary_elements")) is int and res.get("num_boundary_elements") == mesh.bdr_xy.shape[0]
    checks.append(AmrCheck("element_and_boundary_count", elem_claim_ok and bdr_claim_ok, "MESH_MISMATCH",
                           f"elems={E} (claimed {res.get('num_elements')}), bdr={mesh.bdr_xy.shape[0]}"))

    _check_domain_and_partition(mesh, problem, checks)
    _check_boundary_coverage(mesh, problem, checks)

    try:
        constraints, true_dofs, C = detect_hanging_constraints(mesh, coords, order)
    except ArtifactError as exc:
        checks.append(AmrCheck("hanging_node_constraints", False, "HANGING_NODE_DISCONTINUITY", str(exc)))
        return AmrStepReport(step, E, N, None, 0, None, None, None,
                             res.get("self_reported_l2_error"), res.get("self_reported_h1_semi_error"),
                             None, res.get("conforming"), checks)

    dof_claim_ok = (
        type(res.get("num_dofs")) is int and res.get("num_dofs") == N
        and type(res.get("num_true_dofs")) is int and res.get("num_true_dofs") == len(true_dofs)
    )
    checks.append(AmrCheck("dof_count", dof_claim_ok, "MESH_MISMATCH",
                           f"full={N} (claimed {res.get('num_dofs')}), true={len(true_dofs)} (claimed {res.get('num_true_dofs')})"))

    conf_val = res.get("conforming")
    conf_ok = type(conf_val) is bool and not (len(constraints) > 0 and conf_val is True)
    checks.append(AmrCheck("conforming_claim", conf_ok, "MESH_MISMATCH",
                           f"reported conforming={conf_val} with {len(constraints)} hanging constraints"))

    if constraints:
        u_interp = C @ mesh.values[true_dofs]
        slave_ids = sorted(constraints.keys())
        hang_err = float(np.max(np.abs(mesh.values[slave_ids] - u_interp[slave_ids])))
    else:
        hang_err = 0.0
    checks.append(AmrCheck("hanging_node_continuity", hang_err <= HANGING_TOL, "HANGING_NODE_DISCONTINUITY",
                           f"max T-junction trace jump across {len(constraints)} hanging DOFs",
                           hang_err, HANGING_TOL))

    # Compile frozen ProblemSpec expressions + exact AST gradients
    u_fn = expr.compile_expr(problem.exact_solution)
    grad_fn = expr.compile_exact_gradient(problem.exact_solution)
    f_fn = expr.compile_expr(problem.source)
    k_fn = expr.compile_expr(problem.kappa)

    on_dir = _detect_domain_dirichlet(mesh, coords, problem)
    u_dir_exact = u_fn(coords[on_dir, 0], coords[on_dir, 1]) if on_dir.any() else np.empty(0)
    dmax = float(np.max(np.abs(mesh.values[on_dir] - u_dir_exact))) if on_dir.any() else 0.0
    checks.append(AmrCheck("dirichlet_nodal_values", dmax <= val.dirichlet_nodal_abs_tol, "DIRICHLET_VIOLATION",
                           f"max |u_h - u_exact| over {int(on_dir.sum())} Dirichlet DOFs", dmax, val.dirichlet_nodal_abs_tol))

    m_indep = max(order + 6, 16)
    try:
        basis = build_local_basis(mesh, order)
        q = element_quadrature(mesh, basis, m_indep)
    except (ArtifactError, np.linalg.LinAlgError) as exc:
        checks.append(AmrCheck("local_basis", False, "MALFORMED_ARTIFACT", str(exc)))
        return AmrStepReport(step, E, N, len(true_dofs), len(constraints), hang_err, None, None,
                             res.get("self_reported_l2_error"), res.get("self_reported_h1_semi_error"),
                             None, res.get("conforming"), checks)

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

    # Self-report gate at strict ValidationSpec tolerance (1e-6):
    # For singular domains (non-polynomial r^{-1/3}), verify self-report using exact
    # declared Gauss order m_self = expected_rhs_q // 2 + 1 (after quadrature claim check).
    if problem.domain == "lshape_2d":
        m_self = (expected_rhs_q // 2) + 1
        q_self = element_quadrature(mesh, basis, m_self)
        uh_s = np.einsum("eqk,ek->eq", q_self.phi, U)
        guh_s = np.einsum("eqki,ek->eqi", q_self.gphi, U)
        xs, ys = q_self.X[..., 0], q_self.X[..., 1]
        gxs, gys = grad_fn(xs, ys)
        l2_cmp = float(np.sqrt(np.sum(q_self.wdet * (uh_s - u_fn(xs, ys)) ** 2)))
        h1_cmp = float(np.sqrt(np.sum(q_self.wdet * ((guh_s[..., 0] - gxs) ** 2 + (guh_s[..., 1] - gys) ** 2))))
    else:
        l2_cmp, h1_cmp = l2, h1

    for name, mine, theirs in (("l2", l2_cmp, res.get("self_reported_l2_error")),
                               ("h1_semi", h1_cmp, res.get("self_reported_h1_semi_error"))):
        ok_sr = _finite(theirs)
        abs_floor = 1e-14 if name == "l2" else 2e-13
        rel_sr = abs(mine - theirs) / max(abs(mine), 1e-300) if ok_sr else math.inf
        tol_sr = max(val.self_report_rel_tol * abs(mine), abs_floor)
        checks.append(AmrCheck(
            f"self_report_{name}", ok_sr and abs(mine - theirs) <= tol_sr,
            "SELF_REPORT_MISMATCH", f"verifier={mine:.10e}, solver={theirs}",
            rel_sr if math.isfinite(rel_sr) else None, val.self_report_rel_tol,
        ))

    ref_rel_diff = None
    if require_reference_solve and val.require_reference_solve:
        nloc = mesh.elem_dofs.shape[1]
        kappa_vals = None if problem.kappa == "1" else k_fn(x, y)
        weights = q.wdet if kappa_vals is None else (q.wdet * kappa_vals)
        Ke = np.einsum("eq,eqid,eqjd->eij", weights, q.gphi, q.gphi)
        Fe = np.einsum("eq,eq,eqi->ei", q.wdet, f_fn(x, y), q.phi)
        rows = np.repeat(mesh.elem_dofs, nloc, axis=1).ravel()
        cols = np.tile(mesh.elem_dofs, (1, nloc)).ravel()
        K_full = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(N, N)).tocsr()
        F_full = np.bincount(mesh.elem_dofs.ravel(), weights=Fe.ravel(), minlength=N)

        try:
            _assemble_amr_neumann_load(mesh, basis, problem, grad_fn, k_fn, m_indep, F_full)
            K_con = (C.T @ K_full @ C).tocsr()
            F_con = C.T @ F_full
            true_arr = np.asarray(true_dofs, dtype=np.int64)
            true_dir = on_dir[true_arr]
            true_free = ~true_dir
            u_true = np.zeros(true_arr.size, dtype=np.float64)
            u_true[true_dir] = u_fn(coords[true_arr[true_dir], 0], coords[true_arr[true_dir], 1])
            rhs = F_con[true_free] - K_con[true_free][:, true_dir] @ u_true[true_dir]
            K_free = K_con[true_free][:, true_free].tocsr()
            u_free_sol, _ = solve_reference_system(
                K_free, rhs, method=val.reference_solve_method,
                amg_threshold_dofs=val.reference_solve_amg_threshold_dofs,
            )
            u_true[true_free] = u_free_sol
            u_ref = C @ u_true
            ref_rel_diff = float(np.max(np.abs(u_ref - mesh.values)) / max(np.max(np.abs(u_ref)), 1e-300))
            checks.append(AmrCheck(
                "reference_solve_agreement", ref_rel_diff <= val.reference_solve_rel_tol,
                "REFERENCE_MISMATCH", "constrained C^T K C reference solve vs MFEM",
                ref_rel_diff, val.reference_solve_rel_tol,
            ))
        except ArtifactError as exc:
            checks.append(AmrCheck("reference_neumann_assembly", False, "BOUNDARY_COVERAGE", str(exc)))
        except ReferenceSolverError as exc:
            checks.append(AmrCheck("reference_solve", False, "REFERENCE_SOLVER_FAILED", str(exc)))

    return AmrStepReport(
        step=step, num_elements=E, num_dofs=N, num_true_dofs_inferred=len(true_dofs),
        hanging_dof_count=len(constraints), max_hanging_continuity_error=hang_err,
        l2_error=l2, h1_semi_error=h1,
        self_reported_l2_error=res.get("self_reported_l2_error"),
        self_reported_h1_semi_error=res.get("self_reported_h1_semi_error"),
        reference_rel_diff=ref_rel_diff, conforming=res.get("conforming"), checks=checks,
    )


def verify_amr_run(
    amr_dir: Path,
    problem: ProblemSpec,
    validation: ValidationSpec | None = None,
    execution: ExecutionSpec | None = None,
    *,
    require_reference_solve: bool = True,
    require_run_record: bool = False,
) -> dict:
    """Verify an entire sequence of step_XX subdirectories against frozen specs."""
    if problem is None:
        raise ValueError("frozen ProblemSpec is required for independent AMR verification")
    amr_dir = Path(amr_dir)
    val = validation or ValidationSpec()
    study_checks: list[AmrCheck] = []
    if require_run_record:
        try:
            record = json.loads((amr_dir / "run_record.json").read_text())
            proc_ok = (
                isinstance(record, dict) and type(record.get("exit_code")) is int
                and record["exit_code"] == 0 and record.get("exit_meaning") == "OK"
                and record.get("timed_out") is False
            )
            study_checks.append(AmrCheck("process_completed", proc_ok, "SOLVER_EXIT",
                                         "AMR solver exit code and timeout status"))
        except (OSError, ValueError) as exc:
            study_checks.append(AmrCheck("process_completed", False, "MALFORMED_ARTIFACT", str(exc)))

    step_dirs = sorted(p for p in amr_dir.iterdir() if p.is_dir() and p.name.startswith("step_"))
    reports = [
        verify_amr_step(
            sd, problem, val, execution, expected_step=idx, require_reference_solve=require_reference_solve
        )
        for idx, sd in enumerate(step_dirs)
    ]
    step_names = [sd.name for sd in step_dirs]
    seq_ok = bool(step_dirs) and step_names == [f"step_{i:02d}" for i in range(len(step_dirs))]
    study_checks.append(AmrCheck("amr_step_sequence", seq_ok, "MISSING_ARTIFACT",
                                 f"found {step_names}"))
    if len(reports) >= 2 and _finite(reports[0].h1_semi_error) and _finite(reports[-1].h1_semi_error):
        net_decay = reports[-1].h1_semi_error < reports[0].h1_semi_error
    elif len(reports) == 1:
        net_decay = True
    else:
        net_decay = False
    study_checks.append(AmrCheck("net_h1_error_decay", net_decay, "NO_AMR_ERROR_REDUCTION",
                                 "finest AMR step H1-seminorm error < coarse step 0"))

    accuracy = None
    if val.max_l2_error_finest is not None and reports and _finite(reports[-1].l2_error):
        accuracy = AmrCheck(
            "accuracy_target_finest_l2", reports[-1].l2_error <= val.max_l2_error_finest,
            "ACCURACY_NOT_MET", f"finest AMR step L2 error on step {reports[-1].step}",
            reports[-1].l2_error, val.max_l2_error_finest,
        )

    hard_fail = [c for c in study_checks if not c.passed] + [c for r in reports for c in r.checks if not c.passed]
    codes = sorted({c.code for c in hard_fail})
    if hard_fail:
        verdict = REJECTED
    elif accuracy is not None and not accuracy.passed:
        verdict = RETRY_ELIGIBLE
        codes = ["ACCURACY_NOT_MET"]
    else:
        verdict = ACCEPTED

    return {
        "verifier": "verifier.amr",
        "verifier_version": VERIFIER_VERSION,
        "verdict": verdict,
        "failure_codes": codes,
        "statement": ("numerical (code) AMR verification passed for frozen ProblemSpec"
                      if verdict == ACCEPTED else "numerical AMR verification did NOT pass"),
        "study_checks": [asdict(c) for c in study_checks],
        "accuracy_check": asdict(accuracy) if accuracy else None,
        "step_count": len(reports),
        "net_h1_error_decay": net_decay,
        "steps": [asdict(r) | {"ok": r.ok} for r in reports],
    }


def verify_amr_attempt(attempt_dir: Path, *, require_run_records: bool = False) -> dict:
    """Verify one controller attempt_NN directory containing amr/step_XX artifacts."""
    from .verifier import SHARED_ASSUMPTIONS, check_mms_consistency

    attempt_dir = Path(attempt_dir)
    run_dir = attempt_dir.parent.parent
    frozen = run_dir / "frozen"
    try:
        problem = ProblemSpec.model_validate_json((frozen / "problem.json").read_text())
        validation = ValidationSpec.model_validate_json((frozen / "validation.json").read_text())
        execution = ExecutionSpec.model_validate_json((attempt_dir / "execution.json").read_text())
        validation.check_problem(problem)
        mms = check_mms_consistency(problem, validation.mms_consistency_rel_tol)
    except (OSError, ValueError) as exc:
        code = "MISSING_ARTIFACT" if isinstance(exc, FileNotFoundError) else "MALFORMED_ARTIFACT"
        return {
            "verifier": "verifier.amr", "verifier_version": VERIFIER_VERSION, "verdict": REJECTED,
            "failure_codes": [code], "statement": "AMR verification inputs failed validation",
            "study_checks": [asdict(AmrCheck("attempt_inputs", False, code, str(exc)))],
            "accuracy_check": None, "levels": [], "steps": [], "convergence": {"pairs": []},
            "shared_assumptions": SHARED_ASSUMPTIONS, "input_sha256": {},
        }

    if not mms.passed:
        return {
            "verifier": "verifier.amr", "verifier_version": VERIFIER_VERSION, "verdict": REJECTED,
            "failure_codes": [mms.code], "statement": "MMS consistency check failed",
            "study_checks": [asdict(mms)], "accuracy_check": None, "levels": [], "steps": [],
            "convergence": {"pairs": []}, "shared_assumptions": SHARED_ASSUMPTIONS, "input_sha256": {},
        }

    amr_dir = attempt_dir / "amr"
    res = verify_amr_run(
        amr_dir, problem, validation, execution,
        require_reference_solve=validation.require_reference_solve,
        require_run_record=require_run_records,
    )
    steps = res["steps"]
    levels = [
        {
            "n": int(s["step"]),
            "level_dir": str(amr_dir / f"step_{s['step']:02d}"),
            "l2_error": s["l2_error"],
            "h1_semi_error": s["h1_semi_error"],
            "self_reported_l2_error": s["self_reported_l2_error"],
            "self_reported_h1_semi_error": s["self_reported_h1_semi_error"],
            "reference_rel_diff": s["reference_rel_diff"],
            "num_dofs": s["num_true_dofs_inferred"] or s["num_dofs"],
            "num_true_dofs": s["num_true_dofs_inferred"],
            "hanging_dof_count": s["hanging_dof_count"],
            "max_hanging_continuity_error": s["max_hanging_continuity_error"],
            "checks": s["checks"],
            "ok": s["ok"],
        }
        for s in steps
    ]

    inputs = {
        "frozen/problem.json": sha256_file(frozen / "problem.json"),
        "frozen/validation.json": sha256_file(frozen / "validation.json"),
        "execution.json": sha256_file(attempt_dir / "execution.json"),
    }
    if (run_dir / ARCHIVED_MFEM_MESH).is_file():
        inputs[ARCHIVED_MFEM_MESH] = sha256_file(run_dir / ARCHIVED_MFEM_MESH)
    for rel, digest in artifact_hashes(amr_dir).items():
        if rel.endswith((".json", ".txt")):
            inputs[f"amr/{rel}"] = digest

    return {
        "verifier": "verifier.amr",
        "verifier_version": VERIFIER_VERSION,
        "verdict": res["verdict"],
        "failure_codes": res["failure_codes"],
        "statement": res["statement"],
        "study_checks": [asdict(mms), *res["study_checks"]],
        "accuracy_check": res["accuracy_check"],
        "levels": levels,
        "steps": steps,
        "step_count": res["step_count"],
        "net_h1_error_decay": res["net_h1_error_decay"],
        "convergence": {"mode": "amr_zz", "pairs": [], "net_h1_error_decay": res["net_h1_error_decay"]},
        "shared_assumptions": SHARED_ASSUMPTIONS,
        "input_sha256": inputs,
    }
