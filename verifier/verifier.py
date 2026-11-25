"""Independent verification of MFEM scalar diffusion runs against a frozen spec.

Inputs per attempt directory (written by the runner, read-only here):
    frozen/problem.json, frozen/validation.json, execution.json
    levels/n<NNNN>/{result.json, elements.txt, element_dofs.txt,
                    dof_values.txt, boundary.txt}

Output: a typed verdict (ACCEPTED / RETRY_ELIGIBLE / REJECTED) with every
check, its measured value, its threshold, and failure codes.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sciagent.spec import (ANISOTROPIC, GAUSSIAN_EXACT, LAYERED, LAYERED_EXACT, LAYERED_LEVELS, POISSON,
                           SIDE_GEOMETRY, SIDE_NORMAL, ExecutionSpec, ProblemSpec, StudySpec,
                           ValidationSpec, compact_expression)
from sciagent.evidence import artifact_hashes, sha256_file
from sciagent.profiling import mark, phase
from sciagent.gmsh import ARCHIVED_MESH, CartesianGmsh, UnstructuredGmsh, inspect_gmsh, inspect_gmsh_unstructured, read_mesh

from . import expr, layered
from .fem import (ArtifactError, MeshData, build_local_basis, cartesian_partition,
                  element_quadrature, eval_monomials, evaluate_quadrature_slice, gauss_01,
                  inverse_map, lattice_indices, load_mesh_data, reference_quadrature,
                  repeated_element_period)
from .reference_solver import ReferenceSolverError, solve_reference_system

VERIFIER_VERSION = "0.6.0"
GEOM_TOL = 1e-12

ACCEPTED = "ACCEPTED"
RETRY_ELIGIBLE = "RETRY_ELIGIBLE"
REJECTED = "REJECTED"

SHARED_ASSUMPTIONS = [
    "Solver and verifier read the same frozen ProblemSpec; an exact solution or source that is "
    "physically wrong but mutually consistent would pass (this is code verification, not model "
    "validation).",
    "The independent reference solve uses the same standard Galerkin weak form; it checks the "
    "implementation of that discretization, not the choice of discretization.",
    "The verifier checks the exported Cartesian cell partition and supported P1/P2 node "
    "lattice. Its integration and reference solve still use the same exported coordinates.",
    "The local basis is assumed to be nodal (Lagrange) at the exported node points, which holds "
    "for MFEM's default H1 Gauss-Lobatto basis; any violation surfaces as a self-report mismatch.",
]


@dataclass
class Check:
    name: str
    passed: bool
    code: str            # failure code used when passed is False
    detail: str = ""
    value: float | None = None
    threshold: float | None = None


@dataclass
class LevelReport:
    n: int
    level_dir: str
    checks: list[Check] = field(default_factory=list)
    l2_error: float | None = None
    h1_semi_error: float | None = None
    self_reported_l2_error: float | None = None
    self_reported_h1_semi_error: float | None = None
    reference_rel_diff: float | None = None
    num_dofs: int | None = None
    solver_iterations: int | None = None
    injected_fault_label: str | None = None  # reported, never used for decisions

    def add(self, c: Check) -> Check:
        self.checks.append(c)
        return c

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)


def _finite(v) -> bool:
    try:
        return type(v) in (int, float) and math.isfinite(v)
    except OverflowError:
        return False


def check_mms_consistency(problem: ProblemSpec, rel_tol: float) -> Check:
    """Compare the declared source with the strong operator on the exact field."""
    try:
        u = expr.compile_expr(problem.exact_solution)
        f = expr.compile_expr(problem.source)
    except expr.ExpressionError as exc:
        return Check("mms_expressions_parse", False, "INVALID_EXPRESSION", str(exc))
    if problem.equation == LAYERED:
        x = np.array([0., 0.25, 0.5, 0.75, 1.])
        match = (compact_expression(problem.exact_solution) == LAYERED_EXACT
                 and compact_expression(problem.source) == "0")
        error = float(np.max(np.abs(u(x, x) - layered.field(x))))
        return Check("layered_exact_unit_drop", match and error <= 1e-14, "MMS_INCONSISTENT",
                     "registered solution matches the continuous unit drop with constant a*du/dx", error, 1e-14)
    rng = np.random.default_rng(12345)
    pts = rng.uniform(0.05, 0.95, size=(256, 2))
    x, y = pts.T
    operator = expr.neg_laplacian(u, x, y, diffusion_x=1000.0 if problem.equation == ANISOTROPIC else 1.0)
    description = "(-div(A grad u_exact))" if problem.equation == ANISOTROPIC else "(-lap u_exact)"
    if problem.kappa != "1":
        kappa = expr.compile_expr(problem.kappa)
        kx, ky = expr.gradient(kappa, x, y)
        ux, uy = expr.gradient(u, x, y)
        operator = kappa(x, y) * operator - (kx * ux + ky * uy)
        description = "(-div(kappa*grad u_exact))"
    fv = f(x, y)
    if not (np.all(np.isfinite(operator)) and np.all(np.isfinite(fv))):
        return Check("mms_consistency", False, "MMS_INCONSISTENT", "non-finite expression values")
    err = float(np.max(np.abs(operator - fv) / (1.0 + np.abs(fv))))
    return Check("mms_consistency", err <= rel_tol, "MMS_INCONSISTENT",
                 f"max |{description} - f| / (1+|f|) at 256 fixed random interior points "
                 "(4th-order finite differences, h=1e-3)", err, rel_tol)


def _unique_dof_coords(mesh: MeshData, N: int) -> tuple[np.ndarray, float]:
    coords = np.full((N, 2), np.nan)
    flat_d = mesh.elem_dofs.ravel()
    flat_x = mesh.elem_node_xy.reshape(-1, 2)
    coords[flat_d] = flat_x
    spread = float(np.max(np.abs(coords[flat_d] - flat_x))) if flat_d.size else 0.0
    if np.isnan(coords).any():
        raise ArtifactError("some global dofs are not referenced by any element")
    return coords, spread


def _side_of_points(xy: np.ndarray, label: str) -> np.ndarray:
    c, v = SIDE_GEOMETRY[label]
    return np.abs(xy[..., c] - v) < GEOM_TOL


def _attributes(value):
    if isinstance(value, list) and all(type(a) is int for a in value):
        return sorted(value)
    return None


def _native_claims(res, execution, rep, resolved_peak=False):
    p = execution.polynomial_order
    degree = max(2 * p + 4, 12) if execution.mesh_topology == "unstructured" else (2 * p + 4)
    if resolved_peak:
        degree = max(degree, 20)
    orders = [res.get(key) for key in ("rhs_quadrature_order", "error_quadrature_order")]
    rep.add(Check("quadrature_orders", all(type(q) is int and q == degree for q in orders),
                  "QUADRATURE_CLAIM_MISMATCH", f"reported RHS/error orders {orders}, expected {degree}"))
    iterations = res.get("solver_iterations")
    initial, final = res.get("solver_initial_norm"), res.get("solver_final_norm")
    absolute = res.get("abs_tol")
    valid = (type(iterations) is int and 0 <= iterations <= execution.max_iter
             and _finite(initial) and initial >= 0 and _finite(final) and final >= 0
             and _finite(absolute) and absolute == 0)
    # MFEM CG compares squared preconditioned norms before reporting their roots.
    if valid and res.get("solver_converged") is True:
        limit = math.nextafter(execution.rel_tol * initial, math.inf)
        valid = final <= limit and (iterations != 0 or final == initial)
    rep.add(Check("cg_stopping_record", valid, "CG_RECORD_INCONSISTENT",
                  f"iterations={iterations}, initial={initial}, final={final}, abs_tol={absolute}"))


def _uniform_children(cells, depth):
    """Yield bounded batches of physical corner coordinates under red refinement."""
    if not depth:
        yield cells
        return
    corners = list(cells.transpose(1, 0, 2))
    mid = [(p + q) * 0.5 for p, q in zip(corners, corners[1:] + corners[:1])]
    if len(corners) == 3:
        children = [[corners[i], mid[i], mid[i - 1]] for i in range(3)] + [mid]
    else:
        # The center of the bilinear map is the mean of its four corners.
        center = cells.mean(axis=1)
        children = [[corners[i], mid[i], center, mid[i - 1]] for i in range(4)]
    for child in children:
        yield from _uniform_children(np.stack(child, axis=1), depth - 1)


def _matches_unstructured_source(vertices, source: UnstructuredGmsh, depth: int) -> bool:
    """Match each independently regenerated cell exactly once, allowing cyclic local reordering."""
    from scipy.spatial import cKDTree

    if len(vertices) != source.element_count * 4 ** depth:
        return False
    # Only observed centroids and one seen bit per observed cell scale with the
    # export size. Expected refinements are generated in batches, never materialized.
    observed = cKDTree(vertices.mean(axis=1))
    seen = np.zeros(len(vertices), dtype=bool)
    for start in range(0, source.element_count, 1024):
        coarse = np.asarray(source.coarse_cells[start:start + 1024])
        for expected in _uniform_children(coarse, depth):
            _, ix = observed.query(expected.mean(axis=1), p=np.inf)
            if len(np.unique(ix)) != len(ix) or np.any(seen[ix]):
                return False
            actual = vertices[ix]
            aligned = np.zeros(len(ix), dtype=bool)
            for k in range(vertices.shape[1]):
                aligned |= np.all(np.abs(np.roll(actual, k, axis=1) - expected) <= GEOM_TOL, axis=(1, 2))
            if not np.all(aligned):
                return False
            seen[ix] = True
    return bool(np.all(seen))


def _unstructured_partition_check(mesh: MeshData, rep: LevelReport) -> None:
    """Verify positive element areas, exact unit-square area sum, and conforming edge ownership."""
    v = mesh.verts
    x = v[..., 0]
    y = v[..., 1]
    areas = 0.5 * np.sum(np.roll(x, 1, axis=1) * y - x * np.roll(y, 1, axis=1), axis=1)
    min_area = float(np.min(areas)) if areas.size else 0.0
    total_area = float(np.sum(areas)) if areas.size else 0.0
    area_ok = min_area > 1e-14 and abs(total_area - 1.0) <= 1e-10
    rep.add(Check("unstructured_element_area_partition", area_ok, "MESH_MISMATCH",
                  f"min_area={min_area:.3e}, total_area={total_area:.12f}", total_area, 1.0))

    # Conforming edge topology check: boundary edges belong to 1 element, interior to 2 elements.
    nv = v.shape[1]
    edge_counts: dict[frozenset, int] = {}
    pts_rounded = np.round(v, 11)
    for e_pts in pts_rounded:
        for k in range(nv):
            a = tuple(map(float, e_pts[k]))
            b = tuple(map(float, e_pts[(k + 1) % nv]))
            key = frozenset((a, b))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    bdr_keys = set()
    for seg in np.round(mesh.bdr_xy, 11):
        key = frozenset((tuple(map(float, seg[0])), tuple(map(float, seg[1]))))
        bdr_keys.add(key)
    topo_ok = (len(bdr_keys) == mesh.bdr_xy.shape[0]) and all(
        edge_counts.get(k, 0) == 1 for k in bdr_keys
    ) and all(cnt == 2 for k, cnt in edge_counts.items() if k not in bdr_keys)
    rep.add(Check("unstructured_edge_incidence", topo_ok, "MESH_MISMATCH",
                  "conforming 1-manifold boundary & 2-element interior edges"))


def _refined_diagonals(mesh: MeshData, n: int, source: CartesianGmsh) -> bool:
    """Red refinement of a square split into two triangles preserves its diagonal in every child."""
    if not source.rising_diagonal:
        return mesh.geom == "quad"
    if mesh.geom != "tri":
        return False
    grid = lattice_indices(mesh.verts, n, GEOM_TOL)
    lower = grid.min(axis=1)
    dx, dy = (grid - lower[:, None, :]).transpose(2, 0, 1)
    # A triangle containing SW and NE omits either SE (1) or NW (2).
    missing = 6 - (dx + 2 * dy).sum(axis=1)
    coarse = lower // (n // source.subdivisions)
    indices = coarse[:, 0] + source.subdivisions * coarse[:, 1]
    return bool(np.array_equal(np.isin(missing, [1, 2]), np.asarray(source.rising_diagonal)[indices]))


@phase("verify.solver_claims")
def verify_level(level_dir: Path, problem: ProblemSpec, execution: ExecutionSpec,
                 validation: ValidationSpec, n: int, *, require_run_record: bool = False,
                 external_grid: CartesianGmsh | UnstructuredGmsh | None = None) -> LevelReport:
    rep = LevelReport(n=n, level_dir=str(level_dir))
    level_dir = Path(level_dir)

    # ---- Gate B: execution artifacts and the solver's claims -------------
    rpath = level_dir / "result.json"
    if not rpath.is_file():
        rep.add(Check("result_json_present", False, "MISSING_ARTIFACT", "result.json not found"))
        return rep
    try:
        res = json.loads(rpath.read_text())
        if not isinstance(res, dict):
            raise ValueError("expected a JSON object")
    except (OSError, ValueError) as exc:
        rep.add(Check("result_json_parse", False, "MALFORMED_ARTIFACT", str(exc)))
        return rep
    rep.self_reported_l2_error = res.get("self_reported_l2_error")
    rep.self_reported_h1_semi_error = res.get("self_reported_h1_semi_error")
    rep.num_dofs = res.get("num_dofs")
    rep.solver_iterations = res.get("solver_iterations")
    rep.injected_fault_label = res.get("injected_fault")

    claimed = {
        "n": (res.get("n"), n),
        "order": (res.get("order"), execution.polynomial_order),
        "mesh_type": (res.get("mesh_type"), execution.mesh_type),
        "assembly": (res.get("assembly"), execution.assembly),
        "solver": (res.get("solver"), execution.solver),
        "rel_tol": (res.get("rel_tol"), execution.rel_tol),
        "max_iter": (res.get("max_iter"), execution.max_iter),
        "dirichlet_attributes": (_attributes(res.get("dirichlet_attributes")), problem.dirichlet_attributes()),
        "neumann_attributes": (_attributes(res.get("neumann_attributes")), problem.neumann_attributes()),
        "exact_solution": (compact_expression(res.get("case_exact_solution")), compact_expression(problem.exact_solution)),
        "source": (compact_expression(res.get("case_source")), compact_expression(problem.source)),
        "equation": (res.get("case_equation", POISSON), problem.equation),
    }
    # The original constant-coefficient binary omitted both fields.
    if problem.kappa != "1" or "case_diffusivity" in res:
        claimed["equation"] = (compact_expression(res.get("case_equation")), compact_expression(problem.equation))
        claimed["kappa"] = (compact_expression(res.get("case_diffusivity")), compact_expression(problem.kappa))
    bad = [k for k, (got, want) in claimed.items() if got != want or isinstance(got, bool)]
    rep.add(Check("solver_claims_match_spec", not bad, "CLAIM_SPEC_MISMATCH",
                  f"mismatched fields: {bad}" if bad else "run parameters match frozen spec"))

    if problem.kappa != "1":
        order = res.get("stiffness_quadrature_order")
        if execution.mesh_topology == "unstructured":
            expected = max(2 * execution.polynomial_order + 7, 11) if execution.mesh_type == "quad" else max(2 * execution.polynomial_order + 3, 9)
        else:
            expected = 2 * execution.polynomial_order + 1
        rep.add(Check("variable_stiffness_quadrature_order", type(order) is int and order == expected,
                      "QUADRATURE_CLAIM_MISMATCH", f"reported weighted stiffness order {order}, expected {expected}"))

    conv = res.get("solver_converged")
    rep.add(Check("solver_reported_convergence",
                  type(conv) is bool and (conv or not validation.require_solver_converged), "SOLVER_NOT_CONVERGED",
                  f"solver_converged={conv} (necessary, not sufficient)"))

    resolved_peak = (n <= 32 and compact_expression(problem.exact_solution)
                     in {"exp(-120*((x-0.375)**2+(y-0.625)**2))",
                         "exp(-40*((x-0.45)**2+(y-0.55)**2))"})
    if require_run_record or res.get("producer") == "poisson_mms":
        _native_claims(res, execution, rep, resolved_peak)

    if require_run_record:
        try:
            record = json.loads((level_dir / "run_record.json").read_text())
            ok = (isinstance(record, dict) and type(record.get("exit_code")) is int
                  and record["exit_code"] == 0 and record.get("exit_meaning") == "OK"
                  and record.get("timed_out") is False and record.get("n") == n)
            rep.add(Check("process_completed", ok, "SOLVER_EXIT", "runner exit, timeout and level"))
        except (OSError, ValueError) as exc:
            rep.add(Check("process_completed", False, "MALFORMED_ARTIFACT", str(exc)))

    mark("verify.field_load")
    try:
        mesh = load_mesh_data(level_dir)
    except (ArtifactError, OSError, UnicodeError) as exc:
        rep.add(Check("artifacts_load", False, "MALFORMED_ARTIFACT", str(exc)))
        return rep
    N = mesh.values.size
    finite = bool(np.all(np.isfinite(mesh.values)))
    rep.add(Check("finite_solution", finite, "NON_FINITE_SOLUTION",
                  f"{int(np.sum(~np.isfinite(mesh.values)))} non-finite nodal values"))
    if not finite:
        return rep

    mark("verify.geometry_boundary")
    try:
        coords, spread = _unique_dof_coords(mesh, N)
    except ArtifactError as exc:
        rep.add(Check("dof_coordinates", False, "MALFORMED_ARTIFACT", str(exc)))
        return rep
    rep.add(Check("dof_coordinate_consistency", spread <= GEOM_TOL, "MALFORMED_ARTIFACT",
                  "max coordinate disagreement of a global dof across elements", spread, GEOM_TOL))

    unstructured = execution.mesh_topology == "unstructured" or isinstance(external_grid, UnstructuredGmsh)
    if unstructured:
        depth = None
        if isinstance(external_grid, UnstructuredGmsh):
            try:
                depth = external_grid.refinements(n)
            except ValueError:
                pass  # A direct level verification also rejects incompatible level labels.
        expected_e = external_grid.element_count * 4 ** depth if depth is not None else None
        elem_ok = mesh.geom == execution.mesh_type and mesh.verts.shape[0] > 0 and (
            expected_e is None or mesh.verts.shape[0] == expected_e
        )
        rep.add(Check("element_count", elem_ok, "MESH_MISMATCH",
                      f"{mesh.verts.shape[0]} elements" + (f", expected {expected_e}" if expected_e else "")))
        rep.add(Check("dof_count", type(rep.num_dofs) is int and rep.num_dofs == N >= 4,
                      "MESH_MISMATCH", f"claimed {rep.num_dofs}, exported {N}"))
        _unstructured_partition_check(mesh, rep)
        source_ok = (elem_ok and depth is not None and external_grid.geom == mesh.geom
                     and _matches_unstructured_source(mesh.verts, external_grid, depth))
        rep.add(Check("external_mesh_refinement", bool(source_ok), "MESH_MISMATCH",
                      "one-to-one correspondence with uniform children of the frozen Gmsh cells"))
    else:
        expected_E = n * n * (2 if execution.mesh_type == "tri" else 1)
        rep.add(Check("element_count", mesh.geom == execution.mesh_type and mesh.verts.shape[0] == expected_E, "MESH_MISMATCH",
                      f"{mesh.verts.shape[0]} elements, expected {expected_E}"))
        rep.add(Check("dof_count", type(rep.num_dofs) is int and rep.num_dofs == N == execution.estimated_dofs(n),
                      "MESH_MISMATCH", f"claimed {rep.num_dofs}, exported {N}, expected {execution.estimated_dofs(n)}"))
        partition_ok = cartesian_partition(mesh, n, GEOM_TOL)
        rep.add(Check("cartesian_cell_partition", partition_ok, "MESH_MISMATCH",
                      "each unit-grid cell has one cyclic quadrilateral or two complementary triangles"))
        if isinstance(external_grid, CartesianGmsh):
            rep.add(Check("external_mesh_refinement", partition_ok and _refined_diagonals(mesh, n, external_grid),
                          "MESH_MISMATCH", "exported elements follow uniform refinement of the archived input"))
        p = execution.polynomial_order
        if p in (1, 2):
            grid = lattice_indices(coords, n * p, GEOM_TOL)
            complete = (grid is not None and len(coords) == (n * p + 1) ** 2
                        and len(np.unique(grid[:, 0] + (n * p + 1) * grid[:, 1])) == len(coords))
            rep.add(Check("cartesian_node_lattice", complete, "MESH_MISMATCH",
                          f"bijective coordinates on the {n*p + 1} by {n*p + 1} nodal lattice"))

    # ---- Gate A (mesh side): boundary attribute <-> label <-> geometry ---
    by_attr = {b.attribute: b for b in problem.boundaries}
    mesh_attrs = set(int(a) for a in np.unique(mesh.bdr_attr))
    undeclared = sorted(mesh_attrs - set(by_attr))
    missing = sorted(set(by_attr) - mesh_attrs)
    rep.add(Check("boundary_attribute_coverage", not undeclared and not missing,
                  "BOUNDARY_COVERAGE",
                  f"undeclared mesh attributes {undeclared}; declared but absent {missing}"))
    label_bad = []
    side_len = {lbl: 0.0 for lbl in SIDE_GEOMETRY}
    intervals = {lbl: [] for lbl in SIDE_GEOMETRY}
    for a, seg in zip(mesh.bdr_attr, mesh.bdr_xy):
        bc = by_attr.get(int(a))
        if bc is None:
            continue
        if not np.all(_side_of_points(seg, bc.label)):
            label_bad.append(int(a))
        else:
            side_len[bc.label] += float(np.linalg.norm(seg[1] - seg[0]))
            axis = 1 - SIDE_GEOMETRY[bc.label][0]
            intervals[bc.label].append(sorted(seg[:, axis]))
    rep.add(Check("boundary_label_geometry", not label_bad, "BOUNDARY_LABEL_MISMATCH",
                  f"attributes whose segments are not on their declared side: {sorted(set(label_bad))}"))
    len_err = max(abs(v - 1.0) for v in side_len.values())
    rep.add(Check("boundary_side_length", len_err <= 1e-10, "BOUNDARY_COVERAGE",
                  "max |declared side length - 1|", len_err, 1e-10))
    if unstructured:
        bad_sides = []
        for label, segments in intervals.items():
            segs = sorted(segments)
            cur = 0.0
            for a0, a1 in segs:
                if abs(a0 - cur) > 1e-10:
                    bad_sides.append(label)
                    break
                cur = a1
            else:
                if abs(cur - 1.0) > 1e-10:
                    bad_sides.append(label)
        rep.add(Check("boundary_partition", not bad_sides, "BOUNDARY_COVERAGE",
                      f"sides with missing or overlapping unstructured segments: {bad_sides}"))
    else:
        knots = np.arange(n + 1) / n
        expected_intervals = np.column_stack((knots[:-1], knots[1:]))
        bad_sides = [label for label, segments in intervals.items()
                     if len(segments) != n or not np.allclose(sorted(segments), expected_intervals,
                                                              rtol=0, atol=GEOM_TOL)]
        rep.add(Check("boundary_partition", not bad_sides, "BOUNDARY_COVERAGE",
                      f"sides with missing, overlapping or misplaced Cartesian segments: {bad_sides}"))

    u_fn = expr.compile_expr(problem.exact_solution)
    f_fn = expr.compile_expr(problem.source)

    # Dirichlet dofs located geometrically from the declared labels.
    dir_labels = [b.label for b in problem.boundaries if b.kind == "dirichlet"]
    on_dir = np.zeros(N, dtype=bool)
    for lbl in dir_labels:
        on_dir |= _side_of_points(coords, lbl)
    u_nodes = u_fn(coords[:, 0], coords[:, 1])
    dmax = float(np.max(np.abs(mesh.values[on_dir] - u_nodes[on_dir]))) if on_dir.any() else 0.0
    rep.add(Check("dirichlet_nodal_values", dmax <= validation.dirichlet_nodal_abs_tol,
                  "DIRICHLET_VIOLATION",
                  f"max |u_h - u_exact| over {int(on_dir.sum())} Dirichlet nodes",
                  dmax, validation.dirichlet_nodal_abs_tol))

    # ---- Gate C: independent error integration ---------------------------
    mark("verify.integration")
    p = execution.polynomial_order
    m = p + 1 + validation.verifier_quadrature_extra_points
    if resolved_peak:
        m = max(m, 13)
    if unstructured:
        m = max(m, 2 * p + 6)
    streaming = mesh.verts.shape[0] > 32768 or validation.reference_solve_method == "amg_pcg"
    q = None
    system = None
    try:
        basis = build_local_basis(mesh, p)
        if streaming and validation.require_reference_solve and N <= validation.reference_solve_max_dofs:
            with phase("verify.integration_and_assembly"):
                system, (min_detJ, l2, h1) = _streaming_assembly(
                    mesh, basis, m, problem, u_fn, f_fn, integrate_errors=True)
        elif streaming:
            min_detJ, l2, h1 = _streaming_errors(mesh, basis, m, u_fn, problem.equation)
        else:
            q = element_quadrature(mesh, basis, m)
            min_detJ = float(np.min(q.detJ))
            l2, h1 = _integrated_errors(mesh, q, u_fn, problem.equation)
    except (ArtifactError, np.linalg.LinAlgError) as exc:
        rep.add(Check("local_basis", False, "MALFORMED_ARTIFACT", str(exc)))
        return rep
    rep.add(Check("mesh_jacobian_positive", min_detJ > 0.0, "INVALID_MESH",
                  "min det J over quadrature points", min_detJ, 0.0))
    rep.l2_error, rep.h1_semi_error = l2, h1
    rep.add(Check("independent_errors_finite", math.isfinite(l2) and math.isfinite(h1),
                  "NON_FINITE_SOLUTION", f"L2={l2:.6e}, H1semi={h1:.6e}"))

    # Self-report consistency (a lying or buggy self-report is rejected).
    for name, mine, theirs in (("l2", l2, rep.self_reported_l2_error),
                               ("h1_semi", h1, rep.self_reported_h1_semi_error)):
        if not _finite(theirs):
            rep.add(Check(f"self_report_{name}", False, "SELF_REPORT_MISMATCH",
                          f"self-reported {name} error is missing or non-finite"))
            continue
        if validation.layered_q1:
            # Q1 contains the exact field. Near zero a relative comparison is singular.
            tol = max(validation.layered_q1.self_report_abs_tol, validation.self_report_rel_tol * abs(mine))
            rep.add(Check(f"self_report_{name}", abs(mine - theirs) <= tol, "SELF_REPORT_MISMATCH",
                          f"absolute difference between verifier and native {name}", abs(mine - theirs), tol))
            continue
        rel = abs(mine - theirs) / max(abs(mine), 1e-300)
        # Higher-order error norms can approach the roundoff of field reconstruction.
        # Preserve the historical relative gate for P1/P2 and record the actual
        # absolute threshold whenever the high-order roundoff bound is active.
        floor = (1e-14 if name == "l2" else 2e-13) if execution.polynomial_order >= 3 else 0.0
        if floor > validation.self_report_rel_tol * abs(mine):
            rep.add(Check(f"self_report_{name}", abs(mine - theirs) <= floor,
                          "SELF_REPORT_MISMATCH", f"absolute difference at the {name} reconstruction floor",
                          abs(mine - theirs), floor))
        else:
            rep.add(Check(f"self_report_{name}", rel <= validation.self_report_rel_tol,
                          "SELF_REPORT_MISMATCH",
                          f"|verifier - solver| / verifier for {name} error "
                          f"(verifier {mine:.10e}, solver {theirs:.10e})",
                          rel, validation.self_report_rel_tol))

    if validation.layered_q1:
        try:
            measured = layered.diagnostics(mesh, n)
        except ArtifactError as exc:
            rep.add(Check("layered_geometry", False, "MESH_MISMATCH", str(exc)))
            return rep
        for key in ("nodal_abs", "flux_rel", "jump_rel", "energy_rel"):
            tol = getattr(validation.layered_q1, f"{key}_tol")
            detail = (f"integral a|grad u_h|^2 = {measured['energy']:.16g}, exact {layered.FLUX:.16g}"
                      if key == "energy_rel" else "from exported physical Q1 corners")
            rep.add(Check(f"layered_{key}", measured[key] <= tol, f"LAYERED_{key.upper()}",
                          detail, measured[key], tol))

    # ---- Gate D: independent reference solve -----------------------------
    mark("verify.reference")
    if validation.require_reference_solve:
        if N > validation.reference_solve_max_dofs:
            rep.add(Check("reference_solve", False, "REFERENCE_SKIPPED_TOO_LARGE",
                          f"{N} dofs > reference_solve_max_dofs"))
        else:
            try:
                if streaming:
                    u_ref = _streaming_reference_solve(
                        mesh, basis, m, problem, coords, on_dir, u_fn, f_fn, validation, system=system
                    )
                else:
                    u_ref = reference_solve(
                        mesh, basis, q, problem, coords, on_dir, u_fn, f_fn, m,
                        method=validation.reference_solve_method,
                        amg_threshold_dofs=validation.reference_solve_amg_threshold_dofs,
                    )
            except ReferenceSolverError as exc:
                rep.add(Check("reference_solve", False, "REFERENCE_SOLVER_FAILED", str(exc)))
                return rep
            except (ArtifactError, np.linalg.LinAlgError) as exc:
                rep.add(Check("reference_solve", False, "INVALID_MESH", str(exc)))
                return rep
            rel = float(np.max(np.abs(u_ref - mesh.values)) / max(np.max(np.abs(u_ref)), 1e-300))
            rep.reference_rel_diff = rel
            rep.add(Check("reference_solve_agreement", rel <= validation.reference_solve_rel_tol,
                          "REFERENCE_MISMATCH",
                          "max|u_mfem - u_ref| / max|u_ref| (independent scipy assembly + "
                          "reference solve)", rel, validation.reference_solve_rel_tol))
    return rep


def _quadrature_batches(q):
    """Limit expression and gradient temporaries to about 64K quadrature points."""
    elements, points = q.wdet.shape
    batch = max(1, 65536 // points)
    for start in range(0, elements, batch):
        yield slice(start, min(start + batch, elements))


def _integrated_errors(mesh, q, u_fn, equation=POISSON):
    # Preserve NumPy's full-array reduction order and historical error values.
    l2_density = np.empty_like(q.wdet)
    h1_density = np.empty_like(q.wdet)
    for block in _quadrature_batches(q):
        U = mesh.values[mesh.elem_dofs[block]]
        uh = np.einsum("eqk,ek->eq", q.phi[block], U)
        guh = np.einsum("eqki,ek->eqi", q.gphi[block], U)
        x, y = q.X[block, :, 0], q.X[block, :, 1]
        ue = u_fn(x, y)
        if equation == LAYERED:
            gx, gy = layered.FLUX / layered.coefficient(x), 0.0
        else:
            gx, gy = expr.gradient(u_fn, x, y)
        l2_density[block] = q.wdet[block] * (uh - ue) ** 2
        h1_density[block] = q.wdet[block] * ((guh[..., 0] - gx) ** 2 + (guh[..., 1] - gy) ** 2)
    return float(np.sqrt(np.sum(l2_density))), float(np.sqrt(np.sum(h1_density)))


def _stiffness_block(gphi, wdet, sx, sy, variable_kappa, equation):
    weights = wdet
    if variable_kappa is not None:
        weights = weights * variable_kappa(sx, sy)
    if equation == LAYERED:
        material = layered.coefficient(sx)
        if not np.all(material == material[:, :1]):
            raise ArtifactError("layered interface cuts an element")
        weights = weights * material
    if equation == ANISOTROPIC:
        return np.einsum("eq,eqid,d,eqjd->eij", weights, gphi, [1000.0, 1.0], gphi)
    return np.einsum("eq,eqid,eqjd->eij", weights, gphi, gphi)


def _slice_errors(mesh, start, stop, qb, u_fn, equation):
    U = mesh.values[mesh.elem_dofs[start:stop]]
    uh = np.einsum("eqk,ek->eq", qb.phi, U)
    guh = np.einsum("eqki,ek->eqi", qb.gphi, U)
    x, y = qb.X[..., 0], qb.X[..., 1]
    ue = u_fn(x, y)
    if equation == LAYERED:
        gx, gy = layered.FLUX / layered.coefficient(x), 0.0
    else:
        gx, gy = expr.gradient(u_fn, x, y)
    return (float(np.min(qb.detJ)), float(np.sum(qb.wdet * (uh - ue) ** 2)),
            float(np.sum(qb.wdet * ((guh[..., 0] - gx) ** 2 + (guh[..., 1] - gy) ** 2))))


def _streaming_errors(mesh, basis, m: int, u_fn, equation=POISSON) -> tuple[float, float, float]:
    """Compute min detJ, L2 error, and H1 seminorm error over bounded element slices."""
    xi, eta, w = reference_quadrature(mesh.geom, m)
    monom = eval_monomials(basis.exps, xi, eta)
    E = mesh.verts.shape[0]
    batch = max(1, 32768 // xi.size)
    min_det = math.inf
    l2_parts, h1_parts = [], []
    for start in range(0, E, batch):
        stop = min(start + batch, E)
        qb = evaluate_quadrature_slice(mesh, basis, xi, eta, w, start, stop, monom_cache=monom)
        det, l2, h1 = _slice_errors(mesh, start, stop, qb, u_fn, equation)
        min_det = np.minimum(min_det, det)
        l2_parts.append(l2)
        h1_parts.append(h1)
    return min_det, math.sqrt(math.fsum(l2_parts)), math.sqrt(math.fsum(h1_parts))


def _assemble_neumann_load(mesh, basis, problem, u_fn, kappa, m: int, F: np.ndarray) -> None:
    """Accumulate outward Neumann boundary flux contributions into F."""
    neu = {b.attribute: b.label for b in problem.boundaries if b.kind == "neumann"}
    if not neu:
        return
    from .fem import boundary_edge_owners

    edge_owner = boundary_edge_owners(mesh, neu)
    g1, w1 = gauss_01(m)
    for attr, seg in zip(mesh.bdr_attr, mesh.bdr_xy):
        lbl = neu.get(int(attr))
        if lbl is None:
            continue
        key = frozenset((tuple(np.round(seg[0], 12)), tuple(np.round(seg[1], 12))))
        e = edge_owner.get(key)
        if e is None:
            raise ArtifactError("boundary segment not found among element edges")
        pts = seg[0][None, :] + g1[:, None] * (seg[1] - seg[0])[None, :]
        L = float(np.linalg.norm(seg[1] - seg[0]))
        ref = inverse_map(mesh.geom, mesh.verts[e:e + 1], pts[None, :, :])[0]
        M, _, _ = eval_monomials(basis.exps, ref[:, 0], ref[:, 1])
        phi = M @ basis.coeffs[e]
        if problem.equation == LAYERED:
            gx, gy = layered.FLUX, 0.0  # a*grad(u_exact) on either material
        else:
            gx, gy = expr.gradient(u_fn, pts[:, 0], pts[:, 1])
        nx, ny = SIDE_NORMAL[lbl]
        gn = kappa(pts[:, 0], pts[:, 1]) * (gx * nx + gy * ny)
        F[mesh.elem_dofs[e]] += (w1 * L * gn) @ phi


def _reduce_csr_sum(parts: list[sp.csr_matrix], N: int) -> sp.csr_matrix:
    if not parts:
        return sp.csr_matrix((N, N), dtype=np.float64)
    while len(parts) > 1:
        nxt = []
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                s = (parts[i] + parts[i + 1]).tocsr()
                s.sum_duplicates()
                nxt.append(s)
            else:
                nxt.append(parts[i])
        parts = nxt
    return parts[0]


def _stream_stiffness(q, kappa, equation):
    """Integrate each element independently, including distorted and variable materials."""
    if equation in (ANISOTROPIC, LAYERED):
        return _stiffness_block(q.gphi, q.wdet, q.X[..., 0], q.X[..., 1], kappa, equation)
    weights = q.wdet
    if kappa is not None:
        weights = weights * kappa(q.X[..., 0], q.X[..., 1])
    if q.gphi.strides[1] == 0:
        # P1 triangles have constant physical gradients within each element.
        # The integral of kappa is still evaluated at every quadrature point.
        grads = q.gphi[:, 0]
        return (grads @ grads.swapaxes(1, 2)) * weights.sum(axis=1)[:, None, None]
    # Batched weighted Gram matrices for non-affine cells or varying gradients.
    grad = q.gphi.transpose(0, 1, 3, 2)
    weighted = (grad * weights[:, :, None, None]).reshape(len(grad), -1, grad.shape[-1])
    grad = grad.reshape(weighted.shape)
    return grad.swapaxes(1, 2) @ weighted


def _streaming_assembly(mesh, basis, m, problem, u_fn, f_fn, *, integrate_errors):
    """Use each physical quadrature slice for both norms and the linear system."""
    N = len(mesh.values)
    E = mesh.verts.shape[0]
    nloc = mesh.elem_dofs.shape[1]
    kappa = expr.compile_expr(problem.kappa)
    var_kappa = None if problem.kappa == "1" else kappa
    xi, eta, w = reference_quadrature(mesh.geom, m)
    monom = eval_monomials(basis.exps, xi, eta)

    # Check if affine constant-coefficient element stiffness repeats with small period.
    Ke_shared = None
    period = E
    if var_kappa is None and problem.equation != LAYERED and basis.coeffs.strides[0] == 0:
        cand = repeated_element_period(mesh.verts[:, 1:] - mesh.verts[:, :1])
        if cand <= 8:
            period = cand
            qp = evaluate_quadrature_slice(mesh, basis, xi, eta, w, 0, period, monom_cache=monom)
            Ke_shared = _stiffness_block(qp.gphi, qp.wdet, qp.X[..., 0], qp.X[..., 1], None, problem.equation)

    F = np.zeros(N, dtype=np.float64)
    sub_batch = max(1, 32768 // xi.size)
    csr_flush = max(sub_batch, min(65536, 1048576 // nloc**2))
    min_det = np.inf
    l2_parts, h1_parts = [], []
    k_parts: list[sp.csr_matrix] = []
    acc_rows, acc_cols, acc_vals = [], [], []
    acc_elems = 0

    def _flush_csr():
        nonlocal acc_elems
        if not acc_vals:
            return
        vals = np.concatenate(acc_vals)
        rows = np.concatenate(acc_rows)
        cols = np.concatenate(acc_cols)
        acc_vals.clear(); acc_rows.clear(); acc_cols.clear()
        part = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
        part.sum_duplicates()
        del vals, rows, cols
        k_parts.append(part)
        acc_elems = 0
        if len(k_parts) >= 8:
            reduced = _reduce_csr_sum(k_parts, N)
            k_parts.clear()
            k_parts.append(reduced)

    for start in range(0, E, sub_batch):
        stop = min(start + sub_batch, E)
        dofs_b = mesh.elem_dofs[start:stop]
        qb = evaluate_quadrature_slice(mesh, basis, xi, eta, w, start, stop, monom_cache=monom)
        x_b, y_b = qb.X[..., 0], qb.X[..., 1]
        fe_b = np.einsum("eq,eq,eqi->ei", qb.wdet, f_fn(x_b, y_b), qb.phi)
        np.add.at(F, dofs_b.ravel(), fe_b.ravel())
        if integrate_errors:
            det, l2, h1 = _slice_errors(mesh, start, stop, qb, u_fn, problem.equation)
            min_det = np.minimum(min_det, det)
            l2_parts.append(l2)
            h1_parts.append(h1)
        if Ke_shared is not None:
            idx = np.arange(start, stop) % period
            ke_b = Ke_shared[idx]
        else:
            ke_b = _stream_stiffness(qb, var_kappa, problem.equation)
        acc_vals.append(ke_b.ravel())
        acc_rows.append(np.repeat(dofs_b, nloc, axis=1).ravel())
        acc_cols.append(np.tile(dofs_b, (1, nloc)).ravel())
        acc_elems += stop - start
        if acc_elems >= csr_flush:
            _flush_csr()
    _flush_csr()
    K = _reduce_csr_sum(k_parts, N)
    k_parts.clear()

    _assemble_neumann_load(mesh, basis, problem, u_fn, kappa, m, F)
    norms = (float(min_det), math.sqrt(math.fsum(l2_parts)), math.sqrt(math.fsum(h1_parts))) if integrate_errors else None
    return (K, F), norms


def _streaming_reference_solve(mesh, basis, m, problem, coords, on_dir, u_fn, f_fn, validation, *, system=None):
    if system is None:
        system, _ = _streaming_assembly(mesh, basis, m, problem, u_fn, f_fn, integrate_errors=False)
    K, F = system
    u = np.zeros(len(F), dtype=np.float64)
    u[on_dir] = u_fn(coords[on_dir, 0], coords[on_dir, 1])
    free = ~on_dir
    rhs = F[free] - K[free][:, on_dir] @ u[on_dir]
    K_free = K[free][:, free].tocsr()
    del K, F
    u_free, _ = solve_reference_system(
        K_free, rhs,
        method=validation.reference_solve_method,
        amg_threshold_dofs=validation.reference_solve_amg_threshold_dofs,
    )
    u[free] = u_free
    return u


def _volume_forms(q, variable_kappa, f_fn, equation=POISSON):
    E, _, nloc = q.phi.shape
    # For affine linear triangles, gradients are stored once per element.
    # Reuse a stiffness integral only after inspecting its actual inputs,
    # including all weights. Other geometries and coefficients use each row.
    period = E
    if variable_kappa is None and equation != LAYERED and q.gphi.strides[1] == 0:
        period = max(repeated_element_period(q.gphi[:, 0]), repeated_element_period(q.wdet))
    Ke = np.empty((period, nloc, nloc))
    Fe = np.empty((E, nloc))
    for block in _quadrature_batches(q):
        x, y = q.X[block, :, 0], q.X[block, :, 1]
        Fe[block] = np.einsum("eq,eq,eqi->ei", q.wdet[block], f_fn(x, y), q.phi[block])
        if block.start < period:
            stiffness = slice(block.start, min(block.stop, period))
            sx, sy = q.X[stiffness, :, 0], q.X[stiffness, :, 1]
            Ke[stiffness] = _stiffness_block(
                q.gphi[stiffness], q.wdet[stiffness], sx, sy, variable_kappa, equation
            )
    if period < E:
        Ke = Ke[np.arange(E) % period]
    return Ke, Fe


def reference_solve(
    mesh, basis, q, problem, coords, on_dir, u_fn, f_fn, m,
    *, method: str = "auto", amg_threshold_dofs: int = 300_000,
) -> np.ndarray:
    N = coords.shape[0]
    nloc = mesh.elem_dofs.shape[1]
    kappa = expr.compile_expr(problem.kappa)
    Ke, Fe = _volume_forms(q, None if problem.kappa == "1" else kappa, f_fn, problem.equation)
    rows = np.repeat(mesh.elem_dofs, nloc, axis=1).ravel()
    cols = np.tile(mesh.elem_dofs, (1, nloc)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(N, N)).tocsr()
    F = np.bincount(mesh.elem_dofs.ravel(), weights=Fe.ravel(), minlength=N)
    del Ke, Fe, rows, cols

    _assemble_neumann_load(mesh, basis, problem, u_fn, kappa, m, F)

    u = np.zeros(N)
    u[on_dir] = u_fn(coords[on_dir, 0], coords[on_dir, 1])
    free = ~on_dir
    rhs = F[free] - K[free][:, on_dir] @ u[on_dir]
    K_free = K[free][:, free].tocsr()
    del K, F
    u_free, _ = solve_reference_system(K_free, rhs, method=method, amg_threshold_dofs=amg_threshold_dofs)
    u[free] = u_free
    return u


def observed_orders(levels: list[LevelReport]) -> list[dict]:
    pairs = []
    for a, b in zip(levels, levels[1:]):
        def order(ea, eb):
            if _finite(ea) and _finite(eb) and ea > 0 and eb > 0:
                return math.log(ea / eb) / math.log(b.n / a.n)
            return None
        pairs.append({"n_coarse": a.n, "n_fine": b.n,
                      "l2_order": order(a.l2_error, b.l2_error),
                      "h1_semi_order": order(a.h1_semi_error, b.h1_semi_error)})
    return pairs


@phase("verify.study")
def verify_attempt(attempt_dir: Path, *, require_run_records: bool = False,
                   reused_levels: dict | None = None) -> dict:
    """Verify one attempt directory from its files alone."""
    attempt_dir = Path(attempt_dir)
    run_dir = attempt_dir.parent.parent
    frozen = run_dir / "frozen"
    external_grid = None
    try:
        problem = ProblemSpec.model_validate_json((frozen / "problem.json").read_text())
        validation = ValidationSpec.model_validate_json((frozen / "validation.json").read_text())
        execution = ExecutionSpec.model_validate_json((attempt_dir / "execution.json").read_text())
        validation.check_problem(problem)
        if problem.equation == LAYERED and (
                (execution.mesh_type, execution.polynomial_order, execution.backend, execution.assembly, execution.solver)
                != ("quad", 1, "serial_cpu", "full", "cg_gs") or any(n not in LAYERED_LEVELS for n in execution.mesh_levels)):
            raise ValueError("frozen execution is outside the reviewed aligned layered Q1 case")
        request_path = run_dir / "request/study_spec.json"
        if request_path.exists() or (run_dir / ARCHIVED_MESH).exists():
            request = StudySpec.model_validate_json(request_path.read_text())
            if request.external_mesh is not None:
                if problem.equation in (LAYERED, ANISOTROPIC) or compact_expression(problem.exact_solution) == GAUSSIAN_EXACT:
                    raise ValueError("this reviewed case requires a generated mesh")
                data = read_mesh(run_dir / ARCHIVED_MESH, request.external_mesh)
                external_grid = (
                    inspect_gmsh_unstructured(data, problem, execution)
                    if execution.mesh_topology == "unstructured"
                    else inspect_gmsh(data, problem, execution)
                )
            elif (run_dir / ARCHIVED_MESH).exists():
                raise ValueError("the request did not declare the archived external mesh")
        mms = check_mms_consistency(problem, validation.mms_consistency_rel_tol)
    except (OSError, ValueError) as exc:
        code = "MISSING_ARTIFACT" if isinstance(exc, FileNotFoundError) else "MALFORMED_ARTIFACT"
        mms = Check("attempt_inputs", False, code, str(exc))
    if not mms.passed:
        return {"verifier": "verifier.verifier", "verifier_version": VERIFIER_VERSION, "verdict": REJECTED,
                "failure_codes": [mms.code], "statement": "verification inputs failed validation",
                "study_checks": [asdict(mms)], "accuracy_check": None, "levels": [],
                "convergence": {"pairs": []}, "shared_assumptions": SHARED_ASSUMPTIONS, "input_sha256": {}}
    study_checks = [mms]
    levels = []
    for n in execution.mesh_levels:
        level_dir = attempt_dir / "levels" / f"n{n:04d}"
        reused = (reused_levels or {}).get(n)
        if reused is None:
            levels.append(verify_level(level_dir, problem, execution, validation, n,
                                       require_run_record=require_run_records, external_grid=external_grid))
            continue
        rep = LevelReport(n=n, level_dir=str(level_dir))
        try:
            if artifact_hashes(level_dir) != reused["sha256"]:
                raise ValueError("copied artifacts differ from the previously verified level")
            values = dict(reused["report"])
            checks = [Check(**c) for c in values.pop("checks")]
            values.pop("ok")
            rep = LevelReport(**{**values, "level_dir": str(level_dir)}, checks=checks)
        except (OSError, TypeError, KeyError, ValueError) as exc:
            rep.add(Check("retry_level_integrity", False, "EVIDENCE_INTEGRITY", str(exc)))
        levels.append(rep)

    study_checks.append(Check("min_levels", len(levels) >= validation.min_levels, "INSUFFICIENT_LEVELS",
                              f"{len(levels)} levels, need {validation.min_levels}",
                              len(levels), validation.min_levels))
    pairs = observed_orders(levels)
    # Convergence checks are evaluated whenever every level produced finite
    # error norms, even if other per-level checks failed: more failure codes
    # give better diagnostics, and the verdict is REJECTED either way.
    errors_available = all(_finite(lv.l2_error) and _finite(lv.h1_semi_error) for lv in levels)
    p = execution.polynomial_order
    conv = {"pairs": pairs, "expected_l2_order": p + 1, "expected_h1_semi_order": p}
    if problem.equation == LAYERED:
        conv = {"mode": "exact_in_q1", "pairs": [], "criterion": "per-level field, flux, interface jump and energy"}
    elif errors_available and pairs:
        l2s = [lv.l2_error for lv in levels]
        h1s = [lv.h1_semi_error for lv in levels]
        if validation.require_monotone_decrease:
            mono = all(b < a for a, b in zip(l2s, l2s[1:])) and all(b < a for a, b in zip(h1s, h1s[1:]))
            study_checks.append(Check("monotone_error_decrease", mono, "NON_MONOTONE_CONVERGENCE",
                                      f"L2={['%.3e' % v for v in l2s]}, H1semi={['%.3e' % v for v in h1s]}"))
        fin = pairs[-1]
        lo, hi = p + 1 - validation.l2_order_min_offset, p + 1 + validation.l2_order_max_offset
        o = fin["l2_order"]
        study_checks.append(Check("finest_pair_l2_order", o is not None and lo <= o <= hi,
                                  "CONVERGENCE_ORDER_OUT_OF_BAND",
                                  f"observed L2 order on n={fin['n_coarse']}->{fin['n_fine']} "
                                  f"must lie in [{lo:.2f}, {hi:.2f}]", o, lo))
        lo, hi = p - validation.h1_order_min_offset, p + validation.h1_order_max_offset
        o = fin["h1_semi_order"]
        study_checks.append(Check("finest_pair_h1_semi_order", o is not None and lo <= o <= hi,
                                  "CONVERGENCE_ORDER_OUT_OF_BAND",
                                  f"observed H1-seminorm order must lie in [{lo:.2f}, {hi:.2f}]", o, lo))
    elif not pairs:
        study_checks.append(Check("convergence_study", False, "INSUFFICIENT_LEVELS", "fewer than 2 levels"))

    accuracy = None
    if validation.max_l2_error_finest is not None and levels and _finite(levels[-1].l2_error):
        accuracy = Check("accuracy_target_finest_l2", levels[-1].l2_error <= validation.max_l2_error_finest,
                         "ACCURACY_NOT_MET", f"finest-level L2 error on n={levels[-1].n}",
                         levels[-1].l2_error, validation.max_l2_error_finest)

    hard_fail = [c for c in study_checks if not c.passed] + [c for lv in levels for c in lv.checks if not c.passed]
    codes = sorted({c.code for c in hard_fail})
    if hard_fail:
        verdict = REJECTED
    elif accuracy is not None and not accuracy.passed:
        verdict = RETRY_ELIGIBLE
        codes = ["ACCURACY_NOT_MET"]
    else:
        verdict = ACCEPTED

    mark("verify.input_hashes")
    inputs = {"frozen/problem.json": sha256_file(frozen / "problem.json"),
              "frozen/validation.json": sha256_file(frozen / "validation.json"),
              "execution.json": sha256_file(attempt_dir / "execution.json")}
    if external_grid is not None:
        inputs[ARCHIVED_MESH] = sha256_file(run_dir / ARCHIVED_MESH)
    for lv in levels:
        for fn in ("result.json", "dof_values.txt", "element_dofs.txt", "elements.txt", "boundary.txt"):
            pth = Path(lv.level_dir) / fn
            inputs[pth.relative_to(attempt_dir).as_posix()] = sha256_file(pth)

    mark("verify.study")
    return {
        "verifier": "verifier.verifier", "verifier_version": VERIFIER_VERSION,
        "verdict": verdict,
        "failure_codes": codes,
        "statement": ("numerical (code) verification passed for this manufactured-solution benchmark"
                      if verdict == ACCEPTED else "numerical verification did NOT pass"),
        "study_checks": [asdict(c) for c in study_checks],
        "accuracy_check": asdict(accuracy) if accuracy else None,
        "levels": [{**{k: v for k, v in asdict(lv).items() if k != "checks"},
                    "checks": [asdict(c) for c in lv.checks], "ok": lv.ok} for lv in levels],
        "convergence": conv,
        "shared_assumptions": SHARED_ASSUMPTIONS,
        "input_sha256": inputs,
    }
