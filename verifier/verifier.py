"""Independent verification of MFEM Poisson runs against a frozen spec.

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

from sciagent.spec import (ANISOTROPIC, LAYERED, LAYERED_EXACT, LAYERED_LEVELS, POISSON,
                           SIDE_GEOMETRY, SIDE_NORMAL, ExecutionSpec, ProblemSpec,
                           ValidationSpec, compact_expression)
from sciagent.evidence import artifact_hashes, sha256_file

from . import expr, layered
from .fem import (ArtifactError, MeshData, build_local_basis, cartesian_partition,
                  element_quadrature, gauss_01, inverse_map, lattice_indices, load_mesh_data)

VERIFIER_VERSION = "0.5.0"
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
    """Gate A: check the declared source against the frozen differential operator."""
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
    diffusion_x = 1000.0 if problem.equation == ANISOTROPIC else 1.0
    lap = expr.neg_laplacian(u, pts[:, 0], pts[:, 1], diffusion_x=diffusion_x)
    fv = f(pts[:, 0], pts[:, 1])
    if not (np.all(np.isfinite(lap)) and np.all(np.isfinite(fv))):
        return Check("mms_consistency", False, "MMS_INCONSISTENT", "non-finite expression values")
    err = float(np.max(np.abs(lap - fv) / (1.0 + np.abs(fv))))
    return Check("mms_consistency", err <= rel_tol, "MMS_INCONSISTENT",
                 "max |(-div(A grad u_exact)) - f| / (1+|f|) at 256 fixed random interior points "
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


def _native_claims(res, execution, rep):
    degree = 2 * execution.polynomial_order + 4
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


def verify_level(level_dir: Path, problem: ProblemSpec, execution: ExecutionSpec,
                 validation: ValidationSpec, n: int, *, require_run_record: bool = False) -> LevelReport:
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
    bad = [k for k, (got, want) in claimed.items() if got != want or isinstance(got, bool)]
    rep.add(Check("solver_claims_match_spec", not bad, "CLAIM_SPEC_MISMATCH",
                  f"mismatched fields: {bad}" if bad else "run parameters match frozen spec"))

    conv = res.get("solver_converged")
    rep.add(Check("solver_reported_convergence",
                  type(conv) is bool and (conv or not validation.require_solver_converged), "SOLVER_NOT_CONVERGED",
                  f"solver_converged={conv} (necessary, not sufficient)"))

    if require_run_record or res.get("producer") == "poisson_mms":
        _native_claims(res, execution, rep)

    if require_run_record:
        try:
            record = json.loads((level_dir / "run_record.json").read_text())
            ok = (isinstance(record, dict) and type(record.get("exit_code")) is int
                  and record["exit_code"] == 0 and record.get("exit_meaning") == "OK"
                  and record.get("timed_out") is False and record.get("n") == n)
            rep.add(Check("process_completed", ok, "SOLVER_EXIT", "runner exit, timeout and level"))
        except (OSError, ValueError) as exc:
            rep.add(Check("process_completed", False, "MALFORMED_ARTIFACT", str(exc)))

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

    try:
        coords, spread = _unique_dof_coords(mesh, N)
    except ArtifactError as exc:
        rep.add(Check("dof_coordinates", False, "MALFORMED_ARTIFACT", str(exc)))
        return rep
    rep.add(Check("dof_coordinate_consistency", spread <= GEOM_TOL, "MALFORMED_ARTIFACT",
                  "max coordinate disagreement of a global dof across elements", spread, GEOM_TOL))

    expected_E = n * n * (2 if execution.mesh_type == "tri" else 1)
    rep.add(Check("element_count", mesh.geom == execution.mesh_type and mesh.verts.shape[0] == expected_E, "MESH_MISMATCH",
                  f"{mesh.verts.shape[0]} elements, expected {expected_E}"))
    rep.add(Check("dof_count", type(rep.num_dofs) is int and rep.num_dofs == N == execution.estimated_dofs(n),
                  "MESH_MISMATCH", f"claimed {rep.num_dofs}, exported {N}, expected {execution.estimated_dofs(n)}"))
    rep.add(Check("cartesian_cell_partition", cartesian_partition(mesh, n, GEOM_TOL), "MESH_MISMATCH",
                  "each unit-grid cell has one cyclic quadrilateral or two complementary triangles"))
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
    try:
        basis = build_local_basis(mesh, p)
        m = p + 1 + validation.verifier_quadrature_extra_points
        q = element_quadrature(mesh, basis, m)
    except (ArtifactError, np.linalg.LinAlgError) as exc:
        rep.add(Check("local_basis", False, "MALFORMED_ARTIFACT", str(exc)))
        return rep
    rep.add(Check("mesh_jacobian_positive", bool(np.all(q.detJ > 0)), "INVALID_MESH",
                  "min det J over quadrature points", float(np.min(q.detJ)), 0.0))
    l2, h1 = integrated_errors(mesh, q, u_fn, problem.equation)
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
    if validation.require_reference_solve:
        if N > validation.reference_solve_max_dofs:
            rep.add(Check("reference_solve", False, "REFERENCE_SKIPPED_TOO_LARGE",
                          f"{N} dofs > reference_solve_max_dofs"))
        else:
            try:
                u_ref = reference_solve(mesh, basis, q, problem, coords, on_dir, u_fn, f_fn, m)
            except (ArtifactError, np.linalg.LinAlgError) as exc:
                rep.add(Check("reference_solve", False, "INVALID_MESH", str(exc)))
                return rep
            rel = float(np.max(np.abs(u_ref - mesh.values)) / max(np.max(np.abs(u_ref)), 1e-300))
            rep.reference_rel_diff = rel
            rep.add(Check("reference_solve_agreement", rel <= validation.reference_solve_rel_tol,
                          "REFERENCE_MISMATCH",
                          "max|u_mfem - u_ref| / max|u_ref| (independent scipy assembly + "
                          "sparse direct solve)", rel, validation.reference_solve_rel_tol))
    return rep


def integrated_errors(mesh, q, u_fn, equation):
    # Bound temporaries for FD gradients, field interpolation and quadrature.
    batch = max(1, 65536 // q.X.shape[1])
    l2_sq, h1_sq = 0.0, 0.0
    for start in range(0, len(mesh.verts), batch):
        s = slice(start, start + batch)
        U = mesh.values[mesh.elem_dofs[s]]
        uh = np.einsum("eqk,ek->eq", q.phi[s], U)
        guh = np.einsum("eqki,ek->eqi", q.gphi[s], U)
        x, y = q.X[s, :, 0], q.X[s, :, 1]
        if equation == LAYERED:
            gx, gy = layered.FLUX / layered.coefficient(x), 0.0
        else:
            gx, gy = expr.gradient(u_fn, x, y)
        l2_sq += np.sum(q.wdet[s] * (uh - u_fn(x, y))**2)
        h1_sq += np.sum(q.wdet[s] * ((guh[..., 0] - gx)**2 + (guh[..., 1] - gy)**2))
    return float(np.sqrt(l2_sq)), float(np.sqrt(h1_sq))


def reference_solve(mesh, basis, q, problem, coords, on_dir, u_fn, f_fn, m) -> np.ndarray:
    N = coords.shape[0]
    E, nloc = mesh.elem_dofs.shape
    shared = q.wdet.strides[0] == 0 and q.gphi.strides[0] == 0
    weights = q.wdet[:1] if shared else q.wdet
    grads = q.gphi[:1] if shared else q.gphi
    if problem.equation == ANISOTROPIC:
        Ke = np.einsum("eq,eqid,d,eqjd->eij", weights, grads, [1000.0, 1.0], grads)
    else:
        Ke = np.einsum("eq,eqid,eqjd->eij", weights, grads, grads)
    if shared:
        Ke = np.broadcast_to(Ke, (E, nloc, nloc))
    if problem.equation == LAYERED:
        # The reviewed grid places the material interface on whole element faces.
        material = layered.coefficient(q.X[..., 0])
        if not np.all(material == material[:, :1]):
            raise ArtifactError("layered interface cuts an element")
        Ke = Ke * material[:, :1, None]
    fq = f_fn(q.X[..., 0], q.X[..., 1])
    Fe = np.einsum("eq,eq,eqi->ei", q.wdet, fq, q.phi)
    rows = np.repeat(mesh.elem_dofs, nloc, axis=1).ravel()
    cols = np.tile(mesh.elem_dofs, (1, nloc)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(N, N)).tocsr()
    F = np.bincount(mesh.elem_dofs.ravel(), weights=Fe.ravel(), minlength=N)

    # Neumann boundary terms: int_{Gamma_N} (grad u_exact . n) v ds
    neu = {b.attribute: b.label for b in problem.boundaries if b.kind == "neumann"}
    if neu:
        nv = mesh.verts.shape[1]
        edge_owner = {}
        # Locate all exterior edges geometrically. A swapped declaration must still
        # reach the independent solve so its reference mismatch stays observable.
        for label in SIDE_GEOMETRY:
            on_side = _side_of_points(mesh.verts, label)
            for e, k in zip(*np.nonzero(on_side & np.roll(on_side, -1, axis=1))):
                a = tuple(np.round(mesh.verts[e, k], 12))
                b = tuple(np.round(mesh.verts[e, (k + 1) % nv], 12))
                edge_owner[frozenset((a, b))] = e
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
            from .fem import eval_monomials
            M, _, _ = eval_monomials(basis.exps, ref[:, 0], ref[:, 1])
            phi = M @ basis.coeffs[e]
            if problem.equation == LAYERED:
                gx, gy = layered.FLUX, 0.0  # a*grad(u_exact) on either material
            else:
                gx, gy = expr.gradient(u_fn, pts[:, 0], pts[:, 1])
            nx, ny = SIDE_NORMAL[lbl]
            gn = gx * nx + gy * ny
            F[mesh.elem_dofs[e]] += (w1 * L * gn) @ phi

    u = np.zeros(N)
    u[on_dir] = u_fn(coords[on_dir, 0], coords[on_dir, 1])
    free = ~on_dir
    rhs = F[free] - K[free][:, on_dir] @ u[on_dir]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
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


def verify_attempt(attempt_dir: Path, *, require_run_records: bool = False,
                   reused_levels: dict | None = None) -> dict:
    """Verify one attempt directory from its files alone."""
    attempt_dir = Path(attempt_dir)
    run_dir = attempt_dir.parent.parent
    frozen = run_dir / "frozen"
    try:
        problem = ProblemSpec.model_validate_json((frozen / "problem.json").read_text())
        validation = ValidationSpec.model_validate_json((frozen / "validation.json").read_text())
        execution = ExecutionSpec.model_validate_json((attempt_dir / "execution.json").read_text())
        validation.check_problem(problem)
        if problem.equation == LAYERED and (
                (execution.mesh_type, execution.polynomial_order, execution.backend, execution.assembly, execution.solver)
                != ("quad", 1, "serial_cpu", "full", "cg_gs") or any(n not in LAYERED_LEVELS for n in execution.mesh_levels)):
            raise ValueError("frozen execution is outside the reviewed aligned layered Q1 case")
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
                                       require_run_record=require_run_records))
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

    inputs = {"frozen/problem.json": sha256_file(frozen / "problem.json"),
              "frozen/validation.json": sha256_file(frozen / "validation.json"),
              "execution.json": sha256_file(attempt_dir / "execution.json")}
    for lv in levels:
        for fn in ("result.json", "dof_values.txt", "element_dofs.txt", "elements.txt", "boundary.txt"):
            pth = Path(lv.level_dir) / fn
            inputs[pth.relative_to(attempt_dir).as_posix()] = sha256_file(pth)

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
