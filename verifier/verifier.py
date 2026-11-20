"""Independent verification of MFEM Poisson runs against a frozen spec.

Inputs per attempt directory (written by the runner, read-only here):
    frozen/problem.json, frozen/validation.json, execution.json
    levels/n<NNNN>/{result.json, elements.txt, element_dofs.txt,
                    dof_values.txt, boundary.txt}

Output: a typed verdict (ACCEPTED / RETRY_ELIGIBLE / REJECTED) with every
check, its measured value, its threshold, and failure codes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sciagent.spec import (SIDE_GEOMETRY, SIDE_NORMAL, ExecutionSpec, ProblemSpec,
                           ValidationSpec)

from . import expr
from .fem import (ArtifactError, MeshData, build_local_basis, element_quadrature,
                  gauss_01, inverse_map, load_mesh_data)

VERIFIER_VERSION = "0.1.0"
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
    "Element vertex coordinates and dof node coordinates are taken from solver-exported "
    "artifacts; their consistency is checked but the mesh generator itself is MFEM's.",
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
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def check_mms_consistency(problem: ProblemSpec, rel_tol: float) -> Check:
    """Gate A: declared source must equal -lap(exact_solution)."""
    try:
        u = expr.compile_expr(problem.exact_solution)
        f = expr.compile_expr(problem.source)
    except expr.ExpressionError as exc:
        return Check("mms_expressions_parse", False, "INVALID_EXPRESSION", str(exc))
    rng = np.random.default_rng(12345)
    pts = rng.uniform(0.05, 0.95, size=(256, 2))
    lap = expr.neg_laplacian(u, pts[:, 0], pts[:, 1])
    fv = f(pts[:, 0], pts[:, 1])
    if not (np.all(np.isfinite(lap)) and np.all(np.isfinite(fv))):
        return Check("mms_consistency", False, "MMS_INCONSISTENT", "non-finite expression values")
    err = float(np.max(np.abs(lap - fv) / (1.0 + np.abs(fv))))
    return Check("mms_consistency", err <= rel_tol, "MMS_INCONSISTENT",
                 "max |(-lap u_exact) - f| / (1+|f|) at 256 fixed random interior points "
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


def verify_level(level_dir: Path, problem: ProblemSpec, execution: ExecutionSpec,
                 validation: ValidationSpec, n: int) -> LevelReport:
    rep = LevelReport(n=n, level_dir=str(level_dir))
    level_dir = Path(level_dir)

    # ---- Gate B: execution artifacts and the solver's claims -------------
    rpath = level_dir / "result.json"
    if not rpath.is_file():
        rep.add(Check("result_json_present", False, "MISSING_ARTIFACT", "result.json not found"))
        return rep
    try:
        res = json.loads(rpath.read_text())
    except json.JSONDecodeError as exc:
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
        "dirichlet_attributes": (sorted(res.get("dirichlet_attributes") or []), problem.dirichlet_attributes()),
        "neumann_attributes": (sorted(res.get("neumann_attributes") or []), problem.neumann_attributes()),
        "exact_solution": (res.get("case_exact_solution"), problem.exact_solution),
        "source": (res.get("case_source"), problem.source),
    }
    bad = [k for k, (got, want) in claimed.items() if got != want]
    rep.add(Check("solver_claims_match_spec", not bad, "CLAIM_SPEC_MISMATCH",
                  f"mismatched fields: {bad}" if bad else "run parameters match frozen spec"))

    conv = bool(res.get("solver_converged"))
    rep.add(Check("solver_reported_convergence",
                  conv or not validation.require_solver_converged, "SOLVER_NOT_CONVERGED",
                  f"solver_converged={conv} (necessary, not sufficient)"))

    try:
        mesh = load_mesh_data(level_dir)
    except ArtifactError as exc:
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
    rep.add(Check("element_count", mesh.verts.shape[0] == expected_E, "MESH_MISMATCH",
                  f"{mesh.verts.shape[0]} elements, expected {expected_E}"))

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
    for a, seg in zip(mesh.bdr_attr, mesh.bdr_xy):
        bc = by_attr.get(int(a))
        if bc is None:
            continue
        if not np.all(_side_of_points(seg, bc.label)):
            label_bad.append(int(a))
        else:
            side_len[bc.label] += float(np.linalg.norm(seg[1] - seg[0]))
    rep.add(Check("boundary_label_geometry", not label_bad, "BOUNDARY_LABEL_MISMATCH",
                  f"attributes whose segments are not on their declared side: {sorted(set(label_bad))}"))
    len_err = max(abs(v - 1.0) for v in side_len.values())
    rep.add(Check("boundary_side_length", len_err <= 1e-10, "BOUNDARY_COVERAGE",
                  "max |declared side length - 1|", len_err, 1e-10))

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
    p = execution.polynomial_order
    try:
        basis = build_local_basis(mesh, p)
    except ArtifactError as exc:
        rep.add(Check("local_basis", False, "MALFORMED_ARTIFACT", str(exc)))
        return rep
    m = p + 1 + validation.verifier_quadrature_extra_points
    q = element_quadrature(mesh, basis, m)
    rep.add(Check("mesh_jacobian_positive", bool(np.all(q.detJ > 0)), "INVALID_MESH",
                  "min det J over quadrature points", float(np.min(q.detJ)), 0.0))
    U = mesh.values[mesh.elem_dofs]                      # (E, nloc)
    uh = np.einsum("eqk,ek->eq", q.phi, U)
    guh = np.einsum("eqki,ek->eqi", q.gphi, U)
    Xq = q.X
    ue = u_fn(Xq[..., 0], Xq[..., 1])
    gx, gy = expr.gradient(u_fn, Xq[..., 0], Xq[..., 1])
    l2 = float(np.sqrt(np.sum(q.wdet * (uh - ue) ** 2)))
    h1 = float(np.sqrt(np.sum(q.wdet * ((guh[..., 0] - gx) ** 2 + (guh[..., 1] - gy) ** 2))))
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
        rel = abs(mine - theirs) / max(abs(mine), 1e-300)
        rep.add(Check(f"self_report_{name}", rel <= validation.self_report_rel_tol,
                      "SELF_REPORT_MISMATCH",
                      f"|verifier - solver| / verifier for {name} error "
                      f"(verifier {mine:.10e}, solver {theirs:.10e})",
                      rel, validation.self_report_rel_tol))

    # ---- Gate D: independent reference solve -----------------------------
    if validation.require_reference_solve:
        if N > validation.reference_solve_max_dofs:
            rep.add(Check("reference_solve", False, "REFERENCE_SKIPPED_TOO_LARGE",
                          f"{N} dofs > reference_solve_max_dofs"))
        else:
            u_ref = reference_solve(mesh, basis, q, problem, coords, on_dir, u_fn, f_fn, m)
            rel = float(np.max(np.abs(u_ref - mesh.values)) / max(np.max(np.abs(u_ref)), 1e-300))
            rep.reference_rel_diff = rel
            rep.add(Check("reference_solve_agreement", rel <= validation.reference_solve_rel_tol,
                          "REFERENCE_MISMATCH",
                          "max|u_mfem - u_ref| / max|u_ref| (independent scipy assembly + "
                          "sparse direct solve)", rel, validation.reference_solve_rel_tol))
    return rep


def reference_solve(mesh, basis, q, problem, coords, on_dir, u_fn, f_fn, m) -> np.ndarray:
    N = coords.shape[0]
    E, nloc = mesh.elem_dofs.shape
    Ke = np.einsum("eq,eqid,eqjd->eij", q.wdet, q.gphi, q.gphi)
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
        for e in range(E):
            for k in range(nv):
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


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def verify_attempt(attempt_dir: Path) -> dict:
    """Verify one attempt directory from its files alone."""
    attempt_dir = Path(attempt_dir)
    run_dir = attempt_dir.parent.parent
    frozen = run_dir / "frozen"
    problem = ProblemSpec.model_validate_json((frozen / "problem.json").read_text())
    validation = ValidationSpec.model_validate_json((frozen / "validation.json").read_text())
    execution = ExecutionSpec.model_validate_json((attempt_dir / "execution.json").read_text())

    study_checks: list[Check] = [check_mms_consistency(problem, validation.mms_consistency_rel_tol)]
    levels = [verify_level(attempt_dir / "levels" / f"n{n:04d}", problem, execution, validation, n)
              for n in execution.mesh_levels]

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
    if errors_available and pairs:
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

    inputs = {"frozen/problem.json": _sha(frozen / "problem.json"),
              "frozen/validation.json": _sha(frozen / "validation.json"),
              "execution.json": _sha(attempt_dir / "execution.json")}
    for lv in levels:
        for fn in ("result.json", "dof_values.txt", "element_dofs.txt", "elements.txt", "boundary.txt"):
            pth = Path(lv.level_dir) / fn
            inputs[str(pth.relative_to(attempt_dir))] = _sha(pth)

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
