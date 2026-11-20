"""Synthetic solver artifacts for MFEM-free unit tests of the verifier.

Writes the same file formats as cpp/poisson_mms.cpp, using an independently
generated Cartesian mesh, optional random permutation of global and local
numbering, and nodal values chosen by the test (the verifier's own Galerkin
solution, the exact interpolant, or perturbed data).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sciagent.spec import ExecutionSpec, StudySpec, canonical_json
from verifier import expr
from verifier.fem import build_local_basis, element_quadrature, load_mesh_data
from verifier.verifier import reference_solve, _unique_dof_coords, _side_of_points


def _lattice_nodes(geom: str, p: int):
    if geom == "tri":
        return [(a / p, b / p) for b in range(p + 1) for a in range(p + 1 - b)]
    return [(a / p, b / p) for b in range(p + 1) for a in range(p + 1)]


def write_level(level_dir: Path, geom: str, n: int, p: int, rng=None):
    level_dir.mkdir(parents=True)
    h = 1.0 / n
    elems = []
    for j in range(n):
        for i in range(n):
            x0, y0 = i * h, j * h
            c = [(x0, y0), (x0 + h, y0), (x0 + h, y0 + h), (x0, y0 + h)]
            if geom == "quad":
                elems.append(c)
            else:
                elems.append([c[0], c[1], c[2]])
                elems.append([c[0], c[2], c[3]])
    ref_nodes = _lattice_nodes(geom, p)
    gid = {}
    rows = []
    for e, v in enumerate(elems):
        v = np.array(v)
        order = list(range(len(ref_nodes)))
        if rng is not None:
            rng.shuffle(order)
        for k, r in enumerate(order):
            xi, eta = ref_nodes[r]
            if geom == "tri":
                X = v[0] + xi * (v[1] - v[0]) + eta * (v[2] - v[0])
            else:
                X = (v[0] * (1 - xi) * (1 - eta) + v[1] * xi * (1 - eta)
                     + v[2] * xi * eta + v[3] * (1 - xi) * eta)
            key = (round(X[0] * n * p), round(X[1] * n * p))
            gid.setdefault(key, len(gid))
            rows.append((e, k, gid[key], X[0], X[1]))
    N = len(gid)
    perm = rng.permutation(N) if rng is not None else np.arange(N)
    with open(level_dir / "elements.txt", "w") as f:
        f.write("# synthetic\n")
        for e, v in enumerate(elems):
            f.write(f"{e} {geom} {len(v)} " + " ".join(f"{a:.17g} {b:.17g}" for a, b in v) + "\n")
    with open(level_dir / "element_dofs.txt", "w") as f:
        f.write("# synthetic\n")
        for e, k, g, x, y in rows:
            f.write(f"{e} {k} {perm[g]} {x:.17g} {y:.17g}\n")
    with open(level_dir / "boundary.txt", "w") as f:
        f.write("# synthetic\n")
        be = 0
        for i in range(n):
            for attr, a, b in ((1, (i * h, 0.0), ((i + 1) * h, 0.0)),
                               (3, ((i + 1) * h, 1.0), (i * h, 1.0)),
                               (4, (0.0, (i + 1) * h), (0.0, i * h)),
                               (2, (1.0, i * h), (1.0, (i + 1) * h))):
                f.write(f"{be} {attr} {a[0]:.17g} {a[1]:.17g} {b[0]:.17g} {b[1]:.17g}\n")
                be += 1
    _write_values(level_dir, np.zeros(N))
    return N


def _write_values(level_dir: Path, values: np.ndarray):
    with open(level_dir / "dof_values.txt", "w") as f:
        f.write("# synthetic\n")
        for i, v in enumerate(values):
            f.write(f"{i} {v:.17g}\n")


def galerkin_values(level_dir: Path, study: StudySpec, extra: int = 3) -> np.ndarray:
    mesh = load_mesh_data(level_dir)
    p = study.execution.polynomial_order
    basis = build_local_basis(mesh, p)
    m = p + 1 + extra
    q = element_quadrature(mesh, basis, m)
    coords, _ = _unique_dof_coords(mesh, mesh.values.size)
    on_dir = np.zeros(coords.shape[0], dtype=bool)
    for b in study.problem.boundaries:
        if b.kind == "dirichlet":
            on_dir |= _side_of_points(coords, b.label)
    u_fn = expr.compile_expr(study.problem.exact_solution)
    f_fn = expr.compile_expr(study.problem.source)
    return reference_solve(mesh, basis, q, study.problem, coords, on_dir, u_fn, f_fn, m)


def make_attempt(tmp: Path, study: StudySpec, *, shuffle: bool = False, mutate=None,
                 self_report: str = "verifier") -> Path:
    """Create run/frozen + run/attempts/attempt_01 with Galerkin nodal values.

    ``mutate(n, values, coords) -> values`` may corrupt the data per level.
    ``self_report``: "verifier" (honest numbers) or "zero" (a lying solver).
    """
    from verifier.verifier import verify_level

    run = tmp / "run"
    (run / "frozen").mkdir(parents=True)
    (run / "frozen" / "problem.json").write_text(canonical_json(study.problem))
    (run / "frozen" / "validation.json").write_text(canonical_json(study.validation))
    att = run / "attempts" / "attempt_01"
    att.mkdir(parents=True)
    (att / "execution.json").write_text(canonical_json(study.execution))
    rng = np.random.default_rng(7) if shuffle else None
    ex: ExecutionSpec = study.execution
    for n in ex.mesh_levels:
        ld = att / "levels" / f"n{n:04d}"
        N = write_level(ld, ex.mesh_type, n, ex.polynomial_order, rng)
        vals = galerkin_values(ld, study)
        if mutate is not None:
            mesh = load_mesh_data(ld)
            coords, _ = _unique_dof_coords(mesh, N)
            vals = mutate(n, vals.copy(), coords)
        _write_values(ld, vals)
        res = {
            "n": n, "order": ex.polynomial_order, "mesh_type": ex.mesh_type,
            "assembly": ex.assembly, "solver": ex.solver, "rel_tol": ex.rel_tol, "max_iter": ex.max_iter,
            "dirichlet_attributes": study.problem.dirichlet_attributes(),
            "neumann_attributes": study.problem.neumann_attributes(),
            "case_exact_solution": study.problem.exact_solution, "case_source": study.problem.source,
            "solver_converged": True, "solver_iterations": 1, "num_dofs": N,
            "self_reported_l2_error": 0.0, "self_reported_h1_semi_error": 0.0, "injected_fault": "none",
        }
        (ld / "result.json").write_text(json.dumps(res))
        if self_report == "verifier" and np.all(np.isfinite(vals)):
            lv = verify_level(ld, study.problem, ex, study.validation, n)
            res["self_reported_l2_error"] = lv.l2_error
            res["self_reported_h1_semi_error"] = lv.h1_semi_error
            (ld / "result.json").write_text(json.dumps(res))
    return att
