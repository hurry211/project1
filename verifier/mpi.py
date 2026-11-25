"""Independent checks for the Stage 3 P1 triangular sine Poisson benchmark.

Rank files carry no global DOF IDs. Lattice coordinates identify nodes, and
each positively oriented triangle and each physical boundary edge have one
owner. A separate five-point stiffness operator follows from integrating the
linear gradients on the two right triangles of each Cartesian square.
"""

from __future__ import annotations

import json
import math
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.sparse import diags, kronsum
from scipy.sparse.linalg import spsolve

from sciagent.spec import ValidationSpec
from .fem import ArtifactError, lattice_indices, load_mesh_data, reference_quadrature

LEVELS = (8, 16, 32, 64)
RANKS = (1, 2, 4)
LIMITS = ValidationSpec()
GEOM_TOL = 1e-12
SHARED_VALUE_TOL = 1e-12


def _check(passed, reason):
    if not passed:
        raise ArtifactError(reason)


def _bounded_bytes(file, limit):
    _check(file.is_file() and file.stat().st_size <= limit, f"{file.name}: missing or oversized artifact")
    with file.open("rb") as stream:
        data = stream.read(limit + 1)
    _check(len(data) <= limit, f"{file.name}: oversized artifact")
    return data


def _rows(file, width, max_rows, *, allow_empty=False):
    # One decimal IEEE-754 token at max_digits10 fits in 32 bytes. Leave room
    # for whitespace and a short header. Bound the read again after stat().
    raw = _bounded_bytes(file, 4096 + 32 * width * max_rows)
    data = []
    for line in raw.splitlines():
        if line.strip() and not line.lstrip().startswith(b"#"):
            _check(len(data) < max_rows, f"{file.name}: excess data rows")
            data.append(line.decode("ascii"))
    if not data and allow_empty:
        return np.empty((0, width))
    _check(0 < len(data) <= max_rows, f"{file.name}: incorrect row count")
    arr = np.loadtxt(data, ndmin=2)
    _check(arr.shape == (len(data), width) and np.isfinite(arr).all(),
           f"{file.name}: need {width} finite values per row")
    return arr


def _ids(xy, n):
    grid = lattice_indices(xy, n, GEOM_TOL)
    _check(grid is not None, "vertex outside the prescribed grid")
    return grid[..., 0] + (n + 1)*grid[..., 1]


def _grid(n):
    xy = np.indices((n+1, n+1))[::-1].reshape(2, -1).T / n
    sw = np.arange((n+1)**2).reshape(n+1, n+1)[:-1, :-1].ravel()
    se, nw, ne = sw+1, sw+n+1, sw+n+2
    cells = np.concatenate((np.stack((sw,se,ne), axis=1), np.stack((sw,ne,nw), axis=1)))
    return xy, cells


def _cell_keys(indices, n):
    """Bijection from a SW--NE triangle's three nodes to its canonical cell ID."""
    a, b, c = np.sort(indices, axis=1).T
    _check(np.all((a % (n+1) < n) & (a // (n+1) < n) & (c == a+n+2) &
                  ((b == a+1) | (b == a+n+1))), "triangle differs from the reviewed Cartesian connectivity")
    return 2 * (a % (n+1) + n * (a // (n+1))) + (b == a+n+1)


def _boundary(n):
    tags = {}
    for i in range(n):
        top = n*(n+1) + i
        left = i*(n+1)
        tags[(i, i+1)] = 1
        tags[(left+n, left+2*n+1)] = 2
        tags[(top, top+1)] = 3
        tags[(left, left+n+1)] = 4
    return tags


def _merge(directory, n, ranks):
    for kind in ("cells", "boundary"):
        _check({f.name for f in directory.glob(f"{kind}.*.tsv")} == {f"{kind}.{k}.tsv" for k in range(ranks)},
               f"{kind}: missing or unexpected rank exports")
    occurrences = np.zeros(2*n*n, dtype=int)
    vmin, vmax = np.full((n+1)**2, np.inf), np.full((n+1)**2, -np.inf)
    sides = _boundary(n)
    counts = []
    for rank in range(ranks):
        rows = _rows(directory / f"cells.{rank}.tsv", 9, 2*n*n).reshape(-1,3,3)
        points, vals = rows[:,:,:2], rows[:,:,2]
        a, b = points[:,1]-points[:,0], points[:,2]-points[:,0]
        _check((a[:,0]*b[:,1] > a[:,1]*b[:,0]).all(), "reversed or degenerate MPI triangle")
        ids = _ids(points, n)
        np.add.at(occurrences, _cell_keys(ids, n), 1)
        np.minimum.at(vmin, ids, vals)
        np.maximum.at(vmax, ids, vals)
        counts.append(len(rows))

        edges = np.sort(np.stack((ids, np.roll(ids, -1, axis=1)), axis=-1).reshape(-1,2), axis=1)
        # Only exterior edges can be claimed by this rank.
        a, b = edges.T
        outer = ((a // (n+1) == b // (n+1)) & ((a < n+1) | (a >= n*(n+1)))) | (
            (a % (n+1) == b % (n+1)) & ((a % (n+1) == 0) | (a % (n+1) == n)))
        local = {tuple(pair) for pair in edges[outer]}
        boundary = _rows(directory / f"boundary.{rank}.tsv", 5, 4*n, allow_empty=True)
        for tag, pair in zip(boundary[:,0], _ids(boundary[:,1:].reshape(-1,2,2), n), strict=True):
            edge = tuple(sorted(pair))
            _check(edge in local and sides.get(edge) == tag, "wrong boundary owner, label or duplicate edge")
            local.remove(edge)
            del sides[edge]
        _check(not local, "missing physical boundary on owning rank")
    _check(np.all(occurrences == 1), "missing or multiply owned MPI triangle")
    _check(not sides, "incomplete exterior boundary")
    _check(np.isfinite(vmin).all() and np.isfinite(vmax).all(), "missing nodal values")
    _check(np.max(vmax-vmin) <= SHARED_VALUE_TOL, "inconsistent shared or repeated nodal value")
    return vmin, counts


def _serial_field(level, n):
    mesh = load_mesh_data(level)
    _check(mesh.geom == "tri" and mesh.elem_dofs.shape == (2*n*n, 3), "serial comparison needs P1 triangles")
    ids = _ids(mesh.elem_node_xy, n)
    _check(np.array_equal(np.sort(ids, axis=1), np.sort(_ids(mesh.verts, n), axis=1)),
           "serial P1 nodes differ from cell vertices")
    _check(np.array_equal(np.sort(_cell_keys(ids, n)), np.arange(2*n*n)), "serial mesh differs from fixed study")
    _check(len(mesh.values) == (n+1)**2 and np.isfinite(mesh.values).all(), "invalid serial nodal values")
    order = np.full((n+1)**2, -1)
    order[ids] = mesh.elem_dofs
    _check(np.array_equal(np.sort(order), np.arange(len(order))) and np.array_equal(order[ids], mesh.elem_dofs),
           "conflicting serial DOF mapping")
    return mesh.values[order]


def _native_quadrature():
    # Symmetric degree-six triangle rule of the pinned MFEM release. Use this
    # rule only to check its algebraic residual. The independent solve and
    # error integrals use tensor Gauss points and a Duffy transformation.
    pts, ws = [], []
    for a,b,w in ((0.06308901449150223, 0.06308901449150223, 0.02542245318510341),
                  (0.24928674517091043, 0.24928674517091043, 0.05839313786318968),
                  (0.6365024991213987, 0.3103524510337844, 0.041425537809186785)):
        orbit = sorted(set(permutations((a,b,1-a-b))))
        pts.extend(orbit)
        ws.extend([w]*len(orbit))
    return np.asarray(pts), np.asarray(ws)


def _source_load(n, coords, cells, phi, weights):
    points = np.einsum("qa,ead->eqd", phi, coords[cells])
    x, y = points.transpose(2,0,1)
    u = np.sin(np.pi*x)*np.sin(np.pi*y)
    local = (u*weights) @ phi * (2*np.pi**2 / n**2)
    load = np.bincount(cells.ravel(), weights=local.ravel(), minlength=len(coords))
    return load.reshape(n+1,n+1)[1:-1,1:-1].ravel(), u, x, y


def _relative(actual, reference):
    scale = np.max(np.abs(reference))
    _check(scale > 0, "zero reference field for sine benchmark")
    return float(np.max(np.abs(actual-reference)) / scale)


def _field_checks(field, serial, n, result):
    coords, cells = _grid(n)
    fixed = (coords == 0).any(axis=1) | (coords == 1).any(axis=1)
    _check(np.max(abs(field[fixed])) <= LIMITS.dirichlet_nodal_abs_tol, "Dirichlet boundary mismatch")
    serial_gap = _relative(field, serial)
    _check(serial_gap <= LIMITS.reference_solve_rel_tol, "MPI field differs from matched serial field")

    # Exact homogeneous Dirichlet data. Each interior P1 row on this grid
    # has diagonal 4 and four -1 neighbors, including nodes next to the sides.
    line = diags((-np.ones(n-2), 2*np.ones(n-1), -np.ones(n-2)), (-1, 0, 1))
    stiffness = kronsum(line, line, format="csr")
    xi, eta, w = reference_quadrature("tri", 1 + 1 + LIMITS.verifier_quadrature_extra_points)
    phi = np.stack((1-xi-eta, xi, eta), axis=1)
    load, exact, x, y = _source_load(n, coords, cells, phi, w)
    reference = np.zeros_like(field)
    reference[~fixed] = spsolve(stiffness, load)
    reference_gap = _relative(field, reference)
    _check(reference_gap <= LIMITS.reference_solve_rel_tol, "MPI field differs from SciPy reference solve")

    def norms(values):
        u = values[cells]
        value_error = u @ phi.T - exact
        # On the lower triangle dx follows SW->SE and dy follows SE->NE.
        # On the upper triangle dy follows SW->NW and dx follows NW->NE.
        lower, upper = np.split(u, 2)
        gx = n*np.concatenate((lower[:,1]-lower[:,0], upper[:,1]-upper[:,2]))
        gy = n*np.concatenate((lower[:,2]-lower[:,1], upper[:,2]-upper[:,0]))
        gradx = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
        grady = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
        error = (gx[:,None]-gradx)**2 + (gy[:,None]-grady)**2
        return np.sqrt([np.sum(value_error**2*w)/n**2, np.sum(error*w)/n**2])

    errors, serial_errors = norms(field), norms(serial)
    norms_gap = float(np.max(abs(errors/serial_errors-1)))
    _check(norms_gap <= LIMITS.self_report_rel_tol, "MPI error norms differ from serial")
    reported = np.array([result["l2_error"], result["h1_semi_error"]])
    _check(np.max(abs(reported/errors-1)) <= LIMITS.self_report_rel_tol, "incorrect native error norms")

    load, *_ = _source_load(n, coords, cells, *_native_quadrature())
    # MFEM's boundary projection evaluates sin(pi) at x or y = 1 in floating
    # point; its tiny roundoff contribution is covered by the absolute floor.
    rhs = float(np.linalg.norm(load))
    residual = float(np.linalg.norm(stiffness @ field[~fixed] - load))
    floor = 64*np.finfo(float).eps * np.sqrt(field.size) * max(1, np.max(abs(field)))
    _check(abs(rhs-result["rhs_l2"]) <= 1e-12*rhs, "incorrect constrained RHS norm")
    _check(abs(residual-result["residual_l2"]) <= floor, "incorrect constrained residual norm")
    _check(residual/rhs <= LIMITS.reference_solve_rel_tol, "independent relative residual exceeds tolerance")
    return {"l2_error": float(errors[0]), "h1_semi_error": float(errors[1]),
            "serial_relative_field_difference": serial_gap, "reference_relative_field_difference": reference_gap,
            "serial_relative_norm_difference": norms_gap, "relative_residual": residual/rhs,
            "independent_residual_l2": residual, "independent_rhs_l2": rhs}


def verify_mpi_level(mpi_dir: Path, serial_level: Path, *, n: int, ranks: int) -> dict:
    """Certify one rank set using the level from a separately pinned, accepted serial study."""
    _check(type(n) is int and n in LEVELS and type(ranks) is int and ranks in RANKS,
           "unreviewed MPI study dimensions")
    directory = Path(mpi_dir)
    try:
        result = json.loads(_bounded_bytes(directory / "result.json", 16384))
        expected = {"n": n, "ranks": ranks, "order": 1, "case": "sin_sin", "solver": "cg_boomeramg",
                    "global_elements": 2*n*n, "global_dofs": (n+1)**2, "max_iter": 20000,
                    "rhs_quadrature_order": 6, "error_quadrature_order": 6, "solver_converged": True}
        _check(isinstance(result, dict) and all(type(result.get(k)) is type(v) and result[k] == v
                   for k,v in expected.items()), "native MPI configuration or convergence claim mismatch")
        metrics = ("abs_tol", "rel_tol", "l2_error", "h1_semi_error", "residual_l2", "rhs_l2",
                   "solver_initial_norm", "solver_final_norm", "setup_seconds_max", "solve_seconds_max")
        _check(all(type(result.get(k)) in (int,float) and math.isfinite(result[k]) and result[k] >= 0
                   for k in metrics), "invalid native MPI numeric record")
        _check(result["rel_tol"] == 1e-12 and result["abs_tol"] == 0, "solver tolerance mismatch")
        its = result.get("solver_iterations")
        _check(type(its) is int and 1 <= its <= result["max_iter"] and result["solver_initial_norm"] > 0
               and result["solver_final_norm"] <= result["rel_tol"]*result["solver_initial_norm"]*(1+1e-12),
               "solver stopping record inconsistent with convergence")
        field, counts = _merge(directory, n, ranks)
        serial = _serial_field(Path(serial_level), n)
        checked = _field_checks(field, serial, n, result)
        return {**expected, **{k:result[k] for k in metrics}, **checked,
                "solver_iterations": its, "rank_elements": counts}
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise ArtifactError(f"{directory}: {exc}") from exc


def verify_mpi_orders(levels: list[dict]) -> dict:
    """Require monotone independently integrated errors and the original finest-pair P1 bands."""
    _check(tuple(lv["n"] for lv in levels) == LEVELS and len({lv["ranks"] for lv in levels}) == 1,
           "incomplete or mixed MPI convergence sequence")
    errors = np.asarray([[lv["l2_error"], lv["h1_semi_error"]] for lv in levels])
    _check(np.isfinite(errors).all() and np.all(errors > 0) and (np.diff(errors, axis=0) < 0).all(),
           "errors must be finite, positive and strictly decreasing")
    l2, h1 = np.log2(errors[-2]/errors[-1])
    _check(2-LIMITS.l2_order_min_offset <= l2 <= 2+LIMITS.l2_order_max_offset and
           1-LIMITS.h1_order_min_offset <= h1 <= 1+LIMITS.h1_order_max_offset, "MPI orders outside fixed bands")
    return {"finest_l2_order": float(l2), "finest_h1_order": float(h1)}
