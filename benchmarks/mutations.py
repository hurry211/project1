"""Controlled single-fault transformations of copies of native Poisson artifacts."""

import json
import shutil

import numpy as np

from .reference import relative_difference


CLEAN = ("original", "renumbered")
NUMERICAL = ("source_sign", "stiffness_scale", "centroid_errors", "quadrature_claim",
             "boundary_labels", "dof_swap", "duplicate_cell", "interior_warp", "nan", "infinity",
             "cg_residual", "cg_iterations", "cached_result", "inconsistent_source")
ARCHIVE = ("relaxed_validation", "relaxed_retry_policy", "substituted_archive")


def require(ok, reason):
    if not ok:
        raise ValueError(f"challenge setup failed: {reason}")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, allow_nan=False, sort_keys=True, indent=2) + "\n")


def write_values(level, values):
    np.savetxt(level / "dof_values.txt", np.column_stack((np.arange(len(values)), values)),
               fmt=["%d", "%.17g"], header="global_dof value")


def write_nodes(level, nodes):
    np.savetxt(level / "element_dofs.txt", nodes.reshape(-1, 5),
               fmt=["%d", "%d", "%d", "%.17g", "%.17g"], header="elem local_index global_dof x y")


def element_rows(level):
    return [r.split() for r in (level / "elements.txt").read_text().splitlines()
            if r.strip() and not r.startswith("#")]


def write_elements(level, rows):
    (level / "elements.txt").write_text("".join(" ".join(r) + "\n" for r in rows))


def renumber(level, clean, rng):
    rows = element_rows(level)
    nodes = np.loadtxt(level / "element_dofs.txt").reshape(len(rows), -1, 5)
    perm = rng.permutation(len(clean.order))
    element_perm = rng.permutation(len(rows))
    changed = int(np.count_nonzero(perm != np.arange(len(perm))))
    require(changed > 0, "DOF renumbering produced no change")
    values = np.empty(len(perm))
    values[perm] = clean.field[clean.order]
    write_values(level, values)
    nodes[:, :, 2] = perm[nodes[:, :, 2].astype(int)]
    nodes = nodes[element_perm]
    nodes[:, :, 0] = np.arange(len(rows))[:, None]
    write_nodes(level, nodes)
    write_elements(level, [[str(k), *rows[e][1:]] for k, e in enumerate(element_perm)])
    decoded, order = clean.ref.native_field(level)
    clean.ref.check_topology(level, order, clean.n)
    require(np.array_equal(decoded, clean.field), "renumbering altered the discrete field")
    return {"renumbered_dofs": changed, "max_field_difference": 0.0}


def numerical_mutation(level, family, clean, validation, rng, donor=None):
    ref, field, order = clean.ref, clean.field, clean.order
    result_path = level / "result.json"
    result = read_json(result_path)
    if family in ("source_sign", "stiffness_scale"):
        u = ref.solve(source_scale=-1.0) if family == "source_sign" else ref.solve(stiffness_scale=1.0001)
        delta = relative_difference(u, clean.solution)
        require(delta > validation.reference_solve_rel_tol, "perturbed solve is within tolerance")
        write_values(level, u[order])
        result["self_reported_l2_error"], result["self_reported_h1_semi_error"] = ref.errors(u)
        write_json(result_path, result)
        return {"relative_field_difference": delta}
    if family == "centroid_errors":
        coarse = ref.errors(field, centroid=True)
        fine = ref.errors(field)
        delta = max(abs(a / b - 1) for a, b in zip(coarse, fine))
        require(delta > validation.self_report_rel_tol, "one-point rule resolved this case")
        result["self_reported_l2_error"], result["self_reported_h1_semi_error"] = coarse
        write_json(result_path, result)
        return {"max_relative_norm_difference": delta}
    if family == "quadrature_claim":
        old = result["rhs_quadrature_order"]
        require(old > 1, "baseline has no higher-order quadrature")
        result["rhs_quadrature_order"] = 1
        write_json(result_path, result)
        return {"original_rhs_order": old, "reported_rhs_order": 1}
    if family == "cg_residual":
        initial = result["solver_initial_norm"]
        require(result["solver_converged"] is True and initial > 0, "expected a converged CG run")
        result["solver_final_norm"] = initial
        write_json(result_path, result)
        return {"reported_final_over_stopping_limit": 1.0 / result["rel_tol"]}
    if family == "cg_iterations":
        result["solver_iterations"] = result["max_iter"] + 1
        write_json(result_path, result)
        return {"iterations_above_maximum": 1}
    if family == "boundary_labels":
        edge = np.loadtxt(level / "boundary.txt")
        a, b = edge[:, 1] == 1, edge[:, 1] == 3
        require(a.any() and b.any(), "no bottom/top edges")
        edge[a, 1], edge[b, 1] = 3, 1
        np.savetxt(level / "boundary.txt", edge, fmt=["%d", "%d"] + ["%.17g"] * 4)
        return {"wrong_side_edges": int(a.sum() + b.sum())}
    if family in ("nan", "infinity", "dof_swap"):
        values = field[order].copy()
        x, y = ref.basis.doflocs[:, order]
        interior = np.flatnonzero((x > 0) & (x < 1) & (y > 0) & (y < 1))
        if family == "dof_swap":
            i, j = interior[values[interior].argmin()], interior[values[interior].argmax()]
            delta = float(abs(values[i] - values[j]))
            require(delta > validation.reference_solve_rel_tol * np.max(np.abs(values)), "equivalent DOFs")
            values[i], values[j] = values[j], values[i]
            witness = {"swapped_value_difference": delta}
        else:
            values[rng.choice(interior)] = np.nan if family == "nan" else np.inf
            witness = {"nonfinite_nodes": int(np.count_nonzero(~np.isfinite(values)))}
        write_values(level, values)
        return witness
    if family in ("duplicate_cell", "interior_warp"):
        rows = element_rows(level)
        vertices = np.array([r[3:] for r in rows], dtype=float).reshape(len(rows), -1, 2)
        nodes = np.loadtxt(level / "element_dofs.txt").reshape(len(rows), -1, 5)
        if family == "duplicate_cell":
            centers = vertices.mean(axis=1)
            inside = np.flatnonzero(np.all((centers > .2) & (centers < .8), axis=1))
            i, j = rng.choice(inside, 2, replace=False)
            require(not np.array_equal(vertices[i], vertices[j]), "donor is already a duplicate")
            vertices[i] = vertices[j]
            nodes[i, :, 2:] = nodes[j, :, 2:]
            witness = {"overwritten_element": int(i), "duplicate_of": int(j)}
        else:
            def warp(xy):
                x, y = xy[..., 0], xy[..., 1]
                displacement = 2e-10 * 16 * x * (1 - x) * y * (1 - y)
                xy[..., 0] += displacement
                return float(displacement.max())
            maximum = warp(vertices)
            warp(nodes[:, :, 3:])
            require(maximum > 1e-12, "mesh warp within coordinate tolerance")
            witness = {"max_x_displacement": maximum}
        write_nodes(level, nodes)
        write_elements(level, [r[:3] + [format(x, ".17g") for x in xy.ravel()]
                               for r, xy in zip(rows, vertices)])
        return witness
    if family == "cached_result":
        other = read_json(donor / "result.json")
        keys = ("case", "mesh_type", "order", "solver", "assembly")
        different = [k for k in keys if result[k] != other[k]]
        require(different and result["n"] == other["n"], "cache donor must be a different configuration at the same n")
        shutil.copyfile(donor / "result.json", result_path)
        return {"different_configuration_fields": different}
    if family == "inconsistent_source":
        path = level.parents[3] / "frozen/problem.json"
        problem = read_json(path)
        problem["source"] = "-(" + problem["source"] + ")"
        write_json(path, problem)
        x, y = .25, .25
        f = ((np.pi**2 - 1) * np.cos(np.pi*x) * np.exp(y) if ref.mixed
             else 2*np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y))
        require(abs(2*f) > validation.mms_consistency_rel_tol, "source sign unobservable")
        return {"abs_source_change_at_one_quarter": float(abs(2*f))}
    raise ValueError(f"unknown mutation {family}")
