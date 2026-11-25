"""Independent tensor finite-element verification for the manufactured field families."""
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import linalg, sparse

END_TIME = 1 / 32
FAMILIES = ("heat", "wave", "elasticity")


def line_operators(n, order):
    """Assemble mass, stiffness and test-gradient matrices by independent Gauss integration."""
    points, weights = np.polynomial.legendre.leggauss(8)
    t = (points + 1) / 2
    weights = weights / (2 * n)
    nodes = np.linspace(0, 1, order + 1)
    polys = [np.poly1d(np.poly(np.delete(nodes, i))) / np.prod(nodes[i] - np.delete(nodes, i))
             for i in range(order + 1)]
    phi = np.column_stack([f(t) for f in polys])
    grad = n * np.column_stack([np.polyder(f)(t) for f in polys])
    dofs = order * np.arange(n)[:, None] + np.arange(order + 1)
    size = n * order + 1

    def assemble(left, right):
        local = left.T @ (weights[:, None] * right)
        rows = np.repeat(dofs, order + 1, axis=1).ravel()
        cols = np.tile(dofs, (1, order + 1)).ravel()
        return sparse.coo_matrix((np.tile(local.ravel(), n), (rows, cols)), shape=(size, size)).tocsr()

    xi = ((np.arange(n)[:, None] + t) / n).ravel()
    rows = np.repeat(np.arange(len(xi)), order + 1)
    cols = np.repeat(dofs, len(t), axis=0).ravel()
    interp = sparse.coo_matrix((np.tile(phi.ravel(), n), (rows, cols)), shape=(len(xi), size)).tocsr()
    return assemble(phi, phi), assemble(grad, grad), assemble(phi, grad), interp, xi, np.tile(weights, n)


def axis_apply(op, field, axis):
    v = np.moveaxis(field, axis, 0)
    return np.moveaxis((op @ v.reshape(v.shape[0], -1)).reshape((op.shape[0], *v.shape[1:])), 0, axis)


def tensor_apply(ops, field):
    for axis, op in enumerate(ops):
        field = axis_apply(op, field, axis)
    return field


def product(vectors):
    result = vectors[0]
    for other in vectors[1:]:
        result = np.multiply.outer(result, other)
    return result


def transient_reference(family, n, order, d, steps, mass, stiff):
    # Generalized eigenpairs on the free 1D nodes diagonalize the complete
    # tensor Galerkin system. This implementation never calls the MFEM solver.
    m = mass[1:-1, 1:-1].toarray()
    k = stiff[1:-1, 1:-1].toarray()
    lam, modes = linalg.eigh(k, m)
    coeff = modes.T @ m @ np.sin(np.pi * np.arange(1, n * order) / (n * order))
    initial = product([coeff] * d)
    eig = sum(lam.reshape([len(lam) if i == j else 1 for i in range(d)]) for j in range(d))
    dt = END_TIME / steps
    for elapsed in (steps // 2, steps):
        if family == "heat":
            factors = [((1 - dt*eig/2) / (1 + dt*eig/2)) ** elapsed]
        else:
            freq = np.sqrt(eig)
            angle = elapsed * 2 * np.arctan(dt * freq / 2)
            factors = [np.cos(angle), -freq * np.sin(angle)]
        for factor in factors:
            yield tensor_apply([modes] * d, factor * initial)


def elasticity_residual(fields, mass, stiff, derivative, interp, xi, weights):
    # With zero traces, the isotropic form is grad(u):grad(v) + 2 div(u)div(v).
    d = len(fields)
    sine = interp.T @ (weights * np.sin(np.pi * xi))
    cosine = interp.T @ (weights * np.cos(np.pi * xi))
    interior = (slice(1, -1),) * d
    residual2 = load2 = 0.
    for i, ui in enumerate(fields):
        action = np.zeros_like(ui)
        for j in range(d):
            ops = [mass] * d
            ops[j] = stiff
            action += tensor_apply(ops, ui)
            ops = [mass] * d
            if i == j:
                ops[i] = stiff
            else:
                ops[i], ops[j] = derivative.T, derivative
            action += 2 * tensor_apply(ops, fields[j])
        if i == 0:
            rhs = (d + 2) * np.pi**2 * product([sine] * d)
        else:
            rhs = -2 * np.pi**2 * product([cosine if j in (0, i) else sine for j in range(d)])
        residual2 += np.sum((action[interior] - rhs[interior])**2)
        load2 += np.sum(rhs[interior]**2)
    return float(np.sqrt(residual2 / load2))


def error_squared(field, amplitude, interp, xi, weights):
    """Integrate in slabs to bound temporary storage on 3D tensor grids."""
    d = field.ndim
    for axis in range(1, d):
        field = axis_apply(interp, field, axis)
    sine = np.sin(np.pi * xi)
    rest = product([sine] * (d - 1))
    other_weights = product([weights] * (d - 1))
    total = 0.
    for start in range(0, len(xi), 8):
        sl = slice(start, start + 8)
        diff = axis_apply(interp[sl], field, 0) - amplitude * np.multiply.outer(sine[sl], rest)
        total += np.sum(np.multiply.outer(weights[sl], other_weights) * diff**2)
    return float(total)


def verify_level(path, family, d, n, order, steps):
    count = n * order + 1
    columns = {"heat": 2, "wave": 4, "elasticity": d}[family]
    raw = np.loadtxt(path, ndmin=2)
    if raw.shape != (count**d, d + columns) or not np.isfinite(raw).all():
        raise ValueError("unexpected field shape or nonfinite values")
    grid = np.rint(raw[:, :d] * (count - 1)).astype(int)
    if np.any(grid < 0) or np.any(grid >= count) or np.max(np.abs(raw[:, :d] - grid / (count - 1))) > 1e-12:
        raise ValueError("invalid nodal coordinates")
    indices = np.ravel_multi_index(grid.T, (count,) * d)
    if len(np.unique(indices)) != count**d:
        raise ValueError("incomplete nodal lattice")
    ordered = np.empty((count**d, columns))
    ordered[indices] = raw[:, d:]
    fields = [col.reshape((count,) * d) for col in ordered.T]
    boundary = ((grid == 0) | (grid == count - 1)).any(axis=1)
    trace = float(np.max(np.abs(raw[boundary, d:])))
    mass, stiff, deriv, interp, xi, weights = line_operators(n, order)
    interior = (slice(1, -1),) * d
    if family == "elasticity":
        check = elasticity_residual(fields, mass, stiff, deriv, interp, xi, weights)
        err = sum(error_squared(f, float(i == 0), interp, xi, weights) for i, f in enumerate(fields))
        amplitude = 1.
    else:
        reference = transient_reference(family, n, order, d, steps, mass, stiff)
        check = max(float(np.max(np.abs(f[interior] - ref)) / np.max(np.abs(ref)))
                    for f, ref in zip(fields, reference, strict=True))
        amplitude = (math.exp(-d * math.pi**2 * END_TIME) if family == "heat" else
                     math.cos(math.sqrt(d) * math.pi * END_TIME))
        err = error_squared(fields[-1 if family == "heat" else -2], amplitude, interp, xi, weights)
    relative_l2 = math.sqrt(err * 2**d) / abs(amplitude)
    # The time discretizations are second order. Spatial convergence is checked
    # separately over three meshes in the study. Elasticity is static.
    target = 12 / n**(order+1) + (0 if family == "elasticity" else 100 * (END_TIME / steps)**2)
    return {"n": n, "scalar_nodes": count**d, "system_dofs": count**d * (d if family == "elasticity" else 1),
            "boundary_linf": trace, "independent_measure": "relative_weak_residual" if family == "elasticity" else "relative_nodal_linf",
            "independent_error": check, "independent_limit": 1e-8, "relative_l2": relative_l2, "l2_limit": target,
            "passed": bool(trace <= 1e-10 and check <= 1e-8 and relative_l2 <= target)}


if __name__ == "__main__":
    field, family, d, n, p, steps = sys.argv[1:]
    print(json.dumps(verify_level(Path(field), family, int(d), int(n), int(p), int(steps)), allow_nan=False))
