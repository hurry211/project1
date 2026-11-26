"""Sparse floating-point proposals. Certification uses independent exact arithmetic."""

from fractions import Fraction

import numpy as np
from scipy.sparse import coo_matrix


def sparse_rows(rows):
    """Convert the oracle's dictionary rows once, without materializing a dense matrix."""
    entries = [(i, j, float(value)) for i, row in enumerate(rows) for j, value in row.items() if value]
    if not entries:
        return coo_matrix((len(rows), len(rows))).tocsr()
    indices, columns, values = zip(*entries, strict=True)
    return coo_matrix((values, (indices, columns)), shape=(len(rows), len(rows))).tocsr()


def cg_steps(matrix, rhs, iterations, precondition=None):
    """Yield live PCG iterates; callers can inspect them without restarting CG."""
    diagonal = matrix.diagonal()
    if precondition is None:
        precondition = lambda residual: residual / diagonal
    x = np.zeros_like(rhs)
    residual = rhs.copy()
    image = precondition(residual)
    work = residual @ image
    direction = image.copy()
    initial = np.linalg.norm(residual)
    for step in range(1, iterations + 1):
        product = matrix @ direction
        curvature = direction @ product
        if not (np.isfinite(work) and work > 0 and np.isfinite(curvature) and curvature > 0):
            break
        alpha = work / curvature
        x += alpha * direction
        residual -= alpha * product
        yield step, x, residual
        if np.linalg.norm(residual) <= 1e-14 * initial:
            break
        image = precondition(residual)
        work_next = residual @ image
        direction = image + (work_next / work) * direction
        work = work_next


def dyadic_field(values):
    """Freeze the exact binary value of each finite double for rational replay."""
    return tuple(Fraction(float(value)) for value in values)
