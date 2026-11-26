"""Sparse P2 residual witness for certifying an unaffordable primal floor."""

from fractions import Fraction as Q

import numpy as np
from scipy.sparse.linalg import spsolve

from .checker import _geometry, edges, witness_lower_sq
from .sparse_linear import dyadic_field
from .sparse_rt1 import quadratic_stream_geometry
from .stopping import inspect_stopping


def screen_primal_floor(problem, trusted_digest, primal):
    """Propose a quadratic residual witness and validate its energy exactly."""
    if problem.original_affine is not None:
        raise ValueError("the P2 witness proposal is assembled for P0 loads")
    cells, boundary, boundary_edges = _geometry(problem.mesh)
    edge_list = edges(problem.mesh)
    n = len(problem.mesh.vertices)
    size = n + len(edge_list)
    ids, _, weights, gram = quadratic_stream_geometry(cells, n, size)
    matrix = float(problem.diffusion) * gram
    free = np.ones(size, dtype=bool)
    free[list(boundary)] = False
    free[[n + index for index in boundary_edges]] = False
    rhs = np.zeros(size)
    np.add.at(rhs, ids, (weights*np.asarray(problem.loads, dtype=float))[:, None])
    residual = rhs - matrix @ np.r_[np.asarray(primal, dtype=float), np.zeros(len(edge_list))]
    response = np.zeros(size)
    response[free] = spsolve(matrix[free][:, free], residual[free])
    coefficients = dyadic_field(response)
    mids = [coefficients[n + e] + (coefficients[i] + coefficients[j])/2
            for e, (i, j) in enumerate(edge_list)]
    lower = witness_lower_sq(problem, trusted_digest, primal, coefficients[:n], mids)
    upper_alg = inspect_stopping(problem, trusted_digest, primal, "classical").algebraic_upper_sq
    return max(Q(0), lower - upper_alg)
