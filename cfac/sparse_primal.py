"""Sparse P1 proposals and an exact feasible RT0 seed on the frozen square."""

from fractions import Fraction as Q

import numpy as np
from scipy.sparse import coo_matrix

from .checker import _geometry, edges, validate_problem
from .prototype import feasible_rt0
from .sparse_linear import cg_steps, dyadic_field


class PrimalSystem:
    def __init__(self, problem, trusted_digest):
        validate_problem(problem, trusted_digest)
        cells, boundary, _ = _geometry(problem.mesh)
        size = len(problem.mesh.vertices)
        self.free = np.array([i for i in range(size) if i not in boundary], dtype=np.int64)
        self.size = size
        area = np.array([float(c.twice_area)/2 for c in cells])
        a = float(problem.diffusion)
        ids = np.array([c.tri for c in cells], dtype=np.int64)
        grads = np.array([c.grads for c in cells], dtype=float)
        local = a * np.einsum("c,cid,cjd->cij", area, grads, grads)
        row = np.broadcast_to(ids[:, :, None], local.shape)
        col = np.broadcast_to(ids[:, None, :], local.shape)
        self.matrix = coo_matrix((local.ravel(), (row.ravel(), col.ravel())), shape=(size, size)).tocsr()
        self.rhs = np.zeros(size)
        np.add.at(self.rhs, ids, (area*np.asarray(problem.loads, dtype=float)/3)[:, None])
        self.flux = feasible_rt0([tuple(zip(c.edge_ids, c.signs, strict=True)) for c in cells],
                                 [-c.twice_area * f / 2 for c, f in zip(cells, problem.loads, strict=True)],
                                 len(edges(problem.mesh)))

    def proposals(self, initial, iterations):
        values = np.asarray(initial, dtype=float)
        residual = (self.rhs - self.matrix @ values)[self.free]
        matrix = self.matrix[self.free][:, self.free]
        for step, update, remainder in cg_steps(matrix, residual, iterations):
            field = values.copy()
            field[self.free] += update
            yield step, field, np.linalg.norm(remainder)

    def zero(self):
        return (Q(0),) * self.size

    def frozen(self, values):
        if len(values) != self.size:
            raise ValueError("wrong primal field size")
        return dyadic_field(values)
