"""Equilibrium-preserving sparse RT1 corrections on the frozen P0 contract."""

from fractions import Fraction as Q
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix, eye, vstack
from scipy.sparse.linalg import splu

from .checker import _geometry, check, edges
from .enrichment import EnrichmentRequest, RTMoments
from .floor import Bounds
from .rt1_checker import RT1Checker, _field_on_cell
from .sparse_linear import cg_steps, dyadic_field


PAIRS = ((1, 2), (2, 0), (0, 1))
MIDPOINTS = tuple(tuple(Q(i != j, 2) for i in range(3)) for j in range(3))


def _curls(cell, lam):
    linear = tuple((gy, -gx) for gx, gy in cell.grads)
    return (*linear, *(tuple(4 * (lam[i]*b + lam[j]*a) for a, b in zip(linear[i], linear[j]))
                       for i, j in PAIRS))


def embed_rt0(request):
    """Represent the current RT0 flux exactly in RT1 moments."""
    problem = request.problem
    cells, _, _ = _geometry(problem.mesh)
    face_moments = [(q, q/2) for q in request.rt0_flux]
    cell_moments = []
    for cell in cells:
        centroid = tuple(sum(x)/3 for x in zip(*cell.points))
        cell_moments.append(tuple(sum(sign * request.rt0_flux[e] * (mid - point[d])
                                      for e, sign, point in zip(cell.edge_ids, cell.signs, cell.points)) / 2
                                  for d, mid in enumerate(centroid)))
    return RTMoments(request.problem_digest, request.parent_state_digest, 1, face_moments, cell_moments)


def dual_rt1_bounds(request, candidate, certificate):
    """Exact mass/Poincare upper and element-diagonal lower for a checked RT1 flux."""
    if (not certificate.exact_equilibrium or certificate.oscillation_sq
            or certificate.state_digest != candidate.digest()):
        raise ValueError("dual attribution requires checked P0 equilibrium and bound state")
    candidate.validate_layout(request)
    p = request.problem
    cells, _, _ = _geometry(p.mesh)
    n = len(p.mesh.vertices)
    size = n + len(candidate.face_moments)
    residual = [Q(0)] * size
    weights = [Q(0)] * size
    diagonal = [Q(0)] * size
    for cell, moment in zip(cells, candidate.cell_moments, strict=True):
        ids = (*cell.tri, *(n + e for e in cell.edge_ids))
        q = _field_on_cell(cell, candidate.face_moments, moment)[:2]
        area = cell.twice_area / 2
        for i in ids:
            weights[i] += area
        for lam in MIDPOINTS:
            curls = _curls(cell, lam)
            # Convert the three linear hats to the true nodal P2 vertex functions.
            nodal = [tuple(value - sum(edge[d] for k, edge in enumerate(curls[3:]) if k != j)/2
                           for d, value in enumerate(curls[j])) for j in range(3)]
            sample = [z[0] + z[1]*lam[1] + z[2]*lam[2] for z in q]
            weight = area / (3*p.diffusion)
            for i, g in zip(ids, (*nodal, *curls[3:]), strict=True):
                residual[i] -= weight * sum(a*b for a, b in zip(sample, g, strict=True))
                diagonal[i] += weight * sum(a*a for a in g)
    if sum(residual):
        raise ArithmeticError("RT1 residual fails the constant kernel condition")
    lower = sum(r*r/d for r, d in zip(residual[1:], diagonal[1:], strict=True)) / 6
    upper = 57 * p.diffusion / 9 * sum(r*r/m for r, m in zip(residual, weights, strict=True))
    return Bounds(lower, min(upper, certificate.gap_sq))


def quadratic_stream_geometry(cells, n, size):
    """Sparse P2 Gram, with P1 hats followed by quadratic edge bubbles."""
    ids = np.array([(*cell.tri, *(n + e for e in cell.edge_ids)) for cell in cells], dtype=np.int64)
    gradient = np.array([cell.grads for cell in cells], dtype=float)
    linear = gradient[..., ::-1] * np.array([1., -1.])
    lam = np.asarray(MIDPOINTS, dtype=float)
    modes = np.empty((len(cells), 3, 6, 2))
    modes[:, :, :3] = linear[:, None, :, :]
    for k, (i, j) in enumerate(PAIRS):
        modes[:, :, k + 3] = 4 * (lam[None, :, i, None]*linear[:, None, j]
                                  + lam[None, :, j, None]*linear[:, None, i])
    weight = np.array([float(c.twice_area)/6 for c in cells])
    local = np.einsum("c,cqid,cqjd->cij", weight, modes, modes)
    row = np.broadcast_to(ids[:, :, None], local.shape)
    col = np.broadcast_to(ids[:, None, :], local.shape)
    gram = coo_matrix((local.ravel(), (row.ravel(), col.ravel())), shape=(size, size)).tocsr()
    return ids, modes, weight, gram


class StreamSystem:
    """A quotient-space operator for the continuous P2 stream."""

    def __init__(self, request, seed, *, coarse):
        p = request.problem
        self.cells, _, _ = _geometry(p.mesh)
        self.edges = edges(p.mesh)
        self.n = len(p.mesh.vertices)
        size = self.n + len(self.edges)
        self.ids, modes, weight, gram = quadratic_stream_geometry(self.cells, self.n, size)
        self.matrix = gram[1:, 1:]  # Remove the global constant mode.
        polynomials = np.array([_field_on_cell(c, seed.face_moments, moment)[:2]
                                for c, moment in zip(self.cells, seed.cell_moments, strict=True)], dtype=float)
        r, s = np.asarray(MIDPOINTS, dtype=float)[:, 1:].T
        sample = polynomials @ np.stack([np.ones(3), r, s, r*r, r*s, s*s])
        rhs = np.zeros(size)
        np.add.at(rhs, self.ids, -np.einsum("c,cdq,cqid->ci", weight, sample, modes))
        self.rhs = rhs[1:]
        self.coarse = splu(self.matrix[:self.n - 1, :self.n - 1].tocsc()) if coarse else None
        self.inverse_diagonal = 1 / self.matrix.diagonal()

    def precondition(self, r):
        image = self.inverse_diagonal * r
        if self.coarse is not None:
            image[:self.n - 1] = self.coarse.solve(r[:self.n - 1])
        return image

    def materialize(self, request, seed, values):
        """Add one exactly divergence-free curl, even for a rounded iterate."""
        stream = (Q(0), *dyadic_field(values))
        faces = []
        for e, ((i, j), (mean, first)) in enumerate(zip(self.edges, seed.face_moments, strict=True)):
            shift = stream[j] - stream[i]
            faces.append((mean + shift, first + shift/2 - 2*stream[self.n + e]/3))
        interior = []
        centroid = (Q(1, 3),)*3
        for cell, ids, base in zip(self.cells, self.ids, seed.cell_moments, strict=True):
            modes = _curls(cell, centroid)
            interior.append(tuple(base[d] + cell.twice_area/2 *
                                  sum(stream[i]*mode[d] for i, mode in zip(ids, modes, strict=True))
                                  for d in (0, 1)))
        return RTMoments(request.problem_digest, request.parent_state_digest, 1, faces, interior)


def optimize_rt1(request, *, seed=None, coarse=True, schedule="predicted", iterations=128):
    """Propose cheaper certificates without moving the primal or authorizing acceptance."""
    if not isinstance(request, EnrichmentRequest) or request.order != 1 or request.problem.original_affine is not None:
        raise ValueError("sparse RT1 optimization requires the original P0 load contract")
    if schedule not in ("eager", "geometric", "predicted") or type(iterations) is not int or iterations <= 0:
        raise ValueError("invalid RT1 inspection schedule or iteration count")
    checker = RT1Checker()
    start = perf_counter()
    if seed is None:
        initial = check(request.problem, request.problem_digest, request.primal, request.rt0_flux)
        if initial.state_digest != request.parent_state_digest:
            raise ValueError("stale RT0 parent")
        seed = embed_rt0(request)
    else:
        initial = checker.verify(request, seed, seed.digest())
    if not initial.exact_equilibrium:
        raise ValueError("sparse stream corrections require an equilibrated seed")
    checking = perf_counter() - start
    result = {"checks": 1, "iterations": 0, "initial_upper_sq": str(initial.upper_sq),
              "setup_seconds": 0., "cg_seconds": 0., "check_seconds": checking, "materialize_seconds": 0.,
              "inspection_steps": [0], "upper_history_sq": [str(initial.upper_sq)]}
    if initial.accepted:
        if initial.state_digest != seed.digest():
            start = perf_counter()
            initial = checker.verify(request, seed, seed.digest())
            result["checks"] += 1
            result["check_seconds"] += perf_counter() - start
        return seed, initial, result
    start = perf_counter()
    system = StreamSystem(request, seed, coarse=coarse)
    result["setup_seconds"] = perf_counter() - start
    target_gain = float((initial.upper_sq - request.problem.epsilon**2) * request.problem.diffusion)
    candidate, final = seed, initial
    last = 0

    def inspect(step, values):
        nonlocal last, candidate, final
        start = perf_counter()
        candidate = system.materialize(request, seed, values)
        result["materialize_seconds"] += perf_counter() - start
        start = perf_counter()
        final = checker.verify(request, candidate, candidate.digest())
        result["check_seconds"] += perf_counter() - start
        result["checks"] += 1
        result["inspection_steps"].append(step)
        result["upper_history_sq"].append(str(final.upper_sq))
        last = step

    started = perf_counter()
    for step, values, residual in cg_steps(system.matrix, system.rhs, iterations, system.precondition):
        result["cg_seconds"] += perf_counter() - started
        result["iterations"] = step
        due = (schedule == "eager" or step == iterations
               or (schedule == "geometric" and step & (step - 1) == 0))
        if schedule == "predicted" and not due:
            # For exact PCG the gain is x.(b+r). Floating gain only schedules an inspection.
            due = (step == 1 or step >= 2 * last
                   or (step > last and float(values @ (system.rhs + residual)) >= target_gain))
        if due:
            inspect(step, values)
            if final.accepted:
                break
        started = perf_counter()
    if result["iterations"] > last:
        inspect(result["iterations"], values)
    elif not result["iterations"] and initial.state_digest != seed.digest():
        start = perf_counter()
        final = checker.verify(request, seed, seed.digest())
        result["checks"] += 1
        result["check_seconds"] += perf_counter() - start
    return candidate, final, result
