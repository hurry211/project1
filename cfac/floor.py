"""Exact algebraic energy bounds and floor attribution for a fixed P1/RT0 pair."""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from .checker import Certificate, check, edges, witness_lower_sq
from .spec import ordered, rational


ZERO = Fraction(0)


@dataclass(frozen=True)
class Bounds:
    """Closed interval for a squared energy."""

    lower: Fraction
    upper: Fraction


@dataclass(frozen=True)
class FloorAssessment:
    certificate: Certificate
    verdict: str
    primal_alg: Bounds | None = None
    dual_alg: Bounds | None = None
    floor: Bounds | None = None
    witness_lower_sq: Fraction | None = None
    primal_disc_lower_sq: Fraction | None = None

    @property
    def state_digest(self):
        return self.certificate.state_digest

    def assert_current(self, problem, trusted_digest, primal, flux):
        """Fail closed before using a previously computed in-process decision."""
        current = check(problem, trusted_digest, primal, flux)
        if not self.state_digest or current.state_digest != self.state_digest:
            raise ValueError("floor assessment belongs to a different state")


def _dot(x, y):
    return sum((a * b for a, b in zip(x, y)), ZERO)


def _matvec(rows, x):
    return [sum((a * x[j] for j, a in row.items()), ZERO) for row in rows]


def _distances(neighbors, anchors):
    """Distances to a pinned zero along mesh edges."""
    distance = {v: 0 for v in anchors}
    queue = deque(distance)
    while queue:
        v = queue.popleft()
        for w in neighbors[v]:
            if w not in distance:
                distance[w] = distance[v] + 1
                queue.append(w)
    return distance


def _exact_energy(rows, residual):
    """Accumulate r' K^-1 r by scalar Schur complements without forming an optimizer."""
    size = len(rows)
    matrix = [[row.get(j, ZERO) for j in range(size)] + [rhs]
              for row, rhs in zip(rows, residual)]
    energy = ZERO
    for i in range(size):
        pivot = matrix[i][i]
        if pivot <= 0:
            raise ArithmeticError("nonpositive pivot in reduced energy system")
        energy += matrix[i][-1]**2 / pivot
        for j in range(i + 1, size):
            factor = matrix[j][i] / pivot
            if factor:
                for k in range(i + 1, size + 1):
                    matrix[j][k] -= factor * matrix[i][k]
    return energy


def _bounds(rows, residual, distance, edge_bound, gap, tight):
    """Enclose r' K^-1 r by a trial correction and anchored path inequalities."""
    if not any(residual):
        return Bounds(ZERO, ZERO)
    if tight:
        energy = _exact_energy(rows, residual)
        return Bounds(energy, energy)
    # Exact line search on D^-1 r. Its gain is an attained lower bound.
    direction = [r / row[i] for i, (row, r) in enumerate(zip(rows, residual))]
    image = _matvec(rows, direction)
    length = _dot(direction, residual) / _dot(direction, image)
    lower = length * _dot(direction, residual)
    remainder = [r - length * y for r, y in zip(residual, image)]
    # |z_i| <= distance[i] * sqrt(edge_bound) * ||z||_K.
    # Apply to the new residual in exactly the same reduced coordinates.
    rest = sum((d * abs(r) for d, r in zip(distance, remainder)), ZERO)
    return Bounds(lower, min(gap, lower + edge_bound * rest**2))


def _sparse_bounds(rows, residual, distance, edge_bound, masses, gap, neumann, iterations):
    """Enclose the residual energy of a floating trial using exact P1 operations."""
    import numpy as np

    from .sparse_linear import cg_steps, dyadic_field, sparse_rows

    old = _bounds(rows, residual, distance, edge_bound, gap, False)
    if not any(residual):
        return old
    trial = None
    for _, trial, _ in cg_steps(sparse_rows(rows), np.array(residual, dtype=float), iterations):
        pass
    if trial is None or not np.isfinite(trial).all():
        return old
    trial = dyadic_field(trial)
    image = _matvec(rows, trial)
    gain = 2 * _dot(residual, trial) - _dot(trial, image)
    remainder = [r - y for r, y in zip(residual, image, strict=True)]
    if neumann:
        # Restore the omitted reaction. It vanishes against constants in the quotient space.
        remainder = [-sum(remainder), *remainder]
    # Consistent P1 mass >= one quarter of lumped mass. On the unit square
    # Poincare gives 1/18 (Dirichlet) or 1/9 (mean-zero Neumann), before diffusion scaling.
    weight = masses[1]
    correction = 4 * sum(r*r/m for r, m in zip(remainder, masses[0], strict=True))
    correction /= weight * (9 if neumann else 18)
    return Bounds(max(old.lower, gain), min(old.upper, gain + correction))


def _assemble(problem, primal, flux, *, include_masses=False):
    """Assemble variational residuals directly from cell means and nodal fields."""
    vertices = problem.mesh.vertices
    edge_list = edges(problem.mesh)
    edge_flux = dict(zip(edge_list, flux))
    a = problem.diffusion
    count = len(vertices)
    stiffness = [{} for _ in vertices]
    primal_rhs = [ZERO] * count
    dual_rhs = [ZERO] * count
    masses = [ZERO] * count if include_masses else None
    neighbors = [set() for _ in vertices]
    multiplicity = dict.fromkeys(edge_list, 0)
    max_edge = ZERO
    for cell, load in zip(problem.mesh.triangles, problem.loads):
        points = [vertices[v] for v in cell]
        x0, y0 = points[0]
        x1, y1 = points[1]
        x2, y2 = points[2]
        det = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)
        area = det / 2
        cx, cy = (x0+x1+x2)/3, (y0+y1+y2)/3
        gradients = []
        qx, qy = ZERO, ZERO
        for i, opposite in enumerate(cell):
            j, k = cell[(i+1) % 3], cell[(i+2) % 3]
            dx, dy = vertices[k][0]-vertices[j][0], vertices[k][1]-vertices[j][1]
            gradients.append((-dy / det, dx / det))
            edge = (min(j, k), max(j, k))
            multiplicity[edge] += 1
            neighbors[j].add(k)
            neighbors[k].add(j)
            max_edge = max(max_edge, (dx*dx + dy*dy) / area)
            outward = edge_flux[edge] if j < k else -edge_flux[edge]
            qx += outward * (cx-vertices[opposite][0]) / det
            qy += outward * (cy-vertices[opposite][1]) / det
        for i, (v, (gx, gy)) in enumerate(zip(cell, gradients)):
            if masses is not None:
                masses[v] += area / 3
            primal_rhs[v] += load * area / 3
            dual_rhs[v] -= area * (qx*gy - qy*gx) / a
            for w, gw in zip(cell, gradients):
                stiffness[v][w] = stiffness[v].get(w, ZERO) + area * _dot(gradients[i], gw)
    if sum(dual_rhs) != 0:
        raise ArithmeticError("dual residual does not annihilate constant streams")
    # The reduced dual Gram matrix fixes stream vertex zero, removing precisely
    # the one constant kernel on a connected simply connected mesh.
    boundary = {v for edge, n in multiplicity.items() if n == 1 for v in edge}
    primal_rhs = [b - a*y for b, y in zip(primal_rhs, _matvec(stiffness, primal))]

    def reduce(anchors, rhs, weight):
        distance = _distances(neighbors, anchors)
        free = [v for v in range(count) if v not in anchors]
        index = {v: i for i, v in enumerate(free)}
        rows = [{index[w]: k * weight for w, k in stiffness[v].items() if w in index} for v in free]
        return rows, [rhs[v] for v in free], [distance[v] for v in free], max_edge / weight

    systems = (reduce(boundary, primal_rhs, a), reduce({0}, dual_rhs, 1 / a), multiplicity)
    if masses is None:
        return systems
    return (*systems, ([m for v, m in enumerate(masses) if v not in boundary], a), (masses, 1 / a))


def _probe(problem, trusted_digest, primal, multiplicity):
    """Try a conforming quadratic witness with unit interior edge midpoints."""
    vertex_values = (ZERO,) * len(primal)
    midpoints = tuple(int(multiplicity[e] == 2) for e in edges(problem.mesh))
    return witness_lower_sq(problem, trusted_digest, primal, vertex_values, midpoints)


def assess(problem, trusted_digest, primal, flux, *, tight=False, witness=False, sparse_iterations=0):
    """Inspect one frozen state. Inputs contain no manufactured or reference truth.

    The default uses a Jacobi trial and a conservative graph upper bound. Tight
    mode evaluates both residual energies by exact Schur elimination on small meshes.
    The optional P2 diagnostic never modifies the P1 state.
    """
    if type(sparse_iterations) is not int or sparse_iterations < 0 or (tight and sparse_iterations):
        raise ValueError("select exact elimination or a nonnegative sparse iteration count")
    try:
        primal, flux = ordered(primal), ordered(flux)
    except (TypeError, ValueError):
        return FloorAssessment(check(problem, trusted_digest, None, None), "UNDETERMINED")
    certificate = check(problem, trusted_digest, primal, flux)
    accepted = certificate.accepted
    if not (certificate.exact_equilibrium and certificate.oscillation_sq == 0
            and certificate.status == "CERTIFIED"):
        return FloorAssessment(certificate, "ACCEPTED" if accepted else "UNDETERMINED")
    primal = tuple(map(rational, primal))
    flux = tuple(map(rational, flux))
    gap = certificate.gap_sq
    if sparse_iterations:
        p_system, d_system, multiplicity, p_mass, d_mass = _assemble(problem, primal, flux, include_masses=True)
        p = _sparse_bounds(*p_system, p_mass, gap, False, sparse_iterations)
        d = _sparse_bounds(*d_system, d_mass, gap, True, sparse_iterations)
    else:
        p_system, d_system, multiplicity = _assemble(problem, primal, flux)
        p = _bounds(*p_system, gap, tight)
        d = _bounds(*d_system, gap, tight)
    floor = Bounds(max(ZERO, gap - p.upper - d.upper), gap - p.lower - d.lower)
    witness_sq = _probe(problem, trusted_digest, primal, multiplicity) if witness else ZERO
    primal_floor = max(ZERO, witness_sq - p.upper)
    if not (0 <= p.lower <= p.upper and 0 <= d.lower <= d.upper
            and primal_floor <= floor.upper and floor.lower <= floor.upper):
        raise ArithmeticError("inconsistent certified bounds")
    threshold = problem.epsilon**2
    if accepted:
        verdict = "ACCEPTED"
    elif primal_floor > threshold:
        verdict = "PRIMAL_SPACE_FUTILE"
    elif floor.lower > threshold:
        verdict = "PAIR_FUTILE"
    elif floor.upper <= threshold:
        verdict = "PAIR_SUFFICIENT"
    else:
        verdict = "UNDETERMINED"
    return FloorAssessment(certificate, verdict, p, d, floor, witness_sq, primal_floor)
