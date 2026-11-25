"""Independent exact integration and proof checks for the CFAC unit-square contract."""

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from math import isqrt

from .spec import Problem, digest_parts, ordered, rational


def _det(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _pairs(t):
    """CCW boundary edges opposite vertices 0, 1, 2, respectively."""
    return ((t[1], t[2]), (t[2], t[0]), (t[0], t[1]))


def _key(i, j):
    return (i, j) if i < j else (j, i)


@lru_cache(maxsize=16)
def edges(mesh):
    return tuple(sorted({_key(i, j) for t in mesh.triangles for i, j in _pairs(t)}))


@dataclass(frozen=True)
class _Cell:
    tri: tuple
    points: tuple
    twice_area: Q
    grads: tuple
    edge_ids: tuple
    signs: tuple


def _boundary_side(p, q):
    for axis in (0, 1):
        if p[axis] == q[axis] and p[axis] in (0, 1):
            return axis, p[axis]
    raise ValueError("unexpected one-sided edge inside the domain")


@lru_cache(maxsize=16)
def _geometry(mesh):
    vertices, triangles = mesh.vertices, mesh.triangles
    if not triangles or len(set(vertices)) != len(vertices):
        raise ValueError("empty mesh or duplicate vertex")
    if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in vertices):
        raise ValueError("vertex outside the square")
    if {i for t in triangles for i in t} != set(range(len(vertices))):
        raise ValueError("unused or out-of-range vertex")
    indices = {e: j for j, e in enumerate(edges(mesh))}
    incidences = defaultdict(list)
    cells = []
    total = Q()
    for t in triangles:
        points = tuple(vertices[i] for i in t)
        twice_area = _det(*points)
        if twice_area <= 0:
            raise ValueError("nonpositive or clockwise triangle")
        total += twice_area
        grads = []
        edge_ids = []
        signs = []
        for i, j in _pairs(t):
            p, q = vertices[i], vertices[j]
            grads.append(((p[1] - q[1]) / twice_area, (q[0] - p[0]) / twice_area))
            e = _key(i, j)
            sign = 1 if i < j else -1
            edge_ids.append(indices[e])
            signs.append(sign)
            incidences[e].append(sign)
        cells.append(_Cell(t, points, twice_area, tuple(grads), tuple(edge_ids), tuple(signs)))
    if total != 2:
        raise ValueError("incorrect total area")
    boundary_vertices = set()
    boundary_edges = set()
    sides = defaultdict(list)
    for e, signs in incidences.items():
        if len(signs) == 2 and sum(signs) == 0:
            continue
        if len(signs) != 1:
            raise ValueError("edge incidence or orientation mismatch")
        p, q = (vertices[i] for i in e)
        axis, side = _boundary_side(p, q)
        extent = tuple(sorted((p[1 - axis], q[1 - axis])))
        sides[axis, side].append(extent)
        boundary_vertices.update(e)
        boundary_edges.add(indices[e])
    for axis in (0, 1):
        for side in (0, 1):
            end = Q()
            for lo, hi in sorted(sides[axis, side]):
                if lo != end or hi <= lo:
                    raise ValueError("overlap or gap on a boundary side")
                end = hi
            if end != 1:
                raise ValueError("incomplete boundary")
    # The surviving oriented triangle boundary is exactly the positive square.
    # Away from edges, its winding number counts covering CCW triangles with
    # multiplicity. It equals one everywhere in the open square. Thus interiors
    # cannot cross or overlap. Oppositely paired whole edges exclude hanging nodes.
    return tuple(cells), frozenset(boundary_vertices), frozenset(boundary_edges)


def _load_values(problem, cell, index):
    mean = problem.loads[index]
    if problem.original_affine is None:
        return (mean,) * 3
    c, a, b = problem.original_affine
    values = tuple(c + a * x + b * y for x, y in cell.points)
    if sum(values) != 3 * mean:
        raise ValueError("cell load disagrees with the original affine load projection")
    return values


def _field(values, size, boundary=()):
    result = tuple(rational(x) for x in ordered(values))
    if len(result) != size or any(result[i] != 0 for i in boundary):
        raise ValueError("invalid field size or homogeneous boundary trace")
    return result


def validate_problem(problem, trusted_digest):
    """Check the retained specification before a controller constructs fields."""
    if type(problem) is not Problem or problem.digest() != trusted_digest:
        raise ValueError("frozen problem digest mismatch")
    cells, _, _ = _geometry(problem.mesh)
    for index, cell in enumerate(cells):
        _load_values(problem, cell, index)


def _prepare(problem, trusted_digest, primal):
    if type(problem) is not Problem or problem.digest() != trusted_digest:
        raise ValueError("frozen problem digest mismatch")
    cells, boundary_vertices, boundary_edges = _geometry(problem.mesh)
    primal = _field(primal, len(problem.mesh.vertices), boundary_vertices)
    return cells, boundary_vertices, boundary_edges, primal


def _grad(cell, primal):
    return tuple(sum(primal[i] * g[d] for i, g in zip(cell.tri, cell.grads)) for d in (0, 1))


def _sqrt_upper(value):
    """Exact rational square root when possible, else round up on a binary grid."""
    n, d = value.numerator, value.denominator
    rn, rd = isqrt(n), isqrt(d)
    if rn * rn == n and rd * rd == d:
        return Q(rn, rd)
    scale = 2**96
    return Q(isqrt(n * scale * scale // d) + 1, scale)


@dataclass(frozen=True)
class Certificate:
    status: str
    accepted: bool
    exact_equilibrium: bool  # With respect to the frozen P0 load
    gap_sq: Q | None
    upper_sq: Q | None
    state_digest: str | None
    defect_sq: Q | None  # Unweighted L2 norm squared of f_h + div(q)
    oscillation_sq: Q | None  # Unweighted L2 norm squared of f - f_h
    reason: str


def check(problem, trusted_digest, primal, flux):
    """Certify the original PDE energy error with an RT0 field and exact integrals."""
    try:
        cells, _, _, primal = _prepare(problem, trusted_digest, primal)
        flux = _field(flux, len(edges(problem.mesh)))
        gap = defect = oscillation = Q()
        k = problem.diffusion
        for index, cell in enumerate(cells):
            grads = _grad(cell, primal)
            outward = tuple(sign * flux[e] for sign, e in zip(cell.signs, cell.edge_ids))
            div = 2 * sum(outward) / cell.twice_area
            area = cell.twice_area / 2
            defect += area * (problem.loads[index] + div) ** 2
            loads = _load_values(problem, cell, index)
            # Three edge midpoints integrate all degree-two polynomials exactly.
            for i, j in ((1, 2), (2, 0), (0, 1)):
                coords = tuple((cell.points[i][d] + cell.points[j][d]) / 2 for d in (0, 1))
                mismatch = []
                for d in (0, 1):
                    q = sum(F * (coords[d] - p[d]) for F, p in zip(outward, cell.points))
                    mismatch.append(q / cell.twice_area - k * grads[d])
                gap += area * sum(a * a for a in mismatch) / (3 * k)
                oscillation += area * ((loads[i] + loads[j]) / 2 - problem.loads[index]) ** 2 / 3
        # The cell-average projection makes the two residuals L2-orthogonal.
        # C_F = 1/(sqrt(2)*pi) <= 1/(3*sqrt(2)) on the unit square.
        correction = (defect + oscillation) / (18 * k)
        upper = gap + correction + 2 * _sqrt_upper(gap * correction)
        state = digest_parts([trusted_digest, list(map(str, primal)), list(map(str, flux))])
        return Certificate("CERTIFIED", upper <= problem.epsilon**2, defect == 0, gap, upper,
                           state, defect, oscillation, "validated original-PDE upper bound")
    except (ValueError, TypeError, OverflowError) as exc:
        return Certificate("ESTIMATED", False, False, None, None, None, None, None, str(exc))


def witness_lower_sq(problem, trusted_digest, primal, witness_vertices, witness_edges):
    """Exact original-PDE residual²/energy on a conforming diagnostic P2 function."""
    cells, boundary_vertices, boundary_edges, primal = _prepare(problem, trusted_digest, primal)
    wv = _field(witness_vertices, len(primal), boundary_vertices)
    we = _field(witness_edges, len(edges(problem.mesh)), boundary_edges)
    residual = energy = Q()
    centroid = (Q(1, 3),) * 3
    # The four-point Hammer rule integrates total degree three exactly.
    rule = [(centroid, Q(-27, 48))]
    for i in range(3):
        rule.append((tuple(Q(3 if j == i else 1, 5) for j in range(3)), Q(25, 48)))
    for index, cell in enumerate(cells):
        loads = _load_values(problem, cell, index)
        verts = tuple(wv[i] for i in cell.tri)
        mids = tuple(we[i] for i in cell.edge_ids)
        if not any(verts) and not any(mids):
            continue
        grad_v = _grad(cell, primal)
        for lam, weight in rule:
            w = sum(v * t * (2 * t - 1) for v, t in zip(verts, lam))
            grad_w = [sum(v * (4 * t - 1) * g[d] for v, t, g in zip(verts, lam, cell.grads))
                      for d in (0, 1)]
            for m, (i, j) in zip(mids, ((1, 2), (2, 0), (0, 1))):
                w += 4 * m * lam[i] * lam[j]
                for d in (0, 1):
                    grad_w[d] += 4 * m * (lam[i] * cell.grads[j][d] + lam[j] * cell.grads[i][d])
            factor = cell.twice_area * weight / 2
            f = sum(t * load for t, load in zip(lam, loads))
            pairing = sum(a * b for a, b in zip(grad_v, grad_w))
            residual += factor * (f * w - problem.diffusion * pairing)
            energy += factor * problem.diffusion * sum(g * g for g in grad_w)
    return residual**2 / energy if energy else Q()
