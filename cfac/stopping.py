"""Fixed residual and certified algebraic/discretization stopping for P1/P0.

These decisions stop primal iterations on one level. The independent field
checker owns all PDE acceptance decisions.
"""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction as Q

from .checker import _grad, _prepare
from .spec import digest_parts, rational


@dataclass(frozen=True)
class StoppingObservation:
    rule: str
    primal_digest: str
    stop: bool
    residual_sq: Q
    rhs_sq: Q
    algebraic_upper_sq: Q | None
    discretization_lower_sq: Q | None


def _inner(x, y):
    return sum((a * b for a, b in zip(x, y, strict=True)), Q(0))


def _tree_lift(cells, vertices, boundary, residual):
    """Return constant cell vectors L satisfying (L, grad w) = r.w on P1_0.

    Each directed child-to-parent transfer of nodal weight s contributes
    s*(x_child-x_parent)/area in one containing cell. Testing with grad(w)
    reproduces s*(w_child-w_parent). The tree telescopes at zero boundary
    values. Only discrete weak moments are imposed on this broken field.
    """
    neighbors = [dict() for _ in vertices]
    for k, cell in enumerate(cells):
        for i, j in zip(cell.tri, cell.tri[1:] + cell.tri[:1]):
            neighbors[i].setdefault(j, k)
            neighbors[j].setdefault(i, k)
    pending = deque(sorted(boundary))
    parent = dict.fromkeys(boundary)
    transfers = []
    while pending:
        i = pending.popleft()
        for j, k in neighbors[i].items():
            if j not in parent:
                parent[j] = i
                pending.append(j)
                transfers.append((j, i, k))
    if len(parent) != len(vertices):
        raise ValueError("mesh has a component without a Dirichlet anchor")
    weights = list(residual)
    field = [[Q(0), Q(0)] for _ in cells]
    for j, i, k in reversed(transfers):
        scale = 2 * weights[j] / cells[k].twice_area
        for d in (0, 1):
            field[k][d] += scale * (vertices[j][d] - vertices[i][d])
        weights[i] += weights[j]
    return field


def _algebraic_bound(cells, vertices, boundary, r, diagonal, diffusion):
    """Exact Jacobi line gain plus a Cauchy bound on the residual remainder."""
    if not any(r):
        return Q(0)
    direction = [ri / di for ri, di in zip(r, diagonal)]
    image = [Q(0)] * len(r)
    for cell in cells:
        g = _grad(cell, direction)
        area = cell.twice_area / 2
        for i, gi in zip(cell.tri, cell.grads):
            image[i] += area * diffusion * _inner(g, gi)
    work = _inner(r, direction)
    step = work / _inner(direction, image)
    # Boundary reactions are absent from the free-node residual.
    remainder = [Q(0) if i in boundary else ri - step * image[i] for i, ri in enumerate(r)]
    gain = step * work
    if not any(remainder):
        return gain
    lift = _tree_lift(cells, vertices, boundary, remainder)
    return gain + sum(c.twice_area * _inner(z, z) / (2 * diffusion) for c, z in zip(cells, lift))


def inspect_stopping(problem, trusted_digest, primal, rule, *,
                     fraction=Q(1, 2), relative_tolerance=Q(1, 10**6)):
    """Recompute a stop decision on a frozen state without a reference solution.

    Fixed mode tests ||r||_2 <= relative_tolerance ||b||_2 on free nodes.
    Classical mode requires the algebraic energy-error upper bound to be
    <= fraction times a lower bound on the primal discretization error.
    """
    if rule not in ("classical", "fixed"):
        raise ValueError("unknown stopping rule")
    parameter = rational(fraction if rule == "classical" else relative_tolerance)
    if not 0 < parameter <= 1:
        raise ValueError("stopping parameter must lie in (0, 1]")
    cells, boundary, _, values = _prepare(problem, trusted_digest, primal)
    if problem.original_affine is not None:
        raise ValueError("stopping comparison requires an original P0 load")
    b = [Q(0)] * len(values)
    r = [Q(0)] * len(values)
    diagonal = [Q(0)] * len(values)
    disc_lower = Q(0)
    kappa = problem.diffusion
    for cell, load in zip(cells, problem.loads, strict=True):
        area = cell.twice_area / 2
        gradient = _grad(cell, values)
        sum_grad_sq = Q(0)
        for i, gi in zip(cell.tri, cell.grads):
            if i not in boundary:
                b[i] += area * load / 3
                r[i] += area * (load / 3 - kappa * _inner(gradient, gi))
            if rule == "classical":
                gi_sq = _inner(gi, gi)
                diagonal[i] += kappa * area * gi_sq
                sum_grad_sq += gi_sq
        if rule == "classical":
            # The element bubble lambda_0 lambda_1 lambda_2 is energy
            # orthogonal to P1 and all other element bubbles. Its integral
            # is area/60 and its gradient norm squared is area*sum_grad_sq/180.
            disc_lower += area * load**2 / (20 * kappa * sum_grad_sq)
    residual_sq, rhs_sq = _inner(r, r), _inner(b, b)
    digest = digest_parts(["CFAC/stopping-P1/v1", trusted_digest, list(map(str, values))])
    if rule == "fixed":
        return StoppingObservation(rule, digest, residual_sq <= parameter**2 * rhs_sq,
                                   residual_sq, rhs_sq, None, None)
    alg_upper = _algebraic_bound(cells, problem.mesh.vertices, boundary, r, diagonal, kappa)
    return StoppingObservation(rule, digest, alg_upper <= parameter**2 * disc_lower,
                               residual_sq, rhs_sq, alg_upper, disc_lower)
