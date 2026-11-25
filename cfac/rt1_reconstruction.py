"""Exact, small-mesh RT1 vertex-patch minimization for P1 Poisson fields.

Global face moments use the right normal of the increasing vertex order.
Each patch has zero normal trace on the edges opposite its central vertex.
Boundary spokes on the physical Dirichlet boundary have free normal trace.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction as F

from .enrichment import EnrichmentRequest, RTMoments
from .prototype import _solve_spd


@dataclass(frozen=True)
class _Triangle:
    vertices: tuple
    points: tuple
    area: F
    gradients: tuple
    faces: tuple


def _geometry(mesh):
    edge_list = sorted({tuple(sorted((t[j-1], t[j]))) for t in mesh.triangles for j in range(3)})
    edge_ids = {e: i for i, e in enumerate(edge_list)}
    cells, stars, owners = [], defaultdict(list), defaultdict(list)
    for k, nodes in enumerate(mesh.triangles):
        points = tuple(mesh.vertices[i] for i in nodes)
        (x0, y0), (x1, y1), (x2, y2) = points
        det = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)
        gradients, faces = [], []
        for i, v in enumerate(nodes):
            j, h = (i+1) % 3, (i+2) % 3
            gradients.append(((points[j][1]-points[h][1])/det, (points[h][0]-points[j][0])/det))
            start, end = nodes[j], nodes[h]
            edge = edge_ids[tuple(sorted((start, end)))]
            sign = 1 if start < end else -1
            faces.append((edge, sign))
            owners[edge].append((k, sign))
            stars[v].append((k, i))
        cells.append(_Triangle(nodes, points, det/2, tuple(gradients), tuple(faces)))
    return edge_list, cells, stars, owners


def _forest(adjacency, roots):
    """Return a child-before-parent order and the edge used to reach each child."""
    parents = dict.fromkeys(roots)
    queue = deque(roots)
    order = []
    while queue:
        i = queue.popleft()
        for j, edge in adjacency[i]:
            if j not in parents:
                parents[j] = (i, edge)
                order.append(j)
                queue.append(j)
    return reversed(order), parents


def _compatible_totals(request, edges, cells, owners):
    """Route P1 nodal residuals to the Dirichlet boundary in element-local pairs.

    Before correction the entry is int_K (a grad(v).grad(phi_i) - f phi_i).
    A transfer changes two entries of one cell by opposite amounts. Thus the
    cell sum stays -|K|f. Interior patch sums become zero exactly. The lift
    vanishes for a Galerkin iterate; its efficiency for inexact fields is open.
    """
    a = request.problem.diffusion
    totals, gradients = [], []
    residual = [F(0)] * len(request.primal)
    for k, cell in enumerate(cells):
        grad = tuple(sum(request.primal[v]*g[d] for v, g in zip(cell.vertices, cell.gradients))
                     for d in (0, 1))
        gradients.append(grad)
        local = [cell.area*(a*(grad[0]*gx+grad[1]*gy) - request.problem.loads[k]/3)
                 for gx, gy in cell.gradients]
        totals.append(local)
        for i, v in enumerate(cell.vertices):
            residual[v] -= local[i]
    adjacency = defaultdict(list)
    boundary = set()
    for edge, (i, j) in enumerate(edges):
        adjacency[i].append((j, edge))
        adjacency[j].append((i, edge))
        if len(owners[edge]) == 1:
            boundary.update((i, j))
    order, parents = _forest(adjacency, sorted(boundary))
    for v in order:
        parent, edge = parents[v]
        k = owners[edge][0][0]
        nodes = cells[k].vertices
        totals[k][nodes.index(v)] += residual[v]
        totals[k][nodes.index(parent)] -= residual[v]
        residual[parent] += residual[v]
    return totals, gradients


def _constant_lift(center, star, spokes, owners, totals):
    """Lift the integral divergence on a star through its radial edges."""
    adjacency = defaultdict(list)
    signs = {}
    boundary = False
    for edge in spokes:
        pair = owners[edge]
        left, s = pair[0]
        if len(pair) == 2:
            right, opposite = pair[1]
            signs[right, edge] = opposite
        else:
            right, boundary = -1, True
        signs[left, edge] = s
        adjacency[left].append((right, edge))
        adjacency[right].append((left, edge))
    root = -1 if boundary else star[0][0]
    remaining = {k: totals[k][i] for k, i in star}
    if boundary:
        remaining[root] = F(0)
    order, parents = _forest(adjacency, [root])
    if len(parents) != len(remaining):
        raise ValueError(f"disconnected star at vertex {center}")
    flux = {}
    for k in order:
        parent, edge = parents[k]
        flux[edge] = signs[k, edge] * remaining[k]
        remaining[parent] += remaining[k]
    if not boundary and remaining[root]:
        raise ArithmeticError("incompatible interior patch divergence")
    return flux


def _dot(p, q):
    return p[0]*q[0] + p[1]*q[1]


def _modes(cell, i, lam):
    # Curls of the central P1 hat and the P2 edge bubbles on its two spokes.
    gx, gy = cell.gradients[i]
    modes = [(gy, -gx)]
    for j in ((i+1) % 3, (i+2) % 3):
        dx, dy = cell.gradients[j]
        modes.append((4*(lam[i]*dy + lam[j]*gy), -4*(lam[i]*dx + lam[j]*gx)))
    return modes


def _base(cell, i, moments, load, lam):
    # The RT0 part supplies the mean divergence. The normal trace of
    # lambda_i (x-p_i) vanishes on every face, and div = 3 lambda_i - 1.
    x = tuple(sum(lam[j]*cell.points[j][d] for j in range(3)) for d in (0, 1))
    weights = [sign*moments.get(edge, 0)/(2*cell.area) for edge, sign in cell.faces]
    weights[i] -= load*lam[i]/3
    return tuple(sum(w*(x[d]-p[d]) for w, p in zip(weights, cell.points)) for d in (0, 1))


# Normalized degree-three Hammer rule. All assembly and solves are rational.
_RULE = [((F(1, 3),)*3, F(-27, 48))]
_RULE += [(tuple(F(3 if j == i else 1, 5) for j in range(3)), F(25, 48)) for i in range(3)]
_VERTICES = [tuple(F(i == j) for i in range(3)) for j in range(3)]
_MIDS = [tuple(F(0) if i == j else F(1, 2) for i in range(3)) for j in range(3)]


def _patch(center, star, cells, owners, totals, gradients, problem):
    spokes = sorted({e for k, i in star for j, (e, _) in enumerate(cells[k].faces) if j != i})
    moments = _constant_lift(center, star, spokes, owners, totals)
    # Increasing spoke index fixes all algebraic and integration order.
    number = {e: i+1 for i, e in enumerate(spokes)}
    n = len(spokes) + 1
    matrix = [[F(0)]*n for _ in range(n)]
    rhs = [F(0)]*n
    indices = {}
    for k, i in star:
        cell = cells[k]
        j, h = (i+1) % 3, (i+2) % 3
        ids = (0, number[cell.faces[h][0]], number[cell.faces[j][0]])
        indices[k] = ids
        for lam, weight in _RULE:
            q = _base(cell, i, moments, problem.loads[k], lam)
            residual = tuple(lam[i]*problem.diffusion*g-y for g, y in zip(gradients[k], q))
            modes = _modes(cell, i, lam)
            # A constant 1/a scales the whole objective and cancels in its minimizer.
            volume = weight * cell.area
            for idx, v in zip(ids, modes):
                rhs[idx] += volume*_dot(residual, v)
                for jdx, w in zip(ids, modes):
                    matrix[idx][jdx] += volume*_dot(v, w)
    psi = _solve_spd(matrix, rhs)
    for k, i in star:
        cell = cells[k]
        coeff = [psi[j] for j in indices[k]]

        def sample(lam):
            base = _base(cell, i, moments, problem.loads[k], lam)
            modes = _modes(cell, i, lam)
            return tuple(base[d] + sum(c*w[d] for c, w in zip(coeff, modes)) for d in (0, 1))

        yield k, [sample(lam) for lam in _VERTICES], [sample(lam) for lam in _MIDS]


class RT1Reconstructor:
    """Reference patch reconstruction for exact P0 data and their frozen projections."""

    def reconstruct(self, request: EnrichmentRequest) -> RTMoments:
        if type(request) is not EnrichmentRequest or request.order != 1:
            raise ValueError("expected a frozen RT1 enrichment request")
        problem = request.problem
        edges, cells, stars, owners = _geometry(problem.mesh)
        totals, gradients = _compatible_totals(request, edges, cells, owners)
        # Assemble cell endpoint values before comparing both traces on shared faces.
        nodal = [[[F(0), F(0)] for _ in range(3)] for _ in cells]
        integrals = [[F(0), F(0)] for _ in cells]
        for v, star in stars.items():
            for k, values, midpoints in _patch(v, star, cells, owners, totals, gradients, problem):
                for d in (0, 1):
                    integrals[k][d] += cells[k].area*sum(p[d] for p in midpoints)/3
                    for i in range(3):
                        nodal[k][i][d] += values[i][d]
        faces = []
        for edge, (i, j) in enumerate(edges):
            p, q = problem.mesh.vertices[i], problem.mesh.vertices[j]
            scaled_normal = (q[1]-p[1], p[0]-q[0])
            pair = []
            for k, _ in owners[edge]:
                u, v = (_dot(scaled_normal, nodal[k][cells[k].vertices.index(node)]) for node in (i, j))
                pair.append(((u+v)/2, (u+2*v)/6))
            if len(pair) == 2 and pair[0] != pair[1]:
                raise ArithmeticError("patch sum has a discontinuous normal trace")
            faces.append(pair[0])
        return RTMoments(request.problem_digest, request.parent_state_digest, 1, faces, integrals)
