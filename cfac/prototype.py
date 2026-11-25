"""Small rational CPU solver for the CFAC P1/RT0 experiment.

The primal solver and the flux lifting use only frozen problem data. All
returned fields are tuples, so trials cannot modify an earlier state.
"""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from .spec import Mesh


def uniform_mesh(n):
    """Unit square with n by n squares and a rising diagonal in every square."""
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    vertices = tuple((Fraction(i, n), Fraction(j, n))
                     for j in range(n + 1) for i in range(n + 1))
    cells = []
    for j in range(n):
        for i in range(n):
            sw = j * (n + 1) + i
            se, nw, ne = sw + 1, sw + n + 1, sw + n + 2
            cells.extend(((sw, se, ne), (sw, ne, nw)))
    return Mesh(vertices, tuple(cells))


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _matvec(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _solve_spd(matrix, rhs):
    """Dense rational elimination for the small coarse and evaluation systems."""
    n = len(rhs)
    rows = [[Fraction(x) for x in row] + [Fraction(b)]
            for row, b in zip(matrix, rhs, strict=True)]
    for i in range(n):
        pivot = rows[i][i]
        if pivot <= 0:
            raise ValueError("expected a positive definite system")
        for k in range(i + 1, n):
            ratio = rows[k][i] / pivot
            if ratio:
                for j in range(i + 1, n + 1):
                    rows[k][j] -= ratio * rows[i][j]
    x = [Fraction(0)] * n
    for i in reversed(range(n)):
        x[i] = (rows[i][-1] - _dot(rows[i][i + 1:n], x[i + 1:])) / rows[i][i]
    return tuple(x)


@dataclass(frozen=True)
class _Cell:
    nodes: tuple
    area: Fraction
    gradients: tuple
    # Opposite-edge indices with the sign from global to local outward flux.
    faces: tuple


class ReferenceSystem:
    """Dense assembly for small meshes. The independent checker owns acceptance."""

    def __init__(self, problem):
        self.problem = problem
        mesh = problem.mesh
        self.vertices = tuple(tuple(map(Fraction, xy)) for xy in mesh.vertices)
        n = len(self.vertices)
        self.edges = tuple(sorted({tuple(sorted((t[i - 1], t[i])))
                                   for t in mesh.triangles for i in range(3)}))
        edge_ids = {edge: i for i, edge in enumerate(self.edges)}
        self.free_nodes = tuple(i for i, (x, y) in enumerate(self.vertices)
                                if 0 < x < 1 and 0 < y < 1)
        self.cells = []
        stiffness = [[Fraction(0)] * n for _ in range(n)]
        load = [Fraction(0)] * n
        diffusion = Fraction(problem.diffusion)
        for nodes, f in zip(mesh.triangles, problem.loads, strict=True):
            p = [self.vertices[i] for i in nodes]
            det = ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                   - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]))
            if det <= 0:
                raise ValueError("expected CCW triangles")
            area = det / 2
            gradients, faces = [], []
            for i in range(3):
                j, k = (i + 1) % 3, (i + 2) % 3
                gradients.append(((p[j][1] - p[k][1]) / det,
                                  (p[k][0] - p[j][0]) / det))
                a, b = nodes[j], nodes[k]
                faces.append((edge_ids[tuple(sorted((a, b)))], 1 if a < b else -1))
            self.cells.append(_Cell(nodes, area, tuple(gradients), tuple(faces)))
            for i, u in enumerate(nodes):
                load[u] += Fraction(f) * area / 3
                for j, v in enumerate(nodes):
                    stiffness[u][v] += area * _dot(gradients[i], gradients[j])
        self.load = tuple(load)
        self.primal_matrix = tuple(tuple(diffusion * x for x in row) for row in stiffness)
        self.stream_matrix = tuple(tuple(x / diffusion for x in row) for row in stiffness)
        self.feasible_flux = self._lift_load()

    def _lift_load(self):
        """Route exact cell balances to the exterior along a dual spanning tree."""
        outside = len(self.cells)
        owners = [[] for _ in self.edges]
        for k, cell in enumerate(self.cells):
            for edge, sign in cell.faces:
                owners[edge].append((k, sign))
        adjacency = [[] for _ in range(outside + 1)]
        for edge, incident in enumerate(owners):
            if len(incident) == 1:
                k, sign = incident[0]
                incident = [(k, sign), (outside, -sign)]
            (k, sk), (j, sj) = incident
            if sk != -sj:
                raise ValueError("incompatible adjacent cells")
            adjacency[k].append((j, edge, sj))
            adjacency[j].append((k, edge, sk))
        parent = {outside: None}
        order = []
        queue = deque([outside])
        while queue:
            k = queue.popleft()
            for j, edge, child_sign in adjacency[k]:
                if j not in parent:
                    parent[j] = (k, edge, child_sign)
                    order.append(j)
                    queue.append(j)
        if len(order) != outside:
            raise ValueError("cell adjacency does not reach the boundary")
        balance = [-c.area * f for c, f in zip(self.cells, self.problem.loads, strict=True)]
        balance.append(Fraction(0))
        flux = [Fraction(0)] * len(self.edges)
        for child in reversed(order):
            k, edge, sign = parent[child]
            flux[edge] = sign * balance[child]
            balance[k] += balance[child]
        return tuple(flux)

    def zero_primal(self):
        return (Fraction(0),) * len(self.vertices)

    def primal_step(self, primal):
        """One Gauss-Seidel sweep in the existing P1 space."""
        values = list(map(Fraction, primal))
        if len(values) != len(self.vertices) or any(
                values[i] for i, (x, y) in enumerate(self.vertices) if x in (0, 1) or y in (0, 1)):
            raise ValueError("expected a conforming homogeneous P1 field")
        for i in self.free_nodes:
            row = self.primal_matrix[i]
            values[i] += (self.load[i] - _dot(row, values)) / row[i]
        return tuple(values)

    def stream_correction(self, flux, stream):
        """Add curl(stream). Global edge moments are endpoint differences."""
        if len(stream) != len(self.vertices) or len(flux) != len(self.edges):
            raise ValueError("invalid stream or flux size")
        return tuple(Fraction(q) + Fraction(stream[j]) - Fraction(stream[i])
                     for q, (i, j) in zip(flux, self.edges, strict=True))

    def stream_rhs(self, flux):
        """Negative gradient of 1/2 ||q||^2_(a^-1) in stream coordinates.

        The primal cross term vanishes for every curl perturbation by
        integration by parts and the homogeneous Dirichlet trace.
        """
        if len(flux) != len(self.edges):
            raise ValueError("invalid flux size")
        rhs = list(self.zero_primal())
        for cell in self.cells:
            p = [self.vertices[i] for i in cell.nodes]
            center = tuple(sum(x) / 3 for x in zip(*p))
            # area times average of the affine RT0 field from opposite-edge moments.
            mass = tuple(sum(sign * Fraction(flux[e]) * (center[d] - p[k][d]) / 2
                             for k, (e, sign) in enumerate(cell.faces)) for d in range(2))
            for i, (gx, gy) in zip(cell.nodes, cell.gradients, strict=True):
                rhs[i] -= (mass[0] * gy - mass[1] * gx) / self.problem.diffusion
        return tuple(rhs)

    def local_dual_sweep(self, flux):
        """Coordinate descent in curl-of-hat directions, with vertex 0 pinned."""
        rhs = self.stream_rhs(flux)
        step = list(self.zero_primal())
        for i in range(1, len(step)):
            row = self.stream_matrix[i]
            step[i] = (rhs[i] - _dot(row, step)) / row[i]
        return self.stream_correction(flux, step)

    def coarse_dual_step(self, flux):
        """Galerkin correction on nodal interpolants of x, y and min(x,y).

        Together with constants these span the P1 space of the two triangles
        on the coarsest rising-diagonal square. Curls remove the constant.
        """
        rhs = self.stream_rhs(flux)
        modes = (tuple(x for x, y in self.vertices), tuple(y for x, y in self.vertices),
                 tuple(min(x, y) for x, y in self.vertices))
        images = [_matvec(self.stream_matrix, m) for m in modes]
        matrix = [[_dot(m, image) for image in images] for m in modes]
        weights = _solve_spd(matrix, [_dot(m, rhs) for m in modes])
        stream = tuple(_dot(weights, column) for column in zip(*modes, strict=True))
        return self.stream_correction(flux, stream)
