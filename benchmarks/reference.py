"""Independent scikit-fem oracle for the two published manufactured solutions."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from skfem import (Basis, ElementQuad1, ElementQuad2, ElementTriP1, ElementTriP2,
                   FacetBasis, LinearForm, MeshQuad, MeshTri, asm)
from skfem.models.poisson import laplace


def exact(mixed, x, y):
    if mixed:
        return np.cos(np.pi * x) * np.exp(y)
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def gradient(mixed, x, y):
    if mixed:
        return -np.pi * np.sin(np.pi * x) * np.exp(y), np.cos(np.pi * x) * np.exp(y)
    return (np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
            np.pi * np.sin(np.pi * x) * np.cos(np.pi * y))


@dataclass
class Reference:
    basis: Basis
    mixed: bool

    @classmethod
    def from_spec(cls, name, execution, n):
        # Bind the analytic problem using the published control identity.
        if name.startswith("poisson_sin_sin_"):
            mixed = False
        elif name.startswith("poisson_cos_exp_mixed_"):
            mixed = True
        else:
            raise ValueError(f"unknown oracle problem {name}")
        tri = execution["mesh_type"] == "tri"
        p = execution["polynomial_order"]
        knots = np.linspace(0, 1, n + 1)
        mesh = (MeshTri if tri else MeshQuad).init_tensor(knots, knots)
        element = {(True, 1): ElementTriP1, (True, 2): ElementTriP2,
                   (False, 1): ElementQuad1, (False, 2): ElementQuad2}[tri, p]()
        return cls(Basis(mesh, element, intorder=12), mixed)

    def solve(self, source_scale=1.0, stiffness_scale=1.0):
        b, mixed = self.basis, self.mixed

        @LinearForm
        def volume(v, w):
            factor = (np.pi ** 2 - 1.0) if mixed else 2.0 * np.pi ** 2
            return source_scale * factor * exact(mixed, *w.x) * v

        K = stiffness_scale * asm(laplace, b)
        F = asm(volume, b)
        x, y = b.doflocs
        fixed = (x == 0) | (x == 1) | (y == 0)
        if mixed:
            facets = b.mesh.facets_satisfying(lambda xy: xy[1] == 1)
            top = FacetBasis(b.mesh, b.elem, facets=facets, intorder=12)

            @LinearForm
            def flux(v, w):
                return np.cos(np.pi * w.x[0]) * np.exp(w.x[1]) * v

            F += asm(flux, top)
        else:
            fixed |= y == 1
        u = np.zeros(b.N)
        u[fixed] = exact(mixed, x[fixed], y[fixed])
        free = ~fixed
        u[free] = spsolve(K[free][:, free], F[free] - K[free][:, fixed] @ u[fixed])
        return u

    def errors(self, u, *, centroid=False):
        b = self.basis
        if centroid:
            tri = isinstance(b.mesh, MeshTri)
            xy = 1.0 / 3.0 if tri else 0.5
            b = Basis(b.mesh, b.elem, quadrature=(np.full((2, 1), xy),
                                                   np.array([0.5 if tri else 1.0])))
        x, y = b.global_coordinates()
        uh = b.interpolate(u)
        gx, gy = gradient(self.mixed, x, y)
        return (float(np.sqrt(np.sum((uh - exact(self.mixed, x, y)) ** 2 * b.dx))),
                float(np.sqrt(np.sum(((uh.grad[0] - gx) ** 2 + (uh.grad[1] - gy) ** 2) * b.dx))))

    def native_field(self, level):
        """Map by position. Every occurrence of every native DOF must agree."""
        nodes = np.loadtxt(level / "element_dofs.txt", ndmin=2)
        values = np.loadtxt(level / "dof_values.txt", ndmin=2)
        ids = nodes[:, 2].astype(int)
        distance, locations = cKDTree(self.basis.doflocs.T).query(nodes[:, 3:])
        order = np.full(self.basis.N, -1)
        if (values.shape != (self.basis.N, 2) or not np.array_equal(values[:, 0], np.arange(len(order)))
                or not np.array_equal(ids, nodes[:, 2]) or ids.min() < 0 or ids.max() >= len(order)):
            raise ValueError("native DOF ids differ from the canonical space")
        order[ids] = locations
        if (distance.max() > 1e-12 or not np.array_equal(order[ids], locations)
                or not np.array_equal(np.sort(order), np.arange(self.basis.N))):
            raise ValueError("native DOF coordinates do not match the independent grid")
        field = np.empty(self.basis.N)
        field[order] = values[:, 1]
        return field, order

    def check_topology(self, level, order, n):
        """Compare per-cell incidence and all labelled edges to the regenerated grid."""
        incidence = np.loadtxt(level / "element_dofs.txt", dtype=int, usecols=(2,))
        local_size = self.basis.element_dofs.shape[0]
        actual = order[incidence].reshape(-1, local_size)
        raw = [row.split()[3:] for row in (level / "elements.txt").read_text().splitlines()
               if row.strip() and not row.startswith("#")]
        vertices = np.asarray(raw, dtype=float).reshape(len(raw), -1, 2)
        dist, ids = cKDTree(self.basis.mesh.p.T).query(vertices)
        def labelled_cells(corners, dofs):
            return sorted((tuple(sorted(v)), tuple(sorted(d))) for v, d in zip(corners, dofs, strict=True))

        adjacent = np.roll(vertices, -1, axis=1)
        twice_area = np.sum(vertices[:, :, 0]*adjacent[:, :, 1] - vertices[:, :, 1]*adjacent[:, :, 0], axis=1)
        expected_area = (1 if vertices.shape[1] == 3 else 2) / n**2
        expected = labelled_cells(self.basis.mesh.t.T, self.basis.element_dofs.T)
        if (dist.max() > 1e-12 or not np.allclose(twice_area, expected_area, rtol=0, atol=1e-12)
                or labelled_cells(ids, actual) != expected):
            raise ValueError("cell vertices, orientation or local DOF incidence differ from the independent mesh")
        edges = np.loadtxt(level / "boundary.txt")
        expected = []
        for k in range(n):
            expected.extend([(1, (k, 0), (k + 1, 0)), (2, (n, k), (n, k + 1)),
                             (3, (k, n), (k + 1, n)), (4, (0, k), (0, k + 1))])
        xy = edges[:, 2:].reshape(-1, 2, 2)
        ij = np.rint(xy * n).astype(int)
        reported = [(attr, *sorted(map(tuple, ends))) for attr, ends in zip(edges[:, 1], ij)]
        if np.max(np.abs(xy - ij / n)) > 1e-12 or sorted(reported) != sorted(expected):
            raise ValueError("boundary edges differ from the independent unit square")


def relative_difference(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - reference)) / np.max(np.abs(reference)))
