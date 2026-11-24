"""Closed-form field, flux traces and energy for the aligned 1:10000 Q1 benchmark."""

import numpy as np

from .fem import ArtifactError, lattice_indices

CONTRAST = 10000.0
FLUX = 1.0 / (0.5 + 0.5 / CONTRAST)


def coefficient(x):
    return np.where(x < 0.5, 1.0, CONTRAST)


def field(x):
    return FLUX * (np.minimum(x, 0.5) + np.maximum(x - 0.5, 0.0) / CONTRAST)


def diagnostics(mesh, n):
    """Reconstruct each bilinear field from physical corners, directly in Cartesian axes."""
    grid = lattice_indices(mesh.elem_node_xy, n, 1e-12)
    vertices = lattice_indices(mesh.verts, n, 1e-12)
    if mesh.geom != "quad" or grid is None or vertices is None or mesh.elem_dofs.shape != (n*n, 4):
        raise ArtifactError("layered flux requires aligned Cartesian Q1 nodes")
    lower = vertices.min(axis=1)
    offsets = grid - lower[:, None]
    corners = offsets[..., 0] + 2 * offsets[..., 1]
    if (n % 2 or np.any((offsets < 0) | (offsets > 1))
            or not np.all(np.sort(corners, axis=1) == np.arange(4))):
        raise ArtifactError("each aligned Q1 cell needs four distinct physical corners")
    U = mesh.values[mesh.elem_dofs]
    u00, u10, u01, u11 = np.take_along_axis(U, np.argsort(corners, axis=1), axis=1).T
    dx = n * np.column_stack((u10 - u00, u11 - u01))  # bottom/top traces
    dy = n * np.column_stack((u01 - u00, u11 - u10))  # left/right traces
    a = coefficient((lower[:, 0] + 0.5) / n)
    flux_x, flux_y = a[:, None] * dx, a[:, None] * dy

    def interface_trace(column):
        cells = np.flatnonzero(lower[:, 0] == column)
        cells = cells[np.argsort(lower[cells, 1])]
        if not np.array_equal(lower[cells, 1], np.arange(n)):
            raise ArtifactError("each interface segment requires exactly one cell on each side")
        return flux_x[cells]

    jump = interface_trace(n // 2) - interface_trace(n // 2 - 1)
    # Integral of the square of a linear function with endpoint values s,t on [0,1].
    def mean_square(v):
        s, t = v.T
        return (s*s + s*t + t*t) / 3.0

    energy = float(np.sum(a * (mean_square(dx) + mean_square(dy))) / n**2)
    return {"nodal_abs": float(np.max(np.abs(U - field(mesh.elem_node_xy[..., 0])))),
            "flux_rel": float(max(np.max(np.abs(flux_x - FLUX)), np.max(np.abs(flux_y))) / FLUX),
            "jump_rel": float(np.max(np.abs(jump)) / FLUX),
            "energy": energy, "energy_rel": abs(energy - FLUX) / FLUX}
