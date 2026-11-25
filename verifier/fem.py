"""Independent finite-element machinery for verification (numpy/scipy only).

Nothing here is shared with, or derived from, the MFEM solver code:

* geometry maps are built from exported element vertex coordinates;
* the local Lagrange basis is constructed from the *exported physical node
  coordinates* (inverse-mapped to the reference element) via a Vandermonde
  matrix on the monomial basis P_p (triangles) or Q_p (quadrilaterals), so the
  verifier does not need to know MFEM's internal node layout or numbering;
* quadrature: Gauss-Legendre (tensor) on quads, collapsed (Duffy) Gauss on
  triangles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ArtifactError(ValueError):
    """Raised when solver artifacts are missing, malformed or inconsistent."""


@dataclass
class MeshData:
    geom: str                 # "tri" or "quad"
    verts: np.ndarray         # (E, nv, 2)
    elem_dofs: np.ndarray     # (E, nloc) global dof ids
    elem_node_xy: np.ndarray  # (E, nloc, 2) physical node coordinates
    values: np.ndarray        # (N,) nodal dof values
    bdr_attr: np.ndarray      # (B,) boundary attribute per segment
    bdr_xy: np.ndarray        # (B, 2, 2) segment endpoints


def boundary_edge_owners(mesh: MeshData, attributes):
    """Locate the requested exported segments using the verifier's edge keys."""
    verts = np.round(mesh.verts, 12)
    candidates = np.zeros(len(verts), dtype=bool)
    keys = set()
    for attr in attributes:
        segments = np.round(mesh.bdr_xy[mesh.bdr_attr == attr], 12)
        if len(segments) == 0:
            continue
        keys.update(frozenset(map(tuple, segment)) for segment in segments)
        # A matching edge must have an endpoint inside this group's bounds.
        # Use the actual segments, including displaced or mixed-label inputs.
        lo, hi = segments.min(axis=(0, 1)), segments.max(axis=(0, 1))
        candidates |= np.any(np.all((verts >= lo) & (verts <= hi), axis=2), axis=1)
    owners = {}
    for e in np.flatnonzero(candidates):
        for k, a in enumerate(verts[e]):
            key = frozenset((tuple(a), tuple(verts[e, (k + 1) % verts.shape[1]])))
            if key in keys:
                owners[key] = e
    return owners


def lattice_indices(xy, subdivisions, tol):
    """Return integer coordinates on the closed unit-square grid."""
    if np.any(xy < -tol) or np.any(xy > 1 + tol):
        return None
    nearest = np.rint(xy * subdivisions)
    if np.any(np.abs(xy - nearest / subdivisions) > tol):
        return None
    return nearest.astype(np.int64)


def cartesian_partition(mesh, n, tol):
    """Check every Cartesian cell once, allowing either triangle diagonal."""
    tri = mesh.geom == "tri"
    if len(mesh.verts) != n * n * (2 if tri else 1):
        return False
    grid = lattice_indices(mesh.verts, n, tol)
    if grid is None:
        return False
    lower = grid.min(axis=1)
    if np.any(grid.max(axis=1) - lower != 1):
        return False
    dx, dy = (grid - lower[:, None, :]).transpose(2, 0, 1)
    corners = dx + 2 * dy
    if np.any(np.diff(np.sort(corners, axis=1), axis=1) == 0):
        return False
    cells = lower[:, 0] + n * lower[:, 1]
    if np.any(np.bincount(cells, minlength=n*n) != (2 if tri else 1)):
        return False
    if tri:
        # Each triangle omits one of 0,1,2,3. Opposite missing corners
        # (0+3 or 1+2) give two complementary halves of the same square.
        missing = 6 - corners.sum(axis=1)
        return bool(np.all(np.bincount(cells, weights=missing) == 3))
    edges = grid - np.roll(grid, 1, axis=1)
    return bool(np.all(np.abs(edges).sum(axis=2) == 1))


def _loadtxt(path: Path, ncols: int) -> np.ndarray:
    if not path.is_file():
        raise ArtifactError(f"missing artifact {path.name}")
    try:
        arr = np.loadtxt(path, comments="#", ndmin=2)
    except (ValueError, OSError) as exc:
        raise ArtifactError(f"malformed artifact {path.name}: {exc}") from exc
    if arr.shape[0] == 0 or arr.shape[1] != ncols:
        raise ArtifactError(f"artifact {path.name} has unexpected shape {arr.shape}")
    return arr


def load_mesh_data(level_dir: Path) -> MeshData:
    level_dir = Path(level_dir)
    # elements.txt: elem geom nverts coords...
    lines = [ln.split() for ln in (level_dir / "elements.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")] if (level_dir / "elements.txt").is_file() else None
    if not lines:
        raise ArtifactError("missing or empty artifact elements.txt")
    if any(len(ln) < 3 for ln in lines):
        raise ArtifactError("truncated row in elements.txt")
    geoms = {ln[1] for ln in lines}
    if len(geoms) != 1 or next(iter(geoms)) not in ("tri", "quad"):
        raise ArtifactError(f"unsupported/mixed element geometries {geoms}")
    geom = geoms.pop()
    nv = 3 if geom == "tri" else 4
    try:
        if any(int(ln[2]) != nv or len(ln) != 3 + 2 * nv for ln in lines):
            raise ArtifactError("wrong vertex count in elements.txt")
        verts = np.array([[float(v) for v in ln[3:]] for ln in lines]).reshape(-1, nv, 2)
        eids = np.array([int(ln[0]) for ln in lines])
    except (ValueError, IndexError) as exc:
        raise ArtifactError(f"malformed elements.txt: {exc}") from exc
    if not np.array_equal(eids, np.arange(len(eids))):
        raise ArtifactError("elements.txt element ids are not 0..E-1 in order")

    ed = _loadtxt(level_dir / "element_dofs.txt", 5)
    E = verts.shape[0]
    if ed.shape[0] % E != 0:
        raise ArtifactError("element_dofs.txt row count is not a multiple of the element count")
    nloc = ed.shape[0] // E
    ed = ed.reshape(E, nloc, 5)
    if not (np.array_equal(ed[:, :, 0], np.repeat(np.arange(E)[:, None], nloc, 1))
            and np.array_equal(ed[:, :, 1], np.repeat(np.arange(nloc)[None, :], E, 0))):
        raise ArtifactError("element_dofs.txt is not ordered by (element, local index)")
    elem_node_xy = ed[:, :, 3:5].copy()

    dv = _loadtxt(level_dir / "dof_values.txt", 2)
    N = dv.shape[0]
    if not np.array_equal(dv[:, 0], np.arange(N)):
        raise ArtifactError("dof_values.txt ids are not 0..N-1 in order")
    values = dv[:, 1].copy()
    ids = ed[:, :, 2]
    if not np.all((ids == np.floor(ids)) & (ids >= 0) & (ids < N)):
        raise ArtifactError("element dof ids must be integers in range of dof_values.txt")
    elem_dofs = ids.astype(np.int64)

    bd = _loadtxt(level_dir / "boundary.txt", 6)
    if not np.array_equal(bd[:, 0], np.arange(len(bd))):
        raise ArtifactError("boundary.txt ids are not 0..B-1 in order")
    attrs = bd[:, 1]
    if not np.all((attrs == np.floor(attrs)) & (attrs > 0) & (attrs <= np.iinfo(np.int32).max)):
        raise ArtifactError("invalid boundary attribute id")
    if not all(np.isfinite(a).all() for a in (verts, elem_node_xy, bd[:, 2:])):
        raise ArtifactError("non-finite mesh coordinates")
    return MeshData(geom=geom, verts=verts, elem_dofs=elem_dofs, elem_node_xy=elem_node_xy,
                    values=values, bdr_attr=bd[:, 1].astype(np.int64),
                    bdr_xy=bd[:, 2:6].reshape(-1, 2, 2))


# --------------------------------------------------------------------------
# Reference element: monomials, quadrature, geometry maps
# --------------------------------------------------------------------------

def monomial_exponents(geom: str, p: int) -> np.ndarray:
    if geom == "tri":
        exps = [(i, j) for j in range(p + 1) for i in range(p + 1 - j)]
    else:
        exps = [(i, j) for j in range(p + 1) for i in range(p + 1)]
    return np.array(exps, dtype=np.int64)


def eval_monomials(exps: np.ndarray, xi: np.ndarray, eta: np.ndarray):
    """Return M, dM/dxi, dM/deta with shape xi.shape + (nmono,)."""
    a = exps[:, 0]
    b = exps[:, 1]
    X = xi[..., None]
    Y = eta[..., None]
    M = X ** a * Y ** b
    dX = np.where(a > 0, a * X ** np.maximum(a - 1, 0), 0.0) * Y ** b
    dY = X ** a * np.where(b > 0, b * Y ** np.maximum(b - 1, 0), 0.0)
    return M, dX, dY


def gauss_01(m: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(m)
    return 0.5 * (x + 1.0), 0.5 * w


def reference_quadrature(geom: str, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points (xi, eta) and weights on the reference element (area 1/2 or 1)."""
    g, w = gauss_01(m)
    U, V = np.meshgrid(g, g, indexing="ij")
    WU, WV = np.meshgrid(w, w, indexing="ij")
    if geom == "quad":
        return U.ravel(), V.ravel(), (WU * WV).ravel()
    # Duffy collapse of the unit square onto the triangle {xi,eta>=0, xi+eta<=1}
    xi = U
    eta = V * (1.0 - U)
    wt = WU * WV * (1.0 - U)
    return xi.ravel(), eta.ravel(), wt.ravel()


def geometry_map(geom: str, verts: np.ndarray, xi: np.ndarray, eta: np.ndarray):
    """Physical points and Jacobians for each element at reference points.

    verts: (E, nv, 2); xi, eta: (Q,). Returns X (E,Q,2), J (E,Q,2,2) with
    J[..., i, j] = d x_i / d xi_j.
    """
    if geom == "tri":
        v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]
        A = np.stack([v1 - v0, v2 - v0], axis=-1)  # (E,2,2) columns dX/dxi, dX/deta
        X = v0[:, None, :] + xi[None, :, None] * A[:, None, :, 0] + eta[None, :, None] * A[:, None, :, 1]
        J = np.broadcast_to(A[:, None, :, :], (verts.shape[0], xi.size, 2, 2))
        return X, J
    v0, v1, v2, v3 = (verts[:, k][:, None, :] for k in range(4))
    s = xi[None, :, None]
    t = eta[None, :, None]
    X = v0 * (1 - s) * (1 - t) + v1 * s * (1 - t) + v2 * s * t + v3 * (1 - s) * t
    dXds = -v0 * (1 - t) + v1 * (1 - t) + v2 * t - v3 * t
    dXdt = -v0 * (1 - s) - v1 * s + v2 * s + v3 * (1 - s)
    J = np.stack([dXds, dXdt], axis=-1)
    return X, J


def inverse_map(geom: str, verts: np.ndarray, X: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Reference coordinates of physical points X (E,K,2) in element e."""
    if geom == "tri":
        v0 = verts[:, 0]
        A = np.stack([verts[:, 1] - v0, verts[:, 2] - v0], axis=-1)
        rhs = X - v0[:, None, :]
        return np.linalg.solve(A[:, None, :, :], rhs[..., None])[..., 0]
    ref = np.full(X.shape, 0.5)
    for _ in range(30):
        xi = ref[..., 0]
        eta = ref[..., 1]
        v = [verts[:, k][:, None, :] for k in range(4)]
        s = xi[..., None]
        t = eta[..., None]
        F = v[0] * (1 - s) * (1 - t) + v[1] * s * (1 - t) + v[2] * s * t + v[3] * (1 - s) * t - X
        dXds = -v[0] * (1 - t) + v[1] * (1 - t) + v[2] * t - v[3] * t
        dXdt = -v[0] * (1 - s) - v[1] * s + v[2] * s + v[3] * (1 - s)
        J = np.stack([dXds, dXdt], axis=-1)
        step = np.linalg.solve(J, F[..., None])[..., 0]
        ref = ref - step
        if np.max(np.abs(step)) < tol:
            break
    return ref


@dataclass
class LocalBasis:
    exps: np.ndarray       # (nloc, 2)
    coeffs: np.ndarray     # (E, nloc, nloc): phi_k = sum_m mono_m * coeffs[e, m, k]
    max_cond: float
    node_ref: np.ndarray   # (E, nloc, 2)
    translated: bool = False


def build_local_basis(mesh: MeshData, p: int) -> LocalBasis:
    exps = monomial_exponents(mesh.geom, p)
    nloc = exps.shape[0]
    if mesh.elem_dofs.shape[1] != nloc:
        raise ArtifactError(
            f"element has {mesh.elem_dofs.shape[1]} local dofs but a {mesh.geom} of order {p} "
            f"needs {nloc}")
    origin = mesh.verts[:, :1]
    def same_offsets(points):
        return np.all(points - origin == points[:1] - origin[:1])

    # Prove congruence including local node ordering from the raw tables.
    translated = bool(same_offsets(mesh.verts) and same_offsets(mesh.elem_node_xy))
    verts = mesh.verts[:1] if translated else mesh.verts
    nodes = mesh.elem_node_xy[:1] if translated else mesh.elem_node_xy
    node_ref = inverse_map(mesh.geom, verts, nodes)
    if np.any(node_ref < -1e-9) or np.any(node_ref > 1 + 1e-9) or (
            mesh.geom == "tri" and np.any(node_ref.sum(-1) > 1 + 1e-9)):
        raise ArtifactError("exported dof node lies outside its element")
    # Share a basis only when all exported nodes have exactly the same
    # reference coordinates in the same local order, including signed zero.
    shared = np.all(node_ref.view(np.uint64) == node_ref[:1].view(np.uint64))
    nodes = node_ref[:1] if shared else node_ref
    V, _, _ = eval_monomials(exps, nodes[..., 0], nodes[..., 1])
    cond = np.linalg.cond(V)
    max_cond = float(np.max(cond))
    if not np.isfinite(max_cond) or max_cond > 1e10:
        raise ArtifactError(f"local node set is not unisolvent (max cond {max_cond:.3e})")
    coeffs = np.linalg.inv(V)  # V[e,i,m] C[e,m,k] = delta_ik
    if shared:
        coeffs = np.broadcast_to(coeffs, (len(origin), nloc, nloc))
    if translated:
        node_ref = np.broadcast_to(node_ref, mesh.elem_node_xy.shape)
    return LocalBasis(exps, coeffs, max_cond, node_ref, translated)


@dataclass
class QuadData:
    X: np.ndarray        # (E,Q,2)
    wdet: np.ndarray     # (E,Q) weight * |det J|
    detJ: np.ndarray     # (E,Q)
    phi: np.ndarray      # (E,Q,nloc)
    gphi: np.ndarray     # (E,Q,nloc,2) physical gradients


def element_quadrature(mesh: MeshData, basis: LocalBasis, m: int) -> QuadData:
    xi, eta, w = reference_quadrature(mesh.geom, m)
    verts = mesh.verts[:1] - mesh.verts[:1, :1] if basis.translated else mesh.verts
    X, J = geometry_map(mesh.geom, verts, xi, eta)
    # Each straight triangle has one affine Jacobian. Quadrilateral maps
    # can vary across quadrature points even when the basis is shared.
    affine = mesh.geom == "tri"
    jac = J[:, :1] if affine else J
    det = jac[..., 0, 0] * jac[..., 1, 1] - jac[..., 0, 1] * jac[..., 1, 0]
    E, Q = X.shape[:2]
    detJ = np.broadcast_to(det, (E, Q))
    M, dMx, dMy = eval_monomials(basis.exps, xi, eta)  # (Q, nmono)
    coeffs = basis.coeffs[:1] if basis.coeffs.strides[0] == 0 else basis.coeffs
    nloc = coeffs.shape[-1]
    phi = np.broadcast_to(np.einsum("qm,emk->eqk", M, coeffs), (E, Q, nloc))
    if affine and nloc == 3:  # Gradients of linear triangle shapes are constant.
        dMx, dMy = dMx[:1], dMy[:1]
    dphi_ref = np.stack([np.einsum("qm,emk->eqk", dMx, coeffs),
                         np.einsum("qm,emk->eqk", dMy, coeffs)], axis=-1)
    gphi = np.einsum("eqkj,eqji->eqki", dphi_ref, np.linalg.inv(jac))
    gphi = np.broadcast_to(gphi, (E, Q, nloc, 2))
    wdet = w[None, :] * np.abs(detJ)
    if basis.translated:
        X = X + mesh.verts[:, :1]
        wdet, detJ, phi, gphi = (np.broadcast_to(a, (len(X), *a.shape[1:]))
                                 for a in (wdet, detJ, phi, gphi))
    return QuadData(X=X, wdet=wdet, detJ=detJ, phi=phi, gphi=gphi)
