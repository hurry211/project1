"""Preflight the MSH 2.2 subset covered by the Cartesian verifier."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .spec import ExecutionSpec, ExternalMeshSpec, ProblemSpec

ARCHIVED_MESH = "frozen/mesh.msh"
MAX_INPUT_BYTES = 8 * 1024 * 1024
_INTEGER = re.compile(r"[0-9]+")
_FLOAT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


@dataclass(frozen=True)
class CartesianGmsh:
    subdivisions: int
    # One flag per coarse cell in row-major order. True denotes SW--NE.
    rising_diagonal: tuple[bool, ...]

    def refinements(self, n: int) -> int:
        ratio, rem = divmod(n, self.subdivisions)
        if rem or ratio < 1 or ratio & (ratio - 1):
            raise ValueError(f"level {n} is not a uniform refinement of the input grid ({self.subdivisions})")
        return ratio.bit_length() - 1


@dataclass(frozen=True)
class UnstructuredGmsh:
    geom: str
    node_count: int
    element_count: int
    total_area: float
    boundary_lengths: dict[str, float]
    subdivisions_hint: int
    coarse_cells: tuple[tuple[tuple[float, float], ...], ...]

    def refinements(self, n: int) -> int:
        ratio, rem = divmod(n, self.subdivisions_hint)
        if rem or ratio < 1 or ratio & (ratio - 1):
            raise ValueError(f"level {n} is not a uniform refinement of the base level {self.subdivisions_hint}")
        return ratio.bit_length() - 1


def read_mesh(path: Path, expected: ExternalMeshSpec) -> bytes:
    if not path.is_file():
        raise ValueError(f"missing regular mesh file {path.name}")
    with path.open("rb") as stream:
        data = stream.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("external mesh exceeds the 8 MiB input limit")
    if hashlib.sha256(data).hexdigest() != expected.sha256:
        raise ValueError("external mesh SHA-256 differs from the request")
    return data


def _integer(token: str) -> int:
    if not _INTEGER.fullmatch(token) or len(token) > 10 or int(token) > 2147483647:
        raise ValueError("expected a nonnegative 32-bit integer")
    return int(token)


def _sections(data: bytes) -> dict[str, list[str]]:
    content = data.decode("ascii").replace("\r\n", "\n")
    if re.search(r"[^\t\n\x20-\x7e]", content):
        raise ValueError("unsupported control character in ASCII Gmsh input")
    lines = iter(content.split("\n"))
    sections = {}
    try:
        for name in ("MeshFormat", "PhysicalNames", "Nodes", "Elements"):
            if next(lines) != "$" + name:
                raise ValueError(f"expected the Gmsh {name} section")
            if name == "MeshFormat":
                header = next(lines)
                if header.split() != ["2.2", "0", "8"] or header != header.rstrip():
                    raise ValueError("expected Gmsh MSH 2.2 ASCII with 8-byte coordinates")
            else:
                # MFEM reads the count followed immediately by LF or CRLF.
                count = _integer(next(lines).lstrip())
                # Pull rows incrementally so a forged count cannot reserve memory.
                rows = []
                for _ in range(count):
                    rows.append(next(lines))
                sections[name] = rows
            if next(lines) != "$End" + name:
                raise ValueError(f"wrong row count or missing end marker for {name}")
    except StopIteration as exc:
        raise ValueError("truncated Gmsh mesh") from exc
    if any(line.strip() for line in lines):
        raise ValueError("unsupported extra Gmsh sections")
    return sections


def _groups(rows: list[str], problem: ProblemSpec) -> tuple[dict[int, str], int]:
    named = {}
    for row in rows:
        match = re.fullmatch(r'\s*([12])\s+([0-9]+)\s+"([^"]+)"\s*', row)
        if not match:
            raise ValueError("malformed physical group name")
        dim, tag, name = match.groups()
        if not name.strip():
            raise ValueError("empty physical group name")
        key = (int(dim), _integer(tag))
        if not 1 <= key[1] <= 65535 or key in named:
            raise ValueError("duplicate or out-of-range physical tag (allowed 1..65535)")
        named[key] = name
    boundaries = {tag: name for (dim, tag), name in named.items() if dim == 1}
    if boundaries != {b.attribute: b.label for b in problem.boundaries}:
        raise ValueError("Gmsh physical boundary names and tags differ from ProblemSpec")
    surfaces = [tag for (dim, tag) in named if dim == 2]
    if len(surfaces) != 1:
        raise ValueError("one named physical surface is required")
    return boundaries, surfaces[0]


def _nodes(rows: list[str], n: int) -> dict[int, tuple[int, int]]:
    nodes = {}
    for row in rows:
        fields = row.split()
        if len(fields) != 4 or not all(_FLOAT.fullmatch(v) for v in fields[1:]):
            raise ValueError("invalid Gmsh node coordinates")
        idx = _integer(fields[0])
        x, y, z = map(float, fields[1:])
        if idx == 0 or idx in nodes or z != 0 or not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("invalid node id or coordinates outside the planar unit square")
        i, j = round(x * n), round(y * n)
        if max(abs(x - i / n), abs(y - j / n)) > 1e-12:
            raise ValueError("node outside the regular input lattice")
        nodes[idx] = (i, j)
    if len(set(nodes.values())) != len(nodes):
        raise ValueError("multiple node ids occupy one Cartesian vertex")
    return nodes


def _boundary_edges(n: int) -> dict:
    edges = {}
    for i in range(n):
        edges[((i, 0), (i + 1, 0))] = "bottom"
        edges[((i, n), (i + 1, n))] = "top"
        edges[((0, i), (0, i + 1))] = "left"
        edges[((n, i), (n, i + 1))] = "right"
    return edges


def _cell(vertices: list[tuple[int, int]], n: int) -> tuple[int, int]:
    x, y = zip(*vertices)
    i, j = min(x), min(y)
    if max(x) - i != 1 or max(y) - j != 1 or len(set(vertices)) != len(vertices):
        raise ValueError("element must occupy one Cartesian cell with distinct corners")
    double_area = sum(x[k - 1] * y[k] - x[k] * y[k - 1] for k in range(len(x)))
    if double_area != len(vertices) - 2:
        raise ValueError("inverted, overlapping or noncyclic Cartesian element")
    mask = sum(1 << ((vx - i) + 2 * (vy - j)) for vx, vy in vertices)
    return i + n * j, mask


def inspect_gmsh(data: bytes, problem: ProblemSpec, execution: ExecutionSpec) -> CartesianGmsh:
    sections = _sections(data)
    boundaries, surface = _groups(sections["PhysicalNames"], problem)
    count = len(sections["Nodes"])
    n = math.isqrt(count) - 1
    if n < 1 or (n + 1) ** 2 != count:
        raise ValueError("node count does not describe a square Cartesian lattice")
    nodes = _nodes(sections["Nodes"], n)
    edges = _boundary_edges(n)
    expected_type = 2 if execution.mesh_type == "tri" else 3
    cells = [[] for _ in range(n*n)]
    element_ids = set()
    for row in sections["Elements"]:
        fields = [_integer(token) for token in row.split()]
        if len(fields) < 5:
            raise ValueError("truncated Gmsh element")
        idx, kind, tag_count, physical, entity = fields[:5]
        if idx == 0 or idx in element_ids or tag_count != 2 or entity == 0:
            raise ValueError("expected unique element id, physical tag and elementary tag")
        element_ids.add(idx)
        if kind not in (1, expected_type) or len(fields) != 5 + kind + 1:
            raise ValueError("expected linear lines and homogeneous linear " + execution.mesh_type + " elements")
        try:
            vertices = [nodes[node] for node in fields[5:]]
        except KeyError as exc:
            raise ValueError("unknown element node id") from exc
        if kind == 1:
            label = edges.pop(tuple(sorted(vertices)), None)
            if label is None or boundaries.get(physical) != label:
                raise ValueError("duplicate, interior or incorrectly tagged boundary edge")
        else:
            if physical != surface:
                raise ValueError("element is outside the named physical surface")
            cell, mask = _cell(vertices, n)
            cells[cell].append(mask)
    if edges:
        raise ValueError("missing exterior boundary edges")
    diagonals = []
    for masks in cells:
        if expected_type == 3 and masks == [15]:
            continue
        if expected_type == 2 and sorted(masks) in ([11, 13], [7, 14]):
            diagonals.append(11 in masks)
            continue
        raise ValueError("each Cartesian cell must contain one quad or two complementary triangles")
    grid = CartesianGmsh(n, tuple(diagonals))
    for level in execution.mesh_levels:
        grid.refinements(level)
    return grid


_SIDE_EPS = 1e-12


def _point_side(xy: tuple[float, float]) -> list[str]:
    x, y = xy
    sides = []
    if abs(y) <= _SIDE_EPS:
        sides.append("bottom")
    if abs(x - 1.0) <= _SIDE_EPS:
        sides.append("right")
    if abs(y - 1.0) <= _SIDE_EPS:
        sides.append("top")
    if abs(x) <= _SIDE_EPS:
        sides.append("left")
    return sides


def inspect_gmsh_unstructured(data: bytes, problem: ProblemSpec, execution: ExecutionSpec) -> UnstructuredGmsh:
    """Validate an arbitrary conforming unstructured 2D triangle/quad Gmsh MSH 2.2 mesh on [0,1]^2."""
    sections = _sections(data)
    boundaries, surface = _groups(sections["PhysicalNames"], problem)

    nodes: dict[int, tuple[float, float]] = {}
    rounded_coords: set[tuple[float, float]] = set()
    for row in sections["Nodes"]:
        fields = row.split()
        if len(fields) != 4 or not all(_FLOAT.fullmatch(v) for v in fields[1:]):
            raise ValueError("invalid Gmsh node coordinates")
        idx = _integer(fields[0])
        x, y, z = map(float, fields[1:])
        if (
            idx == 0 or idx in nodes or not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z))
            or abs(z) > _SIDE_EPS or not (-_SIDE_EPS <= x <= 1.0 + _SIDE_EPS and -_SIDE_EPS <= y <= 1.0 + _SIDE_EPS)
        ):
            raise ValueError("invalid node id or coordinates outside planar unit square")
        rkey = (round(x, 12), round(y, 12))
        if rkey in rounded_coords:
            raise ValueError("duplicate node coordinates in unstructured Gmsh mesh")
        rounded_coords.add(rkey)
        nodes[idx] = (x, y)

    expected_type = 2 if execution.mesh_type == "tri" else 3
    bdr_edges: dict[tuple[int, int], str] = {}
    side_lengths: dict[str, float] = {"bottom": 0.0, "right": 0.0, "top": 0.0, "left": 0.0}
    side_intervals: dict[str, list[tuple[float, float]]] = {s: [] for s in side_lengths}
    elem_edge_counts: dict[tuple[int, int], int] = {}
    element_ids: set[int] = set()
    coarse_cells = []
    total_area = 0.0

    for row in sections["Elements"]:
        fields = [_integer(token) for token in row.split()]
        if len(fields) < 5:
            raise ValueError("truncated Gmsh element")
        idx, kind, tag_count, physical, entity = fields[:5]
        if idx == 0 or idx in element_ids or tag_count != 2 or entity == 0:
            raise ValueError("expected unique element id, physical tag and elementary tag")
        element_ids.add(idx)
        nverts_expected = 2 if kind == 1 else (3 if kind == 2 else 4)
        if kind not in (1, expected_type) or len(fields) != 5 + nverts_expected:
            raise ValueError("expected linear lines and homogeneous linear " + execution.mesh_type + " elements")
        vids = fields[5:]
        if len(set(vids)) != len(vids):
            raise ValueError("element references duplicate vertex ids")
        try:
            pts = [nodes[v] for v in vids]
        except KeyError as exc:
            raise ValueError("unknown element node id") from exc

        if kind == 1:
            ekey = (min(vids[0], vids[1]), max(vids[0], vids[1]))
            lbl = boundaries.get(physical)
            if lbl is None or ekey in bdr_edges:
                raise ValueError("duplicate or unknown physical tag on boundary edge")
            p0, p1 = pts
            common = set(_point_side(p0)) & set(_point_side(p1))
            if lbl not in common:
                raise ValueError("boundary edge lies off its declared unit-square side")
            seg_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if seg_len <= _SIDE_EPS:
                raise ValueError("zero-length boundary edge")
            bdr_edges[ekey] = lbl
            side_lengths[lbl] += seg_len
            axis = 0 if lbl in ("bottom", "top") else 1
            a0, a1 = sorted((p0[axis], p1[axis]))
            side_intervals[lbl].append((a0, a1))
        else:
            if physical != surface:
                raise ValueError("element is outside the named physical surface")
            coarse_cells.append(tuple(pts))
            signed_double = sum(
                pts[k - 1][0] * pts[k][1] - pts[k][0] * pts[k - 1][1] for k in range(len(pts))
            )
            if signed_double <= 1e-13:
                raise ValueError("inverted or degenerate element in unstructured Gmsh mesh")
            total_area += 0.5 * signed_double
            nv = len(vids)
            for k in range(nv):
                a, b = vids[k], vids[(k + 1) % nv]
                ekey = (min(a, b), max(a, b))
                elem_edge_counts[ekey] = elem_edge_counts.get(ekey, 0) + 1

    elem_count = len(coarse_cells)
    if elem_count == 0:
        raise ValueError("unstructured mesh contains no 2D elements")
    if abs(total_area - 1.0) > 1e-10:
        raise ValueError(f"total element area {total_area} differs from unit square (overlap or gap)")
    for lbl, segs in side_intervals.items():
        if not segs or abs(side_lengths[lbl] - 1.0) > 1e-10:
            raise ValueError(f"boundary side {lbl!r} length {side_lengths[lbl]} != 1.0")
        segs_sorted = sorted(segs)
        cur = 0.0
        for a0, a1 in segs_sorted:
            if abs(a0 - cur) > 1e-10:
                raise ValueError(f"gap or overlap along boundary side {lbl!r}")
            cur = a1
        if abs(cur - 1.0) > 1e-10:
            raise ValueError(f"boundary side {lbl!r} does not terminate at 1.0")
    for ekey, cnt in elem_edge_counts.items():
        if ekey in bdr_edges:
            if cnt != 1:
                raise ValueError("boundary edge is shared by multiple or zero elements")
        elif cnt != 2:
            raise ValueError("unmatched hanging or slit interior edge in unstructured mesh")
    for ekey in bdr_edges:
        if ekey not in elem_edge_counts:
            raise ValueError("orphan boundary edge not attached to any element")

    sub_hint = max(1, round(math.sqrt(elem_count / (2 if execution.mesh_type == "tri" else 1))))
    grid = UnstructuredGmsh(
        geom=execution.mesh_type,
        node_count=len(nodes),
        element_count=elem_count,
        total_area=total_area,
        boundary_lengths=side_lengths,
        subdivisions_hint=sub_hint,
        coarse_cells=tuple(coarse_cells),
    )
    for level in execution.mesh_levels:
        grid.refinements(level)
    return grid


def generate_unstructured_msh(
    geom: str,
    nx: int,
    *,
    seed: int = 42,
    jitter: float = 0.22,
    graded_focus: tuple[float, float] | None = None,
    out_path: Path | None = None,
) -> bytes:
    """Deterministically generate a conforming unstructured Delaunay tri or curved-grid quad MSH 2.2 file."""
    import numpy as np
    from scipy.spatial import Delaunay

    if geom not in ("tri", "quad"):
        raise ValueError(f"unsupported geometry {geom!r}")
    if nx < 2:
        raise ValueError("nx must be >= 2")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, nx + 1)
    gx, gy = np.meshgrid(t, t, indexing="ij")
    x = gx.copy()
    y = gy.copy()

    if graded_focus is not None:
        cx, cy = graded_focus
        # Smooth monotone boundary/interior grading concentrating vertices near focus.
        def _grade_1d(vals: np.ndarray, c: float) -> np.ndarray:
            s = vals + 0.22 * (c - vals) * np.sin(np.pi * vals)
            return np.clip(s, 0.0, 1.0)

        x = _grade_1d(x, cx)
        y = _grade_1d(y, cy)

    # Interior perturbation breaking affine translation symmetry so element Jacobians are all distinct
    interior = (slice(1, nx), slice(1, nx))
    h = 1.0 / nx
    dx = rng.uniform(-jitter * h, jitter * h, size=(nx - 1, nx - 1))
    dy = rng.uniform(-jitter * h, jitter * h, size=(nx - 1, nx - 1))
    if geom == "quad":
        # Non-affine bilinear distortion + small bounded jitter guaranteeing convex positive quads
        dx += 0.10 * h * np.sin(2.0 * np.pi * x[interior]) * np.sin(np.pi * y[interior])
        dy += 0.10 * h * np.sin(np.pi * x[interior]) * np.cos(np.pi * y[interior])
    x[interior] = np.clip(x[interior] + dx, 1e-4, 1.0 - 1e-4)
    y[interior] = np.clip(y[interior] + dy, 1e-4, 1.0 - 1e-4)

    # Non-uniform spacing along the boundary edges too
    b_shift = rng.uniform(-0.18 * h, 0.18 * h, size=nx - 1)
    x[1:nx, 0] = np.clip(x[1:nx, 0] + b_shift, 1e-4, 1.0 - 1e-4)
    x[1:nx, nx] = np.clip(x[1:nx, nx] - b_shift, 1e-4, 1.0 - 1e-4)
    y[0, 1:nx] = np.clip(y[0, 1:nx] + b_shift, 1e-4, 1.0 - 1e-4)
    y[nx, 1:nx] = np.clip(y[nx, 1:nx] - b_shift, 1e-4, 1.0 - 1e-4)

    pts = np.column_stack((x.ravel(), y.ravel()))
    def _nid(i: int, j: int) -> int:
        return i * (nx + 1) + j + 1

    elems_lines = []
    eid = 1
    for i in range(nx):
        elems_lines.append((eid, 1, 1, 1, _nid(i, 0), _nid(i + 1, 0))); eid += 1
    for j in range(nx):
        elems_lines.append((eid, 1, 2, 2, _nid(nx, j), _nid(nx, j + 1))); eid += 1
    for i in range(nx):
        elems_lines.append((eid, 1, 3, 3, _nid(i, nx), _nid(i + 1, nx))); eid += 1
    for j in range(nx):
        elems_lines.append((eid, 1, 4, 4, _nid(0, j), _nid(0, j + 1))); eid += 1

    if geom == "tri":
        tri = Delaunay(pts)
        for v0, v1, v2 in tri.simplices:
            p0, p1, p2 = pts[v0], pts[v1], pts[v2]
            area2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
            if abs(area2) <= 1e-13:
                continue
            nodes3 = (int(v0) + 1, int(v1) + 1, int(v2) + 1) if area2 > 0 else (int(v0) + 1, int(v2) + 1, int(v1) + 1)
            elems_lines.append((eid, 2, 5, 5, *nodes3)); eid += 1
    else:
        for j in range(nx):
            for i in range(nx):
                elems_lines.append((eid, 3, 5, 5, _nid(i, j), _nid(i + 1, j), _nid(i + 1, j + 1), _nid(i, j + 1)))
                eid += 1

    out = [
        "$MeshFormat", "2.2 0 8", "$EndMeshFormat",
        "$PhysicalNames", "5",
        '1 1 "bottom"', '1 2 "right"', '1 3 "top"', '1 4 "left"', '2 5 "domain"',
        "$EndPhysicalNames",
        "$Nodes", str(len(pts)),
    ]
    for k, (px, py) in enumerate(pts, start=1):
        out.append(f"{k} {px:.17g} {py:.17g} 0")
    out.extend(["$EndNodes", "$Elements", str(len(elems_lines))])
    for row in elems_lines:
        if row[1] == 1:
            out.append(f"{row[0]} 1 2 {row[2]} {row[3]} {row[4]} {row[5]}")
        elif row[1] == 2:
            out.append(f"{row[0]} 2 2 {row[2]} {row[3]} {row[4]} {row[5]} {row[6]}")
        else:
            out.append(f"{row[0]} 3 2 {row[2]} {row[3]} {row[4]} {row[5]} {row[6]} {row[7]}")
    out.append("$EndElements")
    payload = ("\n".join(out) + "\n").encode("ascii")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(payload)
    return payload
