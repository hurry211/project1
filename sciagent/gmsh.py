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
