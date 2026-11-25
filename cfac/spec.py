"""Immutable rational input to the unit-square P1/RT0 Poisson experiment."""

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from collections.abc import Mapping, Set


def ordered(values):
    """Reject scalar text and unordered containers in indexed coefficient data."""
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise ValueError("expected an ordered sequence of coefficients")
    return tuple(values)


def rational(value):
    """Interpret finite binary floats exactly. Strings may carry rational data."""
    if type(value) not in (int, str, float, Fraction):
        raise ValueError("expected an integer, rational, rational string or finite float")
    try:
        return Fraction(value)
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("expected a finite rational") from exc


def digest_parts(parts):
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class Mesh:
    vertices: tuple
    triangles: tuple

    def __post_init__(self):
        vertices = tuple(tuple(rational(v) for v in ordered(p)) for p in ordered(self.vertices))
        triangles = tuple(ordered(t) for t in ordered(self.triangles))
        if any(len(p) != 2 for p in vertices):
            raise ValueError("expected two coordinates per vertex")
        if any(len(t) != 3 or any(type(i) is not int for i in t) for t in triangles):
            raise ValueError("expected three integer indices per triangle")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)


@dataclass(frozen=True, slots=True)
class Problem:
    """P0 loads belong to the ordered cells. An optional affine original is c+a*x+b*y."""

    mesh: Mesh
    loads: tuple
    diffusion: Fraction
    epsilon: Fraction
    max_actions: int
    original_affine: tuple | None = None

    def __post_init__(self):
        if type(self.mesh) is not Mesh:
            raise ValueError("expected an immutable rational mesh")
        object.__setattr__(self, "loads", tuple(rational(f) for f in ordered(self.loads)))
        for name in ("diffusion", "epsilon"):
            value = rational(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if type(self.max_actions) is not int or self.max_actions < 0:
            raise ValueError("max_actions must be a nonnegative integer")
        if len(self.loads) != len(self.mesh.triangles):
            raise ValueError("expected one load per cell")
        if self.original_affine is not None:
            coef = tuple(rational(v) for v in ordered(self.original_affine))
            if len(coef) != 3:
                raise ValueError("expected c, a, b for the affine original load")
            object.__setattr__(self, "original_affine", coef)

    def digest(self):
        """Capture this value in trusted caller storage before producing candidate fields."""
        affine = self.original_affine
        return digest_parts([
            "CFAC/v1/unit-square/zero-dirichlet/energy/P1/RT0-right-normal",
            [[str(x), str(y)] for x, y in self.mesh.vertices],
            self.mesh.triangles, list(map(str, self.loads)),
            str(self.diffusion), str(self.epsilon), self.max_actions,
            None if affine is None else list(map(str, affine)),
        ])
