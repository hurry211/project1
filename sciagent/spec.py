"""Typed data contracts.

Four separate objects with different mutability rules:

* ``ProblemSpec``     -- the mathematical problem. Frozen once accepted.
* ``ValidationSpec``  -- pre-committed acceptance criteria. Frozen; the
                          controller can never relax it.
* ``RetryPolicy``     -- bounds on the corrective loop. Frozen.
* ``ExecutionSpec``   -- numerical/execution settings. The controller may
                          derive new versions of it, but only through the
                          actions allowed by ``RetryPolicy``.

Immutability is enforced twice: pydantic ``frozen=True`` models inside one
process, and SHA-256 hashes of the canonical JSON that are written to the run
manifest and re-checked before verification and before every retry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "0.2"

SideLabel = Literal["bottom", "right", "top", "left"]

# Geometric definition of side labels on the unit square. This is the single
# place where a *physical label* is tied to *geometry*; mesh attribute numbers
# are tied to labels only through the ProblemSpec and are checked against the
# exported boundary geometry by the verifier.
SIDE_GEOMETRY: dict[str, tuple[int, float]] = {
    "bottom": (1, 0.0),  # coordinate index 1 (y) == 0
    "right": (0, 1.0),   # x == 1
    "top": (1, 1.0),     # y == 1
    "left": (0, 0.0),    # x == 0
}
SIDE_NORMAL: dict[str, tuple[float, float]] = {
    "bottom": (0.0, -1.0),
    "right": (1.0, 0.0),
    "top": (0.0, 1.0),
    "left": (-1.0, 0.0),
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class BoundaryCondition(_Frozen):
    attribute: int = Field(ge=1, description="MFEM boundary attribute number")
    label: SideLabel
    kind: Literal["dirichlet", "neumann"]
    # MVP restriction: boundary data is always derived from the manufactured
    # exact solution (u for Dirichlet, grad(u).n for Neumann). Any other value
    # is rejected rather than silently interpreted.
    value: Literal["from_exact_solution"]


class ProblemSpec(_Frozen):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_\-]+$")
    equation: Literal["-laplacian(u) = f"]
    domain: Literal["unit_square_2d"]
    exact_solution: str = Field(min_length=1, max_length=400)
    source: str = Field(min_length=1, max_length=400)
    boundaries: tuple[BoundaryCondition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_boundaries(self) -> "ProblemSpec":
        attrs = [b.attribute for b in self.boundaries]
        if len(set(attrs)) != len(attrs):
            raise ValueError("duplicate boundary attribute in ProblemSpec")
        labels = [b.label for b in self.boundaries]
        if len(set(labels)) != len(labels):
            raise ValueError("duplicate boundary label in ProblemSpec")
        if set(labels) != set(SIDE_GEOMETRY):
            missing = sorted(set(SIDE_GEOMETRY) - set(labels))
            raise ValueError(f"boundary labels do not cover the unit square; missing {missing}")
        if not any(b.kind == "dirichlet" for b in self.boundaries):
            raise ValueError("pure Neumann problem (constant nullspace) is not supported")
        return self

    def dirichlet_attributes(self) -> list[int]:
        return sorted(b.attribute for b in self.boundaries if b.kind == "dirichlet")

    def neumann_attributes(self) -> list[int]:
        return sorted(b.attribute for b in self.boundaries if b.kind == "neumann")


class ValidationSpec(_Frozen):
    """Pre-committed acceptance criteria (see docs/MVP_SPEC.md for rationale)."""

    require_solver_converged: bool = True
    min_levels: int = Field(default=3, ge=3)
    # Observed order on the finest pair must lie in
    # [expected - min_offset, expected + max_offset]; expected = p+1 (L2), p (H1).
    l2_order_min_offset: float = Field(default=0.2, ge=0.0, le=1.0)
    l2_order_max_offset: float = Field(default=0.5, ge=0.0, le=2.0)
    h1_order_min_offset: float = Field(default=0.2, ge=0.0, le=1.0)
    h1_order_max_offset: float = Field(default=0.5, ge=0.0, le=2.0)
    require_monotone_decrease: bool = True
    # Accuracy target on the finest level (None = no target). Only a failure
    # of this target is eligible for the bounded refinement retry.
    max_l2_error_finest: float | None = Field(default=None, gt=0.0)
    dirichlet_nodal_abs_tol: float = Field(default=1e-10, gt=0.0)
    self_report_rel_tol: float = Field(default=1e-6, gt=0.0)
    require_reference_solve: bool = True
    reference_solve_rel_tol: float = Field(default=1e-7, gt=0.0)
    reference_solve_max_dofs: int = Field(default=300_000, ge=1)
    mms_consistency_rel_tol: float = Field(default=1e-6, gt=0.0)
    verifier_quadrature_extra_points: int = Field(default=3, ge=1, le=10)


class RetryPolicy(_Frozen):
    max_attempts: int = Field(default=3, ge=1, le=6)
    action: Literal["append_refined_level"] = "append_refined_level"
    max_dofs_per_level: int = Field(default=300_000, ge=1)
    max_total_wall_seconds: float = Field(default=900.0, gt=0.0)
    per_level_timeout_seconds: float = Field(default=300.0, gt=0.0)
    per_level_memory_mb: int = Field(default=3072, ge=256)


class ExecutionSpec(_Frozen):
    mesh_type: Literal["tri", "quad"] = "tri"
    polynomial_order: int = Field(default=1, ge=1, le=8)
    mesh_levels: tuple[int, ...] = Field(default=(8, 16, 32), min_length=1)
    backend: Literal["serial_cpu", "mpi_cpu", "cuda", "hip"] = "serial_cpu"
    assembly: Literal["full", "partial"] = "full"
    solver: Literal["cg_gs", "cg_jacobi", "cg_none", "umfpack", "hypre_amg"] = "cg_gs"
    rel_tol: float = Field(default=1e-12, gt=0.0, lt=1e-2)
    max_iter: int = Field(default=20_000, ge=1)
    inject_fault: Literal[
        "none", "wrong_source", "drop_dirichlet_attr", "nan_value",
        "early_stop_lie", "self_report_lie",
    ] = "none"

    @field_validator("mesh_levels")
    @classmethod
    def _levels(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(n < 1 or n > 4096 for n in v):
            raise ValueError("mesh level n must be in [1, 4096]")
        for a, b in zip(v, v[1:]):
            if b != 2 * a:
                raise ValueError("mesh_levels must be successive doublings (h, h/2, ...)")
        return v

    def estimated_dofs(self, n: int) -> int:
        p = self.polynomial_order
        if self.mesh_type == "quad":
            return (n * p + 1) ** 2
        return (n * p + 1) ** 2  # P_p on a uniformly split Cartesian triangulation


class StudySpec(_Frozen):
    """A complete user-facing specification file."""

    schema_version: Literal["0.2"]
    problem: ProblemSpec
    execution: ExecutionSpec
    validation: ValidationSpec = ValidationSpec()
    policy: RetryPolicy = RetryPolicy()


def canonical_json(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def compact_expression(value: object) -> str | None:
    return "".join(value.split()) if isinstance(value, str) else None


def sha256_of(model: BaseModel) -> str:
    return hashlib.sha256(canonical_json(model).encode()).hexdigest()


def load_study(path: str | Path) -> StudySpec:
    data = json.loads(Path(path).read_text())
    return StudySpec.model_validate(data)
