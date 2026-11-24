"""A planner nominates one reviewed input. The user's structured task authorizes it."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .gmsh import read_mesh
from .spec import StudySpec


SPECS = Path(__file__).resolve().parents[1] / "specs"
SELECTORS = ("case", "mesh_type", "polynomial_order", "assembly", "solver", "mesh_source")
Identifier = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")]


class PlanningTask(BaseModel):
    """Structured selectors authorize the physics and permissible discretizations.

    Description ranks authorized alternatives. Use change_request for a new
    model, different thresholds, resource limits, or another change requiring
    review. Provide a new reviewed input through the ordinary revision process.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    case: Identifier
    mesh_type: Identifier | None = None
    polynomial_order: int | None = Field(default=None, ge=1, le=128)
    assembly: Identifier | None = None
    solver: Identifier | None = None
    mesh_source: Identifier | None = None
    description: str = Field(default="", max_length=2000)
    change_request: str = Field(default="", max_length=2000)


@dataclass(frozen=True)
class _Template:
    path: str
    case: str
    sha256: str

    def read(self) -> tuple[bytes, StudySpec]:
        # Preserve the pinned Git blob across Windows text checkouts. Mesh
        # digests below cover raw bytes exactly, matching the existing workflow.
        path = SPECS / self.path
        if not path.is_file():
            raise ValueError(f"missing regular reviewed study: {self.path}")
        with path.open("rb") as stream:
            raw = stream.read(16385)
        if len(raw) > 16384:
            raise ValueError(f"reviewed study exceeds 16 KiB: {self.path}")
        raw = raw.replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError(f"reviewed study digest mismatch: {self.path}")
        return raw, StudySpec.model_validate_json(raw)


# Reviewed identities and canonical spec-byte digests. Execution details in
# catalog_rows() come directly from these checked documents.
_TEMPLATES = {
    Path(t.path).stem: t for t in (
        _Template("poisson_cos_exp_mixed_p1_quad.json", "cos_exp", "58e2f445c187064bf617357a7b61950b3467ea4960d9b8f89069b5dc23032ad2"),
        _Template("poisson_cos_exp_mixed_p1_quad_pa.json", "cos_exp", "3f1ea37e13071ce039f8d57e9228f499d9c79e743bb37fa71a67852472e04f6b"),
        _Template("poisson_cos_exp_mixed_p2_quad.json", "cos_exp", "0dfc173841e3e9e0dbf5093ab226fb089aa356ad60f407843f8ec96021967e68"),
        _Template("poisson_cos_exp_mixed_p2_quad_pa.json", "cos_exp", "1770d5f5cd57cf18dbd400774a1cfae5c54435c15ccd546f5a21a4ed15fefc62"),
        _Template("poisson_sin_sin_p1_tri.json", "sin_sin", "ef13bcb4dda17fc54c2cb8e81fcb8e4e1a7afb7ffe3f72377f85f73d6ef9803c"),
        _Template("poisson_sin_sin_p1_tri_jacobi.json", "sin_sin", "64bb3e4b3d09e1fee71c2bcdfca72a9466c1f7b4f23c6810f07222a58936e6ad"),
        _Template("poisson_sin_sin_p2_tri.json", "sin_sin", "2ac77796808b23372a255200f79ce8f035def1699328634d5551568427f02b4a"),
        _Template("stage2/diffusion_gmsh_p2_quad.json", "sin_sin_variable", "b5d6ba04734ad7452e27fea841549036ceaee06cc4436fe6280c54b57dab1d78"),
        _Template("stage2/diffusion_mixed_p1_tri.json", "sin_sin_variable", "d01e48d4aad7c695d6d7a62b142d7b53c6830ec2f546b94af3ac98e452afc591"),
    )
}


def _strict_object(raw: str | bytes, size_limit: int) -> dict:
    if not isinstance(raw, (str, bytes)) or len(raw) > size_limit:
        raise ValueError("expected a size-limited JSON object")
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    if len(data) > size_limit:
        raise ValueError("JSON object exceeds byte limit")

    def unique_keys(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError("duplicate JSON key")
            obj[k] = v
        return obj

    def reject_constant(value):
        raise ValueError(f"invalid JSON number: {value}")

    try:
        obj = json.loads(data.decode("utf-8"), object_pairs_hook=unique_keys, parse_constant=reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON exceeds nesting limit") from exc
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj


def parse_task(raw: str | bytes) -> PlanningTask:
    return PlanningTask.model_validate(_strict_object(raw, 8192))


def _template(template_id: str) -> _Template:
    if type(template_id) is not str or template_id not in _TEMPLATES:
        raise ValueError("unknown reviewed template")
    return _TEMPLATES[template_id]


def template_sha256(template_id: str) -> str:
    return _template(template_id).sha256


def catalog_rows() -> list[dict]:
    rows = []
    for key, template in sorted(_TEMPLATES.items()):
        _, study = template.read()
        execution = study.execution
        rows.append({"template_id": key, "spec_sha256": template.sha256, "case": template.case,
                     "mesh_type": execution.mesh_type, "polynomial_order": execution.polynomial_order,
                     "assembly": execution.assembly, "solver": execution.solver,
                     "mesh_source": "gmsh22" if study.external_mesh else "cartesian"})
    return rows


def _matches(task: PlanningTask, row: dict) -> bool:
    return all(getattr(task, key) in (None, row[key]) for key in SELECTORS)


def eligible_ids(task: PlanningTask) -> tuple[str, ...]:
    if task.change_request.strip():
        return ()
    return tuple(row["template_id"] for row in catalog_rows() if _matches(task, row))


def _decision(status, code, template_id=None) -> dict:
    return {"status": status, "code": code, "template_id": template_id}


def preflight(task: PlanningTask) -> dict | None:
    if task.change_request.strip():
        return _decision("REVIEW_REQUIRED", "CHANGE_REQUIRES_REVIEW")
    if task.case not in {t.case for t in _TEMPLATES.values()}:
        return _decision("REVIEW_REQUIRED", "UNREVIEWED_CASE")
    if not eligible_ids(task):
        return _decision("REVIEW_REQUIRED", "NO_REVIEWED_CONFIGURATION")
    return None


def rule_proposal(task: PlanningTask) -> str:
    choices = eligible_ids(task)
    return json.dumps({"template_id": choices[0] if choices else None})


def build_prompt(task: PlanningTask) -> str:
    candidates = [{k: v for k, v in row.items() if k != "spec_sha256"}
                  for row in catalog_rows() if _matches(task, row) and not task.change_request.strip()]
    inputs = {"preference": task.description, "authorized_choices": candidates}
    return ('Select the authorized finite-element study that best fits the preference. '
            'P1 is linear and P2 is quadratic. Partial assembly applies the operator matrix-free. '
            'cg_gs uses Gauss-Seidel and cg_jacobi uses Jacobi preconditioning. '
            'If multiple choices fit equally, take the first listed. Return only a JSON object '
            'with one key template_id and the selected identifier as its value. Use null to abstain.\n'
            + json.dumps(inputs, ensure_ascii=False, separators=(",", ":")))


def decide(task: PlanningTask, raw: str | bytes) -> dict:
    review = preflight(task)
    if review is not None:
        return review
    try:
        obj = _strict_object(raw, 512)
        if set(obj) != {"template_id"} or (obj["template_id"] is not None and type(obj["template_id"]) is not str):
            raise ValueError("invalid candidate shape")
    except ValueError:
        return _decision("REJECTED", "INVALID_PROPOSAL")
    key = obj["template_id"]
    if key is None:
        return _decision("REJECTED", "PLANNER_ABSTAINED")
    if key not in eligible_ids(task):
        return _decision("REJECTED", "UNAUTHORIZED_TEMPLATE")
    return _decision("PLANNED", "REVIEWED_TEMPLATE", key)


def materialize(template_id: str, new_directory: Path) -> tuple[Path, str]:
    """Copy a verified byte snapshot into a new directory for the deterministic controller."""
    template = _template(template_id)
    raw, study = template.read()
    mesh = study.external_mesh
    if mesh:
        data = read_mesh((SPECS / template.path).with_name(mesh.file), mesh)
    directory = Path(new_directory)
    directory.mkdir(parents=True, exist_ok=False)
    if mesh:
        with (directory / mesh.file).open("xb") as f:
            f.write(data)
    path = directory / "study.json"
    with path.open("xb") as f:
        f.write(raw)
    return path, template.sha256
