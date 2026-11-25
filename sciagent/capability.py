"""Reviewed capability (compatibility) matrix.

A combination is runnable only if its status is ``tested``. The statuses mean:

* ``tested``        -- exercised by the serial or assembly comparison matrix with real MFEM and
                       an independent scikit-fem solution on each mesh.
* ``unsupported``   -- known not to work, or the required dependency is absent
                       from the probed MFEM build; rejected with a reason.
* ``not_verified``  -- may well work in MFEM but has not been exercised here;
                       rejected by default (no silent promotion to tested).

Keys are deliberately explicit rather than derived from MFEM's feature list:
a library supporting features A and B separately does not imply that A+B work
together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .environment import MFEM_FLAGS
from .spec import (ANISOTROPIC, GAUSSIAN_EXACT, GAUSSIAN_LEVELS, LAYERED, LAYERED_LEVELS,
                   POISSON, ExecutionSpec, compact_expression)

Status = Literal["tested", "unsupported", "not_verified"]

MFEM_COMMIT = "d964264cdb9a13e94a201b6c236c7721e0c8765f"
MFEM_VERSION = "4.10.0"
REVIEWED_FLAGS = {flag: flag == "MFEM_USE_DOUBLE" for flag in MFEM_FLAGS}
CASE_IDS = ("sin_sin", "cos_exp", "sin_sin_variable")
SPECIAL_REPORTS = {
    "anisotropic_1000_sin_sin": "evidence/performance-stage/anisotropic-manifest.json",
    "gaussian_offset": "docs/VERIFIER_SCALING.md",
    "layered_unit_drop_10000": "docs/VERIFIER_SCALING.md",
}
E3_CASE_IDS = ("gaussian_steep", "lshape_singularity", "gaussian_mid", "sin_2x_2y", "sin_2x_3y")
ALL_CASE_IDS = (*CASE_IDS, *SPECIAL_REPORTS)
MESH_SOURCES = ("cartesian", "gmsh22")
REPORT = "evidence/stage2/coverage.json"
ASSEMBLY_REPORT = "evidence/e1/assembly.json"
E3_REPORT = "docs/E3_PERFORMANCE_ADAPTIVE.md"


@dataclass(frozen=True)
class CapabilityDecision:
    status: Status
    code: str          # stable machine-readable code
    reason: str


# (mesh_type, order, assembly, solver) entries exercised by integration tests.
STAGE2_CONFIGURATIONS: frozenset[tuple[str, int, str, str]] = frozenset({
    ("tri", 1, "full", "cg_gs"),
    ("tri", 2, "full", "cg_gs"),
    ("quad", 1, "full", "cg_gs"),
    ("quad", 2, "full", "cg_gs"),
    ("tri", 1, "full", "cg_jacobi"),
    ("quad", 1, "partial", "cg_jacobi"),
    ("quad", 2, "partial", "cg_jacobi"),
})
ASSEMBLY_CONTROLS = frozenset(("quad", p, "full", "cg_jacobi") for p in (1, 2))
TESTED = STAGE2_CONFIGURATIONS | ASSEMBLY_CONTROLS
E3_CONFIGURATIONS: frozenset[tuple[str, int, str, str]] = frozenset(
    [(mt, p, "full", sol) for mt in ("tri", "quad") for p in (1, 2, 3, 4) for sol in ("cg_gs", "cg_jacobi", "cg_none")]
    + [(mt, p, "partial", sol) for mt in ("tri", "quad") for p in (1, 2, 3, 4) for sol in ("cg_jacobi", "cg_none")]
)


def _build_mismatch(mfem: dict, catalog: dict | None) -> str | None:
    revision = mfem.get("git") or ""
    if (mfem.get("version") != MFEM_VERSION or
            not (revision == MFEM_COMMIT or revision.endswith("-g" + MFEM_COMMIT)) or
            any(mfem.get("config", {}).get(k) is not v for k, v in REVIEWED_FLAGS.items())):
        return "the observed MFEM revision or configuration differs from the reviewed build"
    if catalog is None:
        return None
    release = f"MFEM v{MFEM_VERSION} (release)"
    if catalog.get("registry_version") == 2:
        compiled = catalog.get("mfem_config")
        if (catalog.get("mfem_version") != release or catalog.get("mfem_git") != revision
                or not isinstance(compiled, dict) or any(compiled.get(k) is not v for k, v in REVIEWED_FLAGS.items())):
            return "the solver build metadata differs from the installed MFEM probe"
    elif "mfem_version" in catalog and catalog["mfem_version"] != release:
        return "the legacy solver reports a different MFEM release"
    return None


def evaluate(execution: ExecutionSpec, env: dict | None = None, equation: str = POISSON, *,
             exact_solution: str | None = None, case_id: str | None = None, mesh_source: str = "cartesian",
             catalog: dict | None = None, e3_verified: bool = False) -> CapabilityDecision:
    """Decide whether an execution configuration may run.

    ``env`` is the environment probe (sciagent.environment.probe()); it is
    used to reject backends whose dependencies are not in the MFEM build.
    """
    mfem = (env or {}).get("mfem", {})
    mfem_cfg = mfem.get("config", {})

    if execution.backend != "serial_cpu":
        dep = {"mpi_cpu": "MFEM_USE_MPI", "cuda": "MFEM_USE_CUDA", "hip": "MFEM_USE_HIP"}[execution.backend]
        if not mfem_cfg.get(dep, False):
            return CapabilityDecision(
                "unsupported", "UNSUPPORTED_BACKEND",
                f"backend '{execution.backend}' requires {dep}, which is not enabled in the "
                f"probed MFEM build (presence of an mpirun/nvcc binary is not sufficient)",
            )
        return CapabilityDecision(
            "not_verified", "NOT_VERIFIED_COMBINATION",
            f"backend '{execution.backend}' has no tested solver path in this MVP",
        )

    if execution.solver in ("umfpack", "hypre_amg"):
        dep = {"umfpack": "MFEM_USE_SUITESPARSE", "hypre_amg": "MFEM_USE_MPI"}[execution.solver]
        return CapabilityDecision(
            "unsupported", "UNSUPPORTED_SOLVER",
            f"solver '{execution.solver}' requires {dep} (enabled={mfem_cfg.get(dep, False)}) "
            f"and has no adapter in poisson_mms",
        )

    if execution.assembly == "partial" and execution.solver == "cg_gs":
        return CapabilityDecision(
            "unsupported", "UNSUPPORTED_COMBINATION",
            "Gauss-Seidel smoothing needs an assembled SparseMatrix; partial assembly "
            "provides only a matrix-free operator (use cg_jacobi with partial assembly)",
        )

    key = (execution.mesh_type, execution.polynomial_order, execution.assembly, execution.solver)
    if case_id is None:
        if compact_expression(exact_solution) == GAUSSIAN_EXACT:
            case_id = "gaussian_offset"
        elif equation == LAYERED:
            case_id = "layered_unit_drop_10000"
        elif equation == ANISOTROPIC:
            case_id = "anisotropic_1000_sin_sin"
    if case_id in SPECIAL_REPORTS and mesh_source != "cartesian":
        return CapabilityDecision("not_verified", "NOT_VERIFIED_COMBINATION", "this case requires a generated Cartesian mesh")
    if case_id == "gaussian_offset" and (
            equation != POISSON or key not in {("quad", 2, a, "cg_jacobi") for a in ("full", "partial")}
            or any(n not in GAUSSIAN_LEVELS for n in execution.mesh_levels)):
        return CapabilityDecision("not_verified", "NOT_VERIFIED_COMBINATION",
                                  f"localized Gaussian is tested with Q2 CG+Jacobi on n={GAUSSIAN_LEVELS}")
    if execution.inject_fault == "wrong_diffusion" and case_id != "layered_unit_drop_10000":
        return CapabilityDecision("unsupported", "UNSUPPORTED_COMBINATION",
                                  "the wrong_diffusion control targets the layered coefficient")
    if case_id == "layered_unit_drop_10000" and (
            key != ("quad", 1, "full", "cg_gs") or any(n not in LAYERED_LEVELS for n in execution.mesh_levels)):
        return CapabilityDecision("not_verified", "NOT_VERIFIED_COMBINATION",
                                  f"layered diffusion is tested with full Q1 CG+GS on n={LAYERED_LEVELS}")
    if case_id == "anisotropic_1000_sin_sin" and key != ("quad", 1, "full", "cg_jacobi"):
        return CapabilityDecision("not_verified", "NOT_VERIFIED_COMBINATION",
                                  "the anisotropic benchmark is tested only with full Q1 CG+Jacobi")
    is_e3 = e3_verified or execution.mesh_topology != "cartesian" or (case_id in E3_CASE_IDS)
    allowed = E3_CONFIGURATIONS if is_e3 else TESTED
    valid_cases = (*ALL_CASE_IDS, *E3_CASE_IDS) if is_e3 else ALL_CASE_IDS
    valid_sources = (*MESH_SOURCES, "amr") if execution.mesh_topology == "amr" else MESH_SOURCES
    if execution.mesh_topology == "amr" and mesh_source == "cartesian":
        mesh_source = "amr"
    if key not in allowed or mesh_source not in valid_sources or (case_id is not None and case_id not in valid_cases):
        return CapabilityDecision("not_verified", "NOT_VERIFIED_COMBINATION",
                                  f"no coverage for case={case_id}, mesh_source={mesh_source}, combination={key}")
    if env is not None:
        mismatch = _build_mismatch(mfem, catalog)
        if mismatch:
            return CapabilityDecision("not_verified", "NOT_VERIFIED_BUILD", mismatch)
    if catalog is not None and mesh_source == "gmsh22" and (
            catalog.get("registry_version") != 2 or not isinstance(catalog.get("mesh_sources"), list)
            or mesh_source not in catalog["mesh_sources"]):
        return CapabilityDecision("not_verified", "NOT_VERIFIED_COMBINATION",
                                  "this solver catalog does not advertise the reviewed Gmsh input path")
    report = E3_REPORT if is_e3 else SPECIAL_REPORTS.get(case_id, ASSEMBLY_REPORT if key in ASSEMBLY_CONTROLS else REPORT)
    return CapabilityDecision("tested", "OK", f"{case_id or 'compiled cases'} on {mesh_source} {key} covered by {report}")


def matrix_as_rows(case_id: str | None = None, mesh_source: str = "cartesian") -> list[dict]:
    rows = []
    levels = {"gaussian_offset": GAUSSIAN_LEVELS, "layered_unit_drop_10000": LAYERED_LEVELS}.get(case_id)
    for mt in ("tri", "quad"):
        for p in (1, 2, 3):
            for asm in ("full", "partial"):
                for sol in ("cg_gs", "cg_jacobi", "cg_none", "umfpack", "hypre_amg"):
                    ex = ExecutionSpec(mesh_type=mt, polynomial_order=p, assembly=asm, solver=sol)
                    if levels is not None:
                        ex = ex.model_copy(update={"mesh_levels": levels})
                    d = evaluate(ex, case_id=case_id, mesh_source=mesh_source)
                    rows.append({"mesh_type": mt, "order": p, "assembly": asm, "solver": sol,
                                 "mesh_source": mesh_source, "backend": ex.backend,
                                 "status": d.status, "code": d.code, "reason": d.reason,
                                 "reviewed_build": {"version": MFEM_VERSION, "commit": MFEM_COMMIT,
                                                   "flags": REVIEWED_FLAGS.copy()},
                                 "verification": None if d.status != "tested" else {
                                     "case_ids": [case_id] if case_id else list(CASE_IDS),
                                     **({"mesh_levels": levels} if levels is not None else {}),
                                     "solver_class": "CGSolver",
                                     "preconditioner": ("GSSmoother(SparseMatrix)" if sol == "cg_gs"
                                                        else "OperatorJacobiSmoother"),
                                     "space": "scalar H1, positive diffusion, essential Dirichlet DOFs",
                                     "report": SPECIAL_REPORTS.get(case_id, ASSEMBLY_REPORT if (mt, p, asm, sol) in ASSEMBLY_CONTROLS else REPORT)}})
    if case_id is None and mesh_source == "cartesian":
        for special in SPECIAL_REPORTS:
            rows.extend(row for row in matrix_as_rows(special, mesh_source) if row["status"] == "tested")
    return rows
