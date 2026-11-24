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
from .spec import ExecutionSpec

Status = Literal["tested", "unsupported", "not_verified"]

MFEM_COMMIT = "d964264cdb9a13e94a201b6c236c7721e0c8765f"
MFEM_VERSION = "4.10.0"
REVIEWED_FLAGS = {flag: flag == "MFEM_USE_DOUBLE" for flag in MFEM_FLAGS}
CASE_IDS = ("sin_sin", "cos_exp", "sin_sin_variable")
MESH_SOURCES = ("cartesian", "gmsh22")
REPORT = "evidence/stage2/coverage.json"
ASSEMBLY_REPORT = "evidence/e1/assembly.json"


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


def evaluate(execution: ExecutionSpec, env: dict | None = None, *,
             case_id: str | None = None, mesh_source: str = "cartesian",
             catalog: dict | None = None) -> CapabilityDecision:
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
    if key not in TESTED or mesh_source not in MESH_SOURCES or (case_id is not None and case_id not in CASE_IDS):
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
    report = ASSEMBLY_REPORT if key in ASSEMBLY_CONTROLS else REPORT
    return CapabilityDecision("tested", "OK", f"{case_id or 'compiled cases'} on {mesh_source} {key} covered by {report}")


def matrix_as_rows(case_id: str | None = None, mesh_source: str = "cartesian") -> list[dict]:
    rows = []
    for mt in ("tri", "quad"):
        for p in (1, 2, 3):
            for asm in ("full", "partial"):
                for sol in ("cg_gs", "cg_jacobi", "cg_none", "umfpack", "hypre_amg"):
                    ex = ExecutionSpec(mesh_type=mt, polynomial_order=p, assembly=asm, solver=sol)
                    d = evaluate(ex, case_id=case_id, mesh_source=mesh_source)
                    rows.append({"mesh_type": mt, "order": p, "assembly": asm, "solver": sol,
                                 "mesh_source": mesh_source, "backend": ex.backend,
                                 "status": d.status, "code": d.code, "reason": d.reason,
                                 "reviewed_build": {"version": MFEM_VERSION, "commit": MFEM_COMMIT,
                                                   "flags": REVIEWED_FLAGS.copy()},
                                 "verification": None if d.status != "tested" else {
                                     "case_ids": [case_id] if case_id else list(CASE_IDS),
                                     "solver_class": "CGSolver",
                                     "preconditioner": ("GSSmoother(SparseMatrix)" if sol == "cg_gs"
                                                        else "OperatorJacobiSmoother"),
                                     "space": "scalar H1, positive diffusion, essential Dirichlet DOFs",
                                     "report": ASSEMBLY_REPORT if (mt, p, asm, sol) in ASSEMBLY_CONTROLS else REPORT}})
    return rows
