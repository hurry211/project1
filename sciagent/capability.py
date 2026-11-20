"""Reviewed capability (compatibility) matrix.

A combination is runnable only if its status is ``tested``. The statuses mean:

* ``tested``        -- exercised end-to-end by tests/test_integration_mfem.py
                       against MFEM v4.10 in this repository's environment.
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

from .spec import ExecutionSpec

Status = Literal["tested", "unsupported", "not_verified"]


@dataclass(frozen=True)
class CapabilityDecision:
    status: Status
    code: str          # stable machine-readable code
    reason: str


# (mesh_type, order, assembly, solver) entries exercised by integration tests.
TESTED: frozenset[tuple[str, int, str, str]] = frozenset({
    ("tri", 1, "full", "cg_gs"),
    ("tri", 2, "full", "cg_gs"),
    ("quad", 1, "full", "cg_gs"),
    ("quad", 2, "full", "cg_gs"),
    ("tri", 1, "full", "cg_jacobi"),
    ("quad", 1, "partial", "cg_jacobi"),
    ("quad", 2, "partial", "cg_jacobi"),
})


def evaluate(execution: ExecutionSpec, env: dict | None = None) -> CapabilityDecision:
    """Decide whether an execution configuration may run.

    ``env`` is the environment probe (sciagent.environment.probe()); it is
    used to reject backends whose dependencies are not in the MFEM build.
    """
    env = env or {}
    mfem_cfg = env.get("mfem", {}).get("config", {})

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
    if key in TESTED:
        return CapabilityDecision("tested", "OK", f"combination {key} is covered by integration tests")
    return CapabilityDecision(
        "not_verified", "NOT_VERIFIED_COMBINATION",
        f"combination {key} has not been exercised by the integration tests",
    )


def matrix_as_rows() -> list[dict]:
    rows = []
    for mt in ("tri", "quad"):
        for p in (1, 2, 3):
            for asm in ("full", "partial"):
                for sol in ("cg_gs", "cg_jacobi", "cg_none", "umfpack", "hypre_amg"):
                    ex = ExecutionSpec(mesh_type=mt, polynomial_order=p, assembly=asm, solver=sol)
                    d = evaluate(ex, {})
                    rows.append({"mesh_type": mt, "order": p, "assembly": asm, "solver": sol,
                                 "status": d.status, "code": d.code})
    return rows
