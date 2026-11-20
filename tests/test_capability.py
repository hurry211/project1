"""Unit tests: the capability matrix rejects unsupported/unverified combinations."""

import pytest

from sciagent import capability
from sciagent.spec import ExecutionSpec

ENV_SERIAL = {"mfem": {"config": {"MFEM_USE_MPI": False, "MFEM_USE_CUDA": False, "MFEM_USE_SUITESPARSE": False}}}


@pytest.mark.parametrize("kw,code", [
    ({"backend": "mpi_cpu"}, "UNSUPPORTED_BACKEND"),
    ({"backend": "cuda"}, "UNSUPPORTED_BACKEND"),
    ({"backend": "hip"}, "UNSUPPORTED_BACKEND"),
    ({"solver": "umfpack"}, "UNSUPPORTED_SOLVER"),
    ({"solver": "hypre_amg"}, "UNSUPPORTED_SOLVER"),
    ({"assembly": "partial", "solver": "cg_gs"}, "UNSUPPORTED_COMBINATION"),
    ({"polynomial_order": 3}, "NOT_VERIFIED_COMBINATION"),
    ({"solver": "cg_none"}, "NOT_VERIFIED_COMBINATION"),
])
def test_rejections(kw, code):
    d = capability.evaluate(ExecutionSpec(**kw), ENV_SERIAL)
    assert d.status != "tested" and d.code == code


def test_mpi_flag_alone_does_not_promote_to_tested():
    env = {"mfem": {"config": {"MFEM_USE_MPI": True}}}
    d = capability.evaluate(ExecutionSpec(backend="mpi_cpu"), env)
    assert d.status == "not_verified"


@pytest.mark.parametrize("key", sorted(capability.TESTED))
def test_tested_entries_accepted(key):
    mt, p, asm, sol = key
    d = capability.evaluate(ExecutionSpec(mesh_type=mt, polynomial_order=p, assembly=asm, solver=sol), ENV_SERIAL)
    assert d.status == "tested"


def test_matrix_rows_cover_all_statuses():
    statuses = {r["status"] for r in capability.matrix_as_rows()}
    assert statuses == {"tested", "unsupported", "not_verified"}
