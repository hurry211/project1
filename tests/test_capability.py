"""Unit tests: the capability matrix rejects unsupported/unverified combinations."""

import pytest

from sciagent import capability
from sciagent.spec import ExecutionSpec

ENV_SERIAL = {"mfem": {"version": capability.MFEM_VERSION, "git": capability.MFEM_COMMIT,
                       "config": capability.REVIEWED_FLAGS.copy()}}


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


@pytest.mark.parametrize("overrides", [
    {"version": "4.9"}, {"git": None}, {"git": capability.MFEM_COMMIT + "-dirty"},
    {"config": {}}, {"config": {**capability.REVIEWED_FLAGS, "MFEM_USE_SINGLE": True}},
])
def test_unreviewed_build_cannot_inherit_tested_status(overrides):
    env = {"mfem": {**ENV_SERIAL["mfem"], **overrides}}
    assert capability.evaluate(ExecutionSpec(), env).code == "NOT_VERIFIED_BUILD"


def test_reviewed_coverage_is_explicit():
    assert capability.evaluate(ExecutionSpec(), case_id="unreviewed").status == "not_verified"
    assert capability.evaluate(ExecutionSpec(), mesh_source="curved").status == "not_verified"
    rows = capability.matrix_as_rows("sin_sin_variable", "gmsh22")
    tested = [r for r in rows if r["status"] == "tested"]
    assert len(tested) == 7
    for row in tested:
        assert row["verification"]["case_ids"] == ["sin_sin_variable"]
        assert row["verification"]["preconditioner"].endswith("Smoother") or row["solver"] == "cg_gs"
        assert row["reviewed_build"]["flags"] == capability.REVIEWED_FLAGS


def test_probe_for_another_install_cannot_certify_a_different_solver():
    catalog = {"registry_version": 2, "mfem_git": capability.MFEM_COMMIT,
               "mfem_version": "MFEM v4.10.0 (release)", "mfem_config": capability.REVIEWED_FLAGS}
    assert capability.evaluate(ExecutionSpec(), ENV_SERIAL, catalog=catalog).code == "OK"
    assert capability.evaluate(ExecutionSpec(), ENV_SERIAL, mesh_source="gmsh22", catalog=catalog).status == "not_verified"
    assert capability.evaluate(ExecutionSpec(), ENV_SERIAL, mesh_source="gmsh22",
                               catalog={**catalog, "mesh_sources": ["gmsh22"]}).status == "tested"
    for fields in ({"mfem_config": {}}, {"mfem_git": "other"}, {"mfem_version": "MFEM v4.9"}):
        assert capability.evaluate(ExecutionSpec(), ENV_SERIAL, catalog={**catalog, **fields}).code == "NOT_VERIFIED_BUILD"
