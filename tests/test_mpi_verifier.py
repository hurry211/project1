"""Rank ownership, ordering, numerical agreement and native stopping records."""

import json
import math

import numpy as np
import pytest

from sciagent.spec import StudySpec
from synthetic import make_attempt
from verifier.fem import ArtifactError, load_mesh_data
from verifier.mpi import LEVELS, _native_quadrature, verify_mpi_level, verify_mpi_orders


@pytest.fixture
def exported(tmp_path, specs_dir):
    # An independently assembled (Duffy quadrature) field, differing slightly
    # from the symmetric degree-six load used in MFEM's algebraic residual.
    request = json.loads((specs_dir / "poisson_sin_sin_p1_tri.json").read_text())
    request["execution"]["mesh_levels"] = [8]
    study = StudySpec.model_validate(request)
    serial = make_attempt(tmp_path, study, shuffle=True) / "levels/n0008"
    mesh = load_mesh_data(serial)
    result = json.loads((serial / "result.json").read_text())
    mpi = tmp_path / "mpi"
    mpi.mkdir()
    # Serial local and global DOF numbering is shuffled. The independent
    # MPI export uses the cyclic triangle order and reverses element order.
    local = np.linalg.norm(mesh.elem_node_xy[:,:,None,:] - mesh.verts[:,None,:,:], axis=3).argmin(axis=1)
    points = np.concatenate((mesh.verts, np.take_along_axis(mesh.values[mesh.elem_dofs], local, axis=1)[:,:,None]), axis=2)
    owners = mesh.verts[:,:,0].mean(axis=1) >= 0.5
    boundaries = np.c_[mesh.bdr_attr, mesh.bdr_xy.reshape(-1,4)]
    for rank in range(2):
        np.savetxt(mpi / f"cells.{rank}.tsv", np.roll(points[owners==rank][::-1], 1, axis=1).reshape(-1,9))
        np.savetxt(mpi / f"boundary.{rank}.tsv", boundaries[(mesh.bdr_xy[:,:,0].mean(axis=1)>=0.5) == rank])
    native = {"case": "sin_sin", "solver": "cg_boomeramg", "n": 8, "ranks": 2, "order": 1,
              "global_dofs": 81, "global_elements": 128, "rhs_quadrature_order": 6, "error_quadrature_order": 6,
              "max_iter": 20000, "abs_tol": 0, "rel_tol": 1e-12, "solver_converged": True, "solver_iterations": 8,
              "solver_initial_norm": 1., "solver_final_norm": 1e-13, "setup_seconds_max": .025, "solve_seconds_max": .01,
              "l2_error": result["self_reported_l2_error"], "h1_semi_error": result["self_reported_h1_semi_error"],
              # Degree-six RHS and its mismatch with the degree-nine Duffy solution at n=8.
              "rhs_l2": 1.202452500953836, "residual_l2": 9.261e-11}
    (mpi / "result.json").write_text(json.dumps(native))
    return mpi, serial


def verify(exported):
    return verify_mpi_level(*exported, n=8, ranks=2)


def test_rank_union_recovers_renumbered_serial_field(exported):
    out = verify(exported)
    assert out["rank_elements"] == [64, 64]
    assert out["serial_relative_field_difference"] < 1e-14
    assert out["reference_relative_field_difference"] < 1e-12
    assert out["l2_error"] == pytest.approx(0.02113277, rel=1e-6)


def test_overlap_with_missing_cell(exported):
    mpi, _ = exported
    path = mpi / "cells.0.tsv"
    data = np.loadtxt(path)
    # Both rows are interior, so the boundary union remains intact.
    interior = np.where(np.all((data.reshape(-1,3,3)[:,:,:2] > 0) &
                               (data.reshape(-1,3,3)[:,:,:2] < 1), axis=(1,2)))[0]
    data[interior[0]] = data[interior[1]]
    np.savetxt(path, data)
    with pytest.raises(ArtifactError, match="multiply owned"):
        verify(exported)


def test_oversized_rank_artifact_rejected_before_open(exported, monkeypatch):
    from pathlib import Path
    mpi, _ = exported
    path = mpi / "cells.1.tsv"
    with path.open("ab") as stream:
        stream.truncate(1 << 20)
    original = Path.open

    def guard(file, *args, **kwargs):
        assert file != path, "verifier opened a file already known to exceed the grid bound"
        return original(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    with pytest.raises(ArtifactError, match="oversized"):
        verify(exported)


@pytest.mark.parametrize("failure,reason", [
    ("duplicate rank file", "rank exports"), ("missing rank file", "rank exports"),
    ("boundary owner", "boundary owner"), ("label", "wrong boundary"), ("missing boundary", "boundary"),
    ("coordinate", "prescribed grid"), ("reverse", "reversed"), ("nan", "finite"),
    ("shared", "inconsistent shared"),
])
def test_rank_export_faults(exported, failure, reason):
    mpi, _ = exported
    path = mpi / "cells.0.tsv"
    if failure == "duplicate rank file":
        (mpi / "cells.2.tsv").write_bytes(path.read_bytes())
    elif failure == "missing rank file":
        path.unlink()
    elif failure == "boundary owner":
        a, b = mpi / "boundary.0.tsv", mpi / "boundary.1.tsv"
        left, right = a.read_bytes(), b.read_bytes()
        a.write_bytes(right)
        b.write_bytes(left)
    elif failure in ("label", "missing boundary"):
        boundary = mpi / "boundary.1.tsv"
        rows = np.loadtxt(boundary)
        rows[0,0] = 42
        np.savetxt(boundary, rows if failure == "label" else rows[1:])
    else:
        data = np.loadtxt(path).reshape(-1,3,3)
        if failure == "coordinate":
            data[0,0,0] += 1e-5
        elif failure == "reverse":
            data[0] = data[0,::-1]
        elif failure == "nan":
            data[0,0,2] = np.nan
        else:
            # Alter a node on the process interface, on every rank-0 copy.
            mask = (data[:,:,0] == .5) & (data[:,:,1] == .5)
            data[mask,2] += 1e-6
        np.savetxt(path, data.reshape(-1,9))
    with pytest.raises(ArtifactError, match=reason):
        verify(exported)


@pytest.mark.parametrize("change,reason", [
    ({"solver_converged": 1}, "configuration"), ({"solver_iterations": 0}, "stopping"),
    ({"solver_final_norm": 1e-5}, "stopping"), ({"rel_tol": 1e-5}, "tolerance"),
    ({"order": 1.0}, "configuration"), ({"rhs_quadrature_order": 4}, "configuration"),
    ({"l2_error": 1e-5}, "error norms"), ({"residual_l2": 1e-6}, "residual norm"),
    ({"rhs_l2": 2}, "RHS norm"), ({"setup_seconds_max": -1}, "numeric"),
])
def test_native_claims(exported, change, reason):
    mpi, _ = exported
    path = mpi / "result.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(ArtifactError, match=reason):
        verify(exported)


def test_damaged_serial_and_mpi_fields_do_not_certify_each_other(exported):
    mpi, serial = exported
    mesh = load_mesh_data(serial)
    mask = np.all(mesh.elem_node_xy == (.375, .375), axis=2)
    dof, = np.unique(mesh.elem_dofs[mask])
    nodal = np.loadtxt(serial / "dof_values.txt")
    nodal[dof, 1] += 1e-3
    np.savetxt(serial / "dof_values.txt", nodal)
    for path in mpi.glob("cells.*.tsv"):
        values = np.loadtxt(path).reshape(-1,3,3)
        values[np.all(values[:,:,:2] == (.375, .375), axis=2), 2] += 1e-3
        np.savetxt(path, values.reshape(-1,9))
    with pytest.raises(ArtifactError, match="SciPy reference"):
        verify(exported)


def test_native_load_rule_integrates_all_degree_six_moments():
    points, weights = _native_quadrature()
    for i in range(7):
        for j in range(7-i):
            integ = weights @ (points[:,0]**i * points[:,1]**j)
            assert integ == pytest.approx(math.factorial(i)*math.factorial(j)/math.factorial(i+j+2), abs=1e-15)


def test_convergence_gate_needs_all_screened_levels_in_order():
    levels = [{"n": n, "ranks": 2, "l2_error": n**-2, "h1_semi_error": 1/n} for n in LEVELS]
    assert verify_mpi_orders(levels) == {"finest_l2_order": 2.0, "finest_h1_order": 1.0}
    for bad in (levels[:-1], levels[::-1], [*levels[:-1], {**levels[-1], "ranks": 4}],
                [*levels[:-1], {**levels[-1], "l2_error": levels[-2]["l2_error"]*.99}]):
        with pytest.raises(ArtifactError):
            verify_mpi_orders(bad)
