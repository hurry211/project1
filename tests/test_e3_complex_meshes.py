"""Positive and negative tests for E3 Task 2: unstructured Gmsh, P3/Q3, Ex6 AMR, and period degradation."""

from __future__ import annotations

import hashlib
import numpy as np
import pytest

from conftest import requires_mfem
from sciagent import controller, mfem_runner
from sciagent.gmsh import generate_unstructured_msh, inspect_gmsh_unstructured
from sciagent.spec import StudySpec, ValidationSpec
from verifier.amr import verify_amr_run, verify_amr_step
from verifier.fem import load_mesh_data, repeated_element_period
from verifier.verifier import ACCEPTED, REJECTED, verify_attempt


def _var_problem(neumann=("top",)):
    sides = [(1, "bottom"), (2, "right"), (3, "top"), (4, "left")]
    return {
        "id": "sin_sin_variable",
        "equation": "-div(kappa*grad(u)) = f",
        "domain": "unit_square_2d",
        "kappa": "1+x",
        "exact_solution": "sin(pi*x)*sin(2*pi*y)",
        "source": "5*pi**2*(1+x)*sin(pi*x)*sin(2*pi*y)-pi*cos(pi*x)*sin(2*pi*y)",
        "boundaries": [
            {"attribute": a, "label": lbl, "kind": "neumann" if lbl in neumann else "dirichlet", "value": "from_exact_solution"}
            for a, lbl in sides
        ],
    }


def _steep_problem():
    sides = [(1, "bottom"), (2, "right"), (3, "top"), (4, "left")]
    return {
        "id": "gaussian_steep",
        "equation": "-laplacian(u) = f",
        "domain": "unit_square_2d",
        "exact_solution": "exp(-120*((x-0.375)**2+(y-0.625)**2))",
        "source": "(480-57600*((x-0.375)**2+(y-0.625)**2))*exp(-120*((x-0.375)**2+(y-0.625)**2))",
        "boundaries": [{"attribute": a, "label": lbl, "kind": "dirichlet", "value": "from_exact_solution"} for a, lbl in sides],
    }


@pytest.mark.parametrize("geom", ["tri", "quad"])
def test_unstructured_gmsh_inspector_accepts_valid_and_detects_period_degradation(geom):
    prob = StudySpec.model_validate({
        "schema_version": "0.3",
        "problem": _var_problem(),
        "execution": {"mesh_type": geom, "mesh_topology": "unstructured", "polynomial_order": 3, "mesh_levels": [6, 12]},
        "external_mesh": {"file": "mesh.msh", "sha256": "0" * 64},
    })
    raw = generate_unstructured_msh(geom, nx=6, seed=11)
    grid = inspect_gmsh_unstructured(raw, prob.problem, prob.execution)
    assert grid.geom == geom
    assert grid.total_area == pytest.approx(1.0, abs=1e-11)
    assert grid.element_count == (72 if geom == "tri" else 36)


@pytest.mark.parametrize("corruption", ["inverted_element", "boundary_gap", "wrong_boundary_label"])
def test_unstructured_gmsh_rejects_corrupted_topology(corruption):
    raw = generate_unstructured_msh("tri", nx=4, seed=5).decode("ascii")
    lines = raw.splitlines()
    elem_start = lines.index("$Elements") + 2
    elem_end = lines.index("$EndElements")
    prob = StudySpec.model_validate({
        "schema_version": "0.3",
        "problem": _var_problem(),
        "execution": {"mesh_type": "tri", "mesh_topology": "unstructured", "polynomial_order": 1, "mesh_levels": [4]},
        "external_mesh": {"file": "mesh.msh", "sha256": "0" * 64},
    })
    if corruption == "inverted_element":
        # Flip winding of the first triangle element -> negative Jacobian / orientation
        for i in range(elem_start, elem_end):
            toks = lines[i].split()
            if toks[1] == "2":
                toks[5], toks[6] = toks[6], toks[5]
                lines[i] = " ".join(toks)
                break
    elif corruption == "boundary_gap":
        # Remove the last boundary line edge AND decrement element count
        lines[elem_start - 1] = str(int(lines[elem_start - 1]) - 1)
        del lines[elem_start]
    elif corruption == "wrong_boundary_label":
        # Change bottom physical tag 1 to top tag 3 on the first line segment
        toks = lines[elem_start].split()
        toks[3] = "3"
        lines[elem_start] = " ".join(toks)
    with pytest.raises(ValueError):
        inspect_gmsh_unstructured("\n".join(lines).encode("ascii"), prob.problem, prob.execution)


def test_repeated_element_period_degrades_cleanly_on_unstructured_pattern():
    # Periodic 2-triangle affine edge vectors -> period == 2
    rng = np.random.default_rng(7)
    base = np.array([[[1.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [0.0, -1.0]]])
    periodic = np.tile(base, (50, 1, 1))
    assert repeated_element_period(periodic) == 2

    # Break periodicity on the very last element -> must scan accurately and degrade to period == E
    almost = periodic.copy()
    almost[-1, 0, 0] += 1e-6
    assert repeated_element_period(almost) == almost.shape[0]


@pytest.mark.mfem
@requires_mfem
def test_unstructured_p3_controller_end_to_end_and_wrong_source_rejection(tmp_path):
    msh = generate_unstructured_msh("tri", nx=8, seed=19, graded_focus=(0.375, 0.625))
    msh_path = tmp_path / "tri8.msh"
    msh_path.write_bytes(msh)
    sha = hashlib.sha256(msh).hexdigest()

    spec = {
        "schema_version": "0.4",
        "problem": _var_problem(("top",)),
        "execution": {
            "mesh_type": "tri",
            "mesh_topology": "unstructured",
            "polynomial_order": 3,
            "mesh_levels": [8, 16, 32],
            "assembly": "full",
            "solver": "cg_gs",
        },
        "external_mesh": {"file": "tri8.msh", "sha256": sha},
    }
    sp_ok = tmp_path / "ok.json"
    sp_ok.write_text(__import__("json").dumps(spec))
    res = controller.run_study(sp_ok, tmp_path / "run_ok")
    assert res["final_state"] == "ACCEPTED", res

    # Negative test: corrupt source in execution fails verification fail-closed
    spec_bad = __import__("json").loads(sp_ok.read_text())
    spec_bad["execution"]["inject_fault"] = "wrong_source"
    sp_bad = tmp_path / "bad.json"
    sp_bad.write_text(__import__("json").dumps(spec_bad))
    res_bad = controller.run_study(sp_bad, tmp_path / "run_bad")
    assert res_bad["final_state"] == "FAILED"
    assert "REFERENCE_MISMATCH" in res_bad["attempts"][-1]["failure_codes"]


@pytest.mark.mfem
@requires_mfem
def test_ex6_amr_q2_hanging_nodes_and_negative_controls(tmp_path):
    prob = StudySpec.model_validate({
        "schema_version": "0.4",
        "problem": _steep_problem(),
        "execution": {"mesh_type": "quad", "mesh_topology": "amr", "polynomial_order": 2, "mesh_levels": [4]},
    }).problem

    # Positive test: real Ex6 AMR Q2 produces T-junction hanging DOFs verified via C^T K C
    out_ok = tmp_path / "amr_ok"
    run_ok = mfem_runner.run_amr("gaussian_steep", "quad", 2, out_ok, n=4, max_steps=5, max_dofs=4000)
    assert run_ok["exit_code"] == 0
    ver = verify_amr_run(out_ok, prob, ValidationSpec())
    assert ver["verdict"] == ACCEPTED
    assert ver["net_h1_error_decay"] is True
    assert any(s["hanging_dof_count"] > 0 for s in ver["steps"])
    assert max(s["max_hanging_continuity_error"] for s in ver["steps"]) < 1e-11
    assert max(s["reference_rel_diff"] for s in ver["steps"]) < 1e-8

    # Negative test 1: broken T-junction hanging node continuity must fail CLOSED
    out_hang = tmp_path / "amr_broken_hang"
    run_hang = mfem_runner.run_amr(
        "gaussian_steep", "quad", 2, out_hang, n=4, max_steps=5, max_dofs=4000,
        inject_fault="break_hanging_node",
    )
    assert run_hang["exit_code"] == 0
    ver_hang = verify_amr_run(out_hang, prob, ValidationSpec())
    assert ver_hang["verdict"] == REJECTED
    assert "HANGING_NODE_DISCONTINUITY" in ver_hang["failure_codes"]

    # Negative test 2: wrong_source fault in AMR must be rejected by independent C^T K C reference solve
    out_src = tmp_path / "amr_wrong_source"
    run_src = mfem_runner.run_amr(
        "gaussian_steep", "quad", 2, out_src, n=4, max_steps=3, max_dofs=2000,
        inject_fault="wrong_source",
    )
    assert run_src["exit_code"] == 0
    ver_src = verify_amr_run(out_src, prob, ValidationSpec())
    assert ver_src["verdict"] == REJECTED
    assert "REFERENCE_MISMATCH" in ver_src["failure_codes"]
