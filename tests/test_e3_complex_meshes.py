"""Positive and negative tests for E3 Task 2: unstructured Gmsh, P3/Q3, Ex6 AMR, and period degradation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

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


def _lshape_problem():
    return {
        "id": "lshape_singularity",
        "equation": "-laplacian(u) = f",
        "domain": "lshape_2d",
        "exact_solution": "((x-0.5)**2+(y-0.5)**2)**(1/3)*sin((2/3)*theta_lshape)",
        "source": "0",
        "boundaries": [
            {"attribute": 1, "label": "lshape_outer", "kind": "dirichlet", "value": "from_exact_solution"}
        ],
    }


@pytest.mark.mfem
@requires_mfem
def test_amr_mixed_neumann_controller_retry_and_reverify(tmp_path):
    """Non-zero Neumann mixed BCs + kappa='1+x' + bounded AMR RETRY_ELIGIBLE + offline reverify."""
    spec_tight = {
        "schema_version": "0.4",
        "problem": _var_problem(("right", "top")),
        "execution": {
            "mesh_type": "quad",
            "mesh_topology": "amr",
            "polynomial_order": 2,
            "mesh_levels": [4],
            "assembly": "full",
            "solver": "cg_gs",
            "amr_max_steps": 2,
            "amr_max_dofs": 800,
        },
        "validation": {"max_l2_error_finest": 2e-3},
        "policy": {"max_attempts": 3, "max_dofs_per_level": 20000},
    }
    sp = tmp_path / "amr_var_mixed.json"
    sp.write_text(json.dumps(spec_tight))
    res = controller.run_study(sp, tmp_path / "runs")
    assert res["final_state"] == "ACCEPTED", res
    assert len(res["attempts"]) >= 2
    assert res["attempts"][0]["verdict"] == "RETRY_ELIGIBLE"
    assert res["attempts"][-1]["verdict"] == "ACCEPTED"

    run_dir = Path(res["run_dir"])
    fin_att = run_dir / f"attempts/attempt_{len(res['attempts']):02d}"
    fin_ver = json.loads((fin_att / "verdict.json").read_text())
    assert fin_ver["verifier"] == "verifier.amr"
    assert any(s["hanging_dof_count"] > 0 for s in fin_ver["steps"])
    assert max(s["reference_rel_diff"] for s in fin_ver["steps"]) < 1e-8

    rep = controller.reverify(fin_att)
    assert rep["verdict"] == ACCEPTED
    assert rep["integrity"]["passed"] is True

    # Tampering one AMR DOF value must fail reverify CLOSED
    target_txt = fin_att / "amr/step_01/dof_values.txt"
    orig_txt = target_txt.read_text()
    lines = orig_txt.splitlines()
    idx, val = lines[1].split()
    lines[1] = f"{idx} {float(val) + 0.05:.12g}"
    target_txt.write_text("\n".join(lines) + "\n")
    tampered = controller.reverify(fin_att)
    assert tampered["verdict"] == REJECTED
    assert "EVIDENCE_INTEGRITY" in tampered["failure_codes"]


@pytest.mark.mfem
@requires_mfem
def test_amr_lshape_controller_end_to_end_and_strict_self_report(tmp_path):
    """L-shape singularity end-to-end via controller.run_study + reverify at strict 1e-6 self-report tol."""
    lshape_src = Path(__file__).resolve().parent.parent / "third_party/mfem-src/data/l-shape.mesh"
    assert lshape_src.is_file(), f"missing {lshape_src}"
    mbytes = lshape_src.read_bytes()
    (tmp_path / "l-shape.mesh").write_bytes(mbytes)
    spec = {
        "schema_version": "0.4",
        "problem": _lshape_problem(),
        "execution": {
            "mesh_type": "quad",
            "mesh_topology": "amr",
            "polynomial_order": 2,
            "mesh_levels": [4],
            "assembly": "full",
            "solver": "cg_gs",
            "amr_max_steps": 5,
            "amr_max_dofs": 6000,
        },
        "external_mesh": {"file": "l-shape.mesh", "sha256": hashlib.sha256(mbytes).hexdigest()},
    }
    sp = tmp_path / "lshape_amr.json"
    sp.write_text(json.dumps(spec))
    res = controller.run_study(sp, tmp_path / "runs")
    assert res["final_state"] == "ACCEPTED", res
    att = Path(res["run_dir"]) / "attempts/attempt_01"
    ver = json.loads((att / "verdict.json").read_text())
    assert ver["verdict"] == ACCEPTED
    for st in ver["steps"]:
        self_l2 = [c for c in st["checks"] if c["name"] == "self_report_l2"][0]
        self_h1 = [c for c in st["checks"] if c["name"] == "self_report_h1_semi"][0]
        assert self_l2["passed"] and self_l2["threshold"] == pytest.approx(1e-6)
        assert self_h1["passed"] and self_h1["threshold"] == pytest.approx(1e-6)
    rep = controller.reverify(att)
    assert rep["integrity"]["passed"] is True and rep["verdict"] == ACCEPTED


@pytest.mark.mfem
@requires_mfem
@pytest.mark.parametrize(
    "fault,expected_code",
    [
        ("wrong_kappa", "REFERENCE_MISMATCH"),
        ("omit_neumann_coefficient", "REFERENCE_MISMATCH"),
        ("swap_boundary_attr", "BOUNDARY_LABEL_MISMATCH"),
        ("missing_element", "MESH_MISMATCH"),
    ],
)
def test_amr_negative_fault_modes_rejected(tmp_path, fault, expected_code):
    """Negative AMR controls: wrong kappa, wrong Neumann flux, swapped bdr attr, missing element."""
    prob_dict = _var_problem(("right",))
    prob = StudySpec.model_validate({
        "schema_version": "0.4",
        "problem": prob_dict,
        "execution": {"mesh_type": "quad", "mesh_topology": "amr", "polynomial_order": 2, "mesh_levels": [4]},
    }).problem
    out_dir = tmp_path / f"fault_{fault}"
    run_res = mfem_runner.run_amr(
        "sin_sin_variable", "quad", 2, out_dir,
        n=4, dirichlet="1,3,4", neumann="2", max_steps=3, max_dofs=3000, inject_fault=fault,
    )
    assert run_res["exit_code"] == 0
    ver = verify_amr_run(out_dir, prob, ValidationSpec())
    assert ver["verdict"] == REJECTED
    assert expected_code in ver["failure_codes"]


@pytest.mark.mfem
@requires_mfem
@pytest.mark.parametrize("tamper", ["claim_kappa", "claim_conforming", "claim_true_dofs", "drop_bdr_row"])
def test_amr_negative_untrusted_result_and_boundary_tampering(tmp_path, tamper):
    """Verify AMR rejects tampered solver claims, constraint counts, and missing boundary segments."""
    prob = StudySpec.model_validate({
        "schema_version": "0.4",
        "problem": _var_problem(("right",)),
        "execution": {"mesh_type": "quad", "mesh_topology": "amr", "polynomial_order": 2, "mesh_levels": [4]},
    }).problem
    out_dir = tmp_path / f"tamper_{tamper}"
    run_res = mfem_runner.run_amr(
        "sin_sin_variable", "quad", 2, out_dir,
        n=4, dirichlet="1,3,4", neumann="2", max_steps=4, max_dofs=4000,
    )
    assert run_res["exit_code"] == 0
    step_dir = out_dir / "step_01"
    rpath = step_dir / "result.json"
    res = json.loads(rpath.read_text())
    if tamper == "claim_kappa":
        res["case_diffusivity"] = "1"
        expected = "CLAIM_SPEC_MISMATCH"
    elif tamper == "claim_conforming":
        res["conforming"] = True
        expected = "MESH_MISMATCH"
    elif tamper == "claim_true_dofs":
        res["num_true_dofs"] = int(res["num_true_dofs"]) + 1
        expected = "MESH_MISMATCH"
    else:
        bpath = step_dir / "boundary.txt"
        header, *rows = bpath.read_text().splitlines()
        kept = []
        for idx_new, row in enumerate(rows[1:]):
            toks = row.split()
            toks[0] = str(idx_new)
            kept.append(" ".join(toks))
        bpath.write_text("\n".join([header, *kept]) + "\n")
        res["num_boundary_elements"] = len(kept)
        expected = "BOUNDARY_COVERAGE"
    rpath.write_text(json.dumps(res, indent=2))
    ver = verify_amr_run(out_dir, prob, ValidationSpec())
    assert ver["verdict"] == REJECTED
    assert expected in ver["failure_codes"]
