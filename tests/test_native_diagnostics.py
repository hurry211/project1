"""Native Poisson diagnostic contracts and independent boundary-flux checks."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, requires_mfem
from sciagent import controller
from sciagent.environment import mfem_install_dir
from sciagent.evidence import sha256_file
from sciagent.mfem_runner import solver_binary
from verifier.fem import load_mesh_data

pytestmark = [pytest.mark.mfem, requires_mfem]


def run_level(directory, *args, expected_exit=0):
    directory.mkdir()
    proc = subprocess.run([str(solver_binary()), *args, "--out", str(directory)],
                          capture_output=True, text=True)
    assert proc.returncode == expected_exit, proc.stderr
    result = json.loads((directory / "result.json").read_text())
    assert result["diagnostics"]["schema_version"] == 1
    return result


def test_sine_flux_integral_converges_to_eight(tmp_path):
    # Reference flux is +2 on each side, even on a side treated as Neumann.
    totals = []
    for n in (8, 16):
        result = run_level(tmp_path / str(n), "--n", str(n), "--dirichlet", "1,2,4", "--neumann", "3")
        diagnostics = result["diagnostics"]
        flux = diagnostics["boundary_flux"]
        by_side = flux["outward_by_side"]
        assert set(by_side) == {"left", "right", "top", "bottom"}
        for value in by_side.values():
            assert value == pytest.approx(2.0, abs=0.25)
        assert flux["source_integral"] == pytest.approx(8.0, abs=1e-7)
        assert flux["outward_total"] == pytest.approx(sum(by_side.values()), abs=1e-12)
        assert flux["signed_imbalance"] == pytest.approx(flux["outward_total"] - flux["source_integral"])
        totals.append(flux["outward_total"])
    assert abs(8.0 - totals[1]) < abs(8.0 - totals[0]) / 2


def tensor_quad_flux(directory, n, order, diffusivity=lambda x: 1.0):
    """Integrate the flux of the exported tensor field for linear or constant kappa(x)."""
    mesh = load_mesh_data(directory)
    grid = np.full((n * order + 1, n * order + 1), np.nan)
    ix, iy = np.rint(mesh.elem_node_xy * n * order).astype(int).transpose(2, 0, 1)
    grid[ix, iy] = mesh.values[mesh.elem_dofs]
    # The first axis of each view goes inward. Flux = kappa * du/d(inward).
    sides = {"left": grid, "right": grid[::-1], "bottom": grid.T, "top": grid.T[::-1]}
    x_nodes = np.linspace(0, 1, 2 * n + 1)
    side_x = {"left": 0.0, "right": 1.0, "bottom": x_nodes, "top": x_nodes}
    integrals = {}
    for side, u in sides.items():
        if order == 1:
            endpoint = n * (u[1] - u[0])
            derivative = np.empty(2 * n + 1)
            derivative[::2] = endpoint
            derivative[1::2] = (endpoint[1:] + endpoint[:-1]) / 2
        else:
            derivative = n * (-3 * u[0] + 4 * u[1] - u[2])
        # Simpson integrates the linear coefficient times the Q1/Q2 trace exactly.
        flux = derivative * diffusivity(side_x[side])
        integrals[side] = (flux[:-1:2] + 4 * flux[1::2] + flux[2::2]).sum() / (6 * n)
    return integrals


@pytest.mark.parametrize("order,assembly,solver", [(1, "full", "cg_gs"), (2, "partial", "cg_jacobi")])
def test_mixed_quad_flux_matches_independent_face_integrals(tmp_path, order, assembly, solver):
    n = 8
    directory = tmp_path / "level"
    result = run_level(directory, "--case", "cos_exp", "--mesh-type", "quad", "--n", str(n),
                       "--order", str(order), "--assembly", assembly, "--solver", solver,
                       "--dirichlet", "1,2,4", "--neumann", "3")
    diagnostics = result["diagnostics"]
    flux = diagnostics["boundary_flux"]
    by_side = flux["outward_by_side"]
    assert by_side == pytest.approx(tensor_quad_flux(directory, n, order), abs=1e-11)
    # cos(pi*x)*exp(y) changes sign: individual discrete fluxes can cancel.
    assert abs(by_side["left"]) > 1e-4
    assert flux["outward_total"] == pytest.approx(sum(by_side.values()), abs=1e-12)
    assert abs(flux["source_integral"]) < 1e-12
    assert abs(flux["outward_total"]) < 1e-10
    assert flux["signed_imbalance"] == pytest.approx(flux["outward_total"] - flux["source_integral"], abs=1e-12)
    residual = diagnostics["algebraic_residual"]
    assert residual["rhs_l2"] > 0
    assert residual["relative_l2"] == pytest.approx(residual["absolute_l2"] / residual["rhs_l2"], abs=1e-25)
    assert 0 <= residual["relative_l2"] < 1e-10


@pytest.mark.parametrize("order,assembly,solver", [(1, "full", "cg_gs"), (2, "partial", "cg_jacobi")])
def test_variable_flux_matches_weighted_exported_gradient(tmp_path, order, assembly, solver):
    n = 16
    directory = tmp_path / "level"
    native = run_level(directory, "--case", "sin_sin_variable", "--mesh-type", "quad",
                       "--n", str(n), "--order", str(order), "--assembly", assembly, "--solver", solver,
                       "--dirichlet", "1,2,4", "--neumann", "3")
    by_side = native["diagnostics"]["boundary_flux"]["outward_by_side"]
    discrete = tensor_quad_flux(directory, n, order, diffusivity=lambda x: 1 + x)
    assert by_side == pytest.approx(discrete, abs=1e-11)
    # The exact outward diffusive flux integrates to +6 at the bottom and -6 at the top.
    assert by_side["bottom"] == pytest.approx(6.0, abs=0.3)
    assert by_side["top"] == pytest.approx(-6.0, abs=0.3)


def test_wrong_source_and_early_stop_have_distinct_residuals(tmp_path):
    wrong = run_level(tmp_path / "wrong", "--inject-fault", "wrong_source")
    stopped = run_level(tmp_path / "stopped", "--inject-fault", "early_stop_lie")
    assert wrong["solver_converged"] and stopped["solver_converged"]
    assert wrong["diagnostics"]["algebraic_residual"]["relative_l2"] < 1e-10
    assert wrong["diagnostics"]["boundary_flux"]["source_integral"] == pytest.approx(1.1 * 8.0, abs=1e-7)
    residual = stopped["diagnostics"]["algebraic_residual"]["relative_l2"]
    assert 1e-3 < residual < 0.1
    reported_cg_ratio = stopped["solver_final_norm"] / stopped["solver_initial_norm"]
    assert residual != pytest.approx(reported_cg_ratio, rel=1e-2)


def test_nonfinite_export_keeps_residual_and_marks_flux_unavailable(tmp_path):
    result = run_level(tmp_path / "bad", "--inject-fault", "nan_value", expected_exit=6)
    diagnostics = result["diagnostics"]
    flux = diagnostics["boundary_flux"]
    assert diagnostics["algebraic_residual"]["relative_l2"] < 1e-10
    assert flux["source_integral"] == pytest.approx(8.0, abs=1e-7)
    assert flux["outward_by_side"] == dict.fromkeys(("left", "right", "top", "bottom"))
    assert flux["outward_total"] is None and flux["signed_imbalance"] is None


def test_replay_of_native_result_without_optional_diagnostics(specs_dir, tmp_path, monkeypatch):
    native_run = controller.mfem_runner.run_level

    def write_previous_result_format(*args, **kwargs):
        record = native_run(*args, **kwargs)
        path = Path(record.level_dir) / "result.json"
        data = json.loads(path.read_text())
        del data["diagnostics"]
        path.write_text(json.dumps(data))
        return record

    # Present the older output format before the controller verifies and seals it.
    monkeypatch.setattr(controller.mfem_runner, "run_level", write_previous_result_format)
    manifest = controller.run_study(specs_dir / "poisson_sin_sin_p1_tri.json", tmp_path / "runs")
    assert manifest["final_state"] == "ACCEPTED"
    run = Path(manifest["run_dir"])
    replay = controller.reverify(run / "attempts/attempt_01", sha256_file(run / "manifest.json"))
    assert replay["verdict"] == "ACCEPTED" and replay["integrity"]["passed"]


def test_residual_normalization_with_zero_and_nonzero_rhs(tmp_path):
    # The registered MMS cases have nonzero discrete RHSs. Use a known
    # operator and retain a measurable absolute residual for the zero-RHS case.
    build = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(ROOT / "tests/native"), "-B", str(build),
                    f"-DMFEM_DIR={mfem_install_dir() / 'lib/cmake/mfem'}"], check=True)
    subprocess.run(["cmake", "--build", str(build)], check=True)
    subprocess.run([str(build / "residual_contract")], check=True)
