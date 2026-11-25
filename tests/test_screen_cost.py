import json

import pytest

from benchmarks.screen_cost import screen
from sciagent.spec import ExecutionSpec


@pytest.mark.parametrize("changes,expected", [
    ({}, []),
    ({"solver_initial_norm": 1.0, "solver_final_norm": 0.5}, ["CG_RECORD_INCONSISTENT"]),
    ({"rhs_quadrature_order": 1}, ["QUADRATURE_CLAIM_MISMATCH"]),
    ({"producer": "unknown", "solver_iterations": 999999}, []),
])
def test_only_a_failed_native_necessary_condition_can_screen_out(tmp_path, changes, expected):
    ex = ExecutionSpec(mesh_levels=[8])
    folder = tmp_path / "levels/n0008"
    folder.mkdir(parents=True)
    record = {"producer": "poisson_mms", "rhs_quadrature_order": 6, "error_quadrature_order": 6,
              "solver_converged": True, "solver_initial_norm": 1.0, "solver_final_norm": 1e-15,
              "solver_iterations": 10, "abs_tol": 0.0, **changes}
    (folder / "result.json").write_text(json.dumps(record))
    assert screen(tmp_path, ex) == expected
