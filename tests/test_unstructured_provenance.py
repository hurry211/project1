"""Bind native cells and their refinement depth to the frozen Gmsh geometry."""

import numpy as np
import pytest

from benchmarks.e3_complex_meshes import _problem_var
from conftest import requires_mfem
from sciagent.gmsh import generate_unstructured_msh, inspect_gmsh_unstructured
from sciagent.mfem_runner import run_level
from sciagent.spec import ExecutionSpec, ValidationSpec
from verifier.fem import load_mesh_data
from verifier.verifier import GEOM_TOL, _matches_unstructured_source, verify_level


@pytest.mark.parametrize("n", [2, 3, 5, 6, 12])
def test_reject_rounded_or_smaller_refinement_label(n):
    source = generate_unstructured_msh("tri", nx=4, seed=121)
    execution = ExecutionSpec(mesh_type="tri", mesh_topology="unstructured", mesh_levels=[n])
    with pytest.raises(ValueError, match="not a uniform refinement"):
        inspect_gmsh_unstructured(source, _problem_var(), execution)


@pytest.mark.parametrize("geom", ["tri", "quad"])
def test_cyclic_reordering_roundoff_and_one_to_one_ownership(geom):
    execution = ExecutionSpec(mesh_type=geom, mesh_topology="unstructured", mesh_levels=[4])
    raw = generate_unstructured_msh(geom, nx=4, seed=29)
    grid = inspect_gmsh_unstructured(raw, _problem_var(), execution)
    corners = np.roll(np.asarray(grid.coarse_cells)[::-1].copy(), 2, axis=1)
    corners[0, 1, 0] += GEOM_TOL / 2
    assert _matches_unstructured_source(corners, grid, 0)
    corners[0, 1, 0] += 2 * GEOM_TOL
    assert not _matches_unstructured_source(corners, grid, 0)
    corners[0] = corners[1]
    assert not _matches_unstructured_source(corners, grid, 0)


@pytest.mark.mfem
@requires_mfem
@pytest.mark.parametrize("geom", ["tri", "quad"])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_independent_native_solve_on_alternate_same_size_mesh_fails_provenance(tmp_path, geom, depth):
    problem = _problem_var()
    n = 4 * 2**depth
    execution = ExecutionSpec(mesh_type=geom, polynomial_order=2, mesh_topology="unstructured", mesh_levels=[n])
    raw = generate_unstructured_msh(geom, nx=4, seed=29)
    actual = inspect_gmsh_unstructured(raw, problem, execution)
    other_raw = generate_unstructured_msh(geom, nx=4, seed=31)
    other = inspect_gmsh_unstructured(other_raw, problem, execution)
    assert actual.element_count == other.element_count and abs(actual.total_area - other.total_area) < GEOM_TOL
    path = tmp_path / "actual.msh"
    path.write_bytes(raw)
    level = tmp_path / "native"
    record = run_level(problem, execution, "sin_sin_variable", n, level, 60, 1024, mesh_path=path, refinements=depth)
    assert record.exit_code == 0

    def check(binding):
        return verify_level(level, problem, execution, ValidationSpec(), n, external_grid=binding,
                            require_run_record=True)

    report = check(actual)
    assert report.ok, [c for c in report.checks if not c.passed]
    assert [c.name for c in check(other).checks if not c.passed] == ["external_mesh_refinement"]
    assert [c.name for c in check(None).checks if not c.passed] == ["external_mesh_refinement"]
    vertices = load_mesh_data(level).verts
    assert _matches_unstructured_source(np.roll(vertices[::-1], 1, axis=1), actual, depth)
