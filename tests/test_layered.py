from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from sciagent.spec import StudySpec, load_study
from synthetic import write_level
from verifier import layered
from verifier.fem import ArtifactError, build_local_basis, element_quadrature, load_mesh_data


def test_translated_quadrature_matches_elementwise_geometry(tmp_path):
    write_level(tmp_path / "mesh", "quad", 4, 2)
    mesh = load_mesh_data(tmp_path / "mesh")
    basis = build_local_basis(mesh, 2)
    assert basis.translated
    fast = element_quadrature(mesh, basis, 6)
    generic = element_quadrature(mesh, replace(basis, translated=False), 6)
    assert fast.gphi.strides[0] == 0 and generic.gphi.strides[0] != 0
    for key in ("X", "detJ", "wdet", "phi", "gphi"):
        np.testing.assert_allclose(getattr(fast, key), getattr(generic, key), rtol=1e-13, atol=1e-13)
    mesh.elem_node_xy[-1, [0, 1]] = mesh.elem_node_xy[-1, [1, 0]]
    changed = build_local_basis(mesh, 2)
    assert not changed.translated
    np.testing.assert_allclose(changed.coeffs[-1, :, 0], basis.coeffs[-1, :, 1], atol=1e-13)


def test_interface_traces_from_shuffled_corners_and_opposing_perturbations(tmp_path):
    n = 8
    write_level(tmp_path / "mesh", "quad", n, 1, np.random.default_rng(8))
    mesh = load_mesh_data(tmp_path / "mesh")
    xy = np.empty((len(mesh.values), 2))
    xy[mesh.elem_dofs] = mesh.elem_node_xy
    mesh.values[:] = layered.field(xy[:, 0])
    clean = layered.diagnostics(mesh, n)
    assert clean["energy"] == pytest.approx(layered.FLUX, abs=1e-13)
    assert clean["jump_rel"] < 1e-10
    # A zero-mean alternating interface disturbance has nonzero endpoint flux jumps.
    face = xy[:, 0] == 0.5
    mesh.values[face] += 1e-7 * (-1.)**np.rint(n * xy[face, 1])
    broken = layered.diagnostics(mesh, n)
    assert broken["jump_rel"] > 1e-3 and broken["flux_rel"] > 1e-3
    mesh.elem_node_xy[0, 0] = mesh.elem_node_xy[0, 1]
    with pytest.raises(ArtifactError, match="distinct"):
        layered.diagnostics(mesh, n)


@pytest.mark.parametrize("update", [{"layered_q1": None}, {"require_monotone_decrease": True},
                                    {"require_reference_solve": False}, {"layered_q1": {"flux_rel_tol": 0.1}}])
def test_layered_validation_contract_is_enforced(specs_dir, update):
    spec = load_study(specs_dir / "extensions/layered_unit_drop_10000.json").model_dump()
    spec["validation"].update(update)
    with pytest.raises(ValidationError):
        StudySpec.model_validate(spec)
