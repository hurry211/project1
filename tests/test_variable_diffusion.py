"""Independent checks of the divergence operator and its outward flux."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from conftest import requires_mfem
from sciagent import controller
from sciagent.spec import StudySpec, load_study, sha256_of
from synthetic import galerkin_values, make_attempt, write_level
from verifier.fem import load_mesh_data
from verifier.verifier import check_mms_consistency, verify_level


def variable_input(specs_dir):
    return json.loads((specs_dir / "stage2" / "diffusion_mixed_p1_tri.json").read_text())


def test_legacy_canonical_hashes(specs_dir):
    # These digests were captured with the original v0.2 reader.
    study = load_study(specs_dir / "poisson_sin_sin_p1_tri.json")
    assert "kappa" not in study.model_dump()["problem"]
    assert sha256_of(study.problem) == "cc463b70cac2fab9416c28332affa67c3825424f125f1ee662077b439903cebc"
    assert sha256_of(study) == "63b28f4d02857036e4e3d5032706b0a39122505415607c06dee6c9361afbf303"


@pytest.mark.parametrize("coefficient", [None, "1", "-1", "1-2*x", "exp(x)", "x"])
def test_only_reviewed_positive_diffusivity_is_admitted(specs_dir, coefficient):
    raw = variable_input(specs_dir)
    if coefficient is None:
        del raw["problem"]["kappa"]
    else:
        raw["problem"]["kappa"] = coefficient
    with pytest.raises(ValidationError):
        StudySpec.model_validate(raw)


def test_diffusivity_is_bound_to_equation_and_version(specs_dir):
    raw = variable_input(specs_dir)
    raw["schema_version"] = "0.2"
    with pytest.raises(ValidationError, match="schema_version 0.3"):
        StudySpec.model_validate(raw)
    raw["schema_version"] = "0.3"
    raw["problem"]["equation"] = "-laplacian(u) = f"
    with pytest.raises(ValidationError, match="implicit unit diffusivity"):
        StudySpec.model_validate(raw)


def test_mms_detects_missing_product_rule_term(specs_dir):
    raw = variable_input(specs_dir)
    study = StudySpec.model_validate(raw)
    tol = study.validation.mms_consistency_rel_tol
    assert StudySpec.model_validate_json(study.model_dump_json()) == study
    assert check_mms_consistency(study.problem, tol).passed
    raw["problem"]["source"] = "5*pi**2*(1+x)*sin(pi*x)*sin(2*pi*y)"
    check = check_mms_consistency(StudySpec.model_validate(raw).problem, tol)
    assert not check.passed and check.code == "MMS_INCONSISTENT"


@pytest.mark.parametrize("geometry,order,dirichlet_side", [("tri", 1, "left"), ("quad", 2, "top")])
def test_weighted_reference_recovers_exact_linear_field(specs_dir, tmp_path, geometry, order, dirichlet_side):
    # u=x+2y gives f=-1 and exact conormal flux (1+x)*(n_x+2*n_y).
    # A linear field is in both spaces, so its nodal values are an analytic oracle.
    raw = variable_input(specs_dir)
    raw["problem"].update(exact_solution="x+2*y", source="-1")
    for boundary in raw["problem"]["boundaries"]:
        boundary["kind"] = "dirichlet" if boundary["label"] == dirichlet_side else "neumann"
    raw["execution"].update(mesh_type=geometry, polynomial_order=order)
    study = StudySpec.model_validate(raw)
    level = tmp_path / "level"
    write_level(level, geometry, 4, order, rng=np.random.default_rng(7))
    mesh = load_mesh_data(level)
    field = galerkin_values(level, study)
    coords = mesh.elem_node_xy
    np.testing.assert_allclose(field[mesh.elem_dofs], coords[..., 0] + 2 * coords[..., 1], rtol=0, atol=2e-11)


def test_variable_claims_bind_equation_and_coefficient(specs_dir, tmp_path):
    raw = variable_input(specs_dir)
    raw["execution"]["mesh_levels"] = [4]
    study = StudySpec.model_validate(raw)
    level = make_attempt(tmp_path, study) / "levels" / "n0004"
    result_file = level / "result.json"
    original = json.loads(result_file.read_text())

    def claims():
        report = verify_level(level, study.problem, study.execution, study.validation, 4)
        return next(c for c in report.checks if c.name == "solver_claims_match_spec")

    assert claims().passed
    for key, value in [("case_diffusivity", "1"), ("case_equation", "-laplacian(u) = f"),
                       ("case_diffusivity", None), ("case_equation", None)]:
        changed = original.copy()
        if value is None:
            del changed[key]
        else:
            changed[key] = value
        result_file.write_text(json.dumps(changed))
        check = claims()
        assert not check.passed and check.code == "CLAIM_SPEC_MISMATCH"


@pytest.mark.mfem
@requires_mfem
@pytest.mark.parametrize("fault", ["omit_coefficient_gradient", "omit_neumann_coefficient"])
def test_native_variable_faults_are_detected(specs_dir, tmp_path, fault):
    raw = variable_input(specs_dir)
    raw["execution"]["inject_fault"] = fault
    spec = tmp_path / "fault.json"
    spec.write_text(json.dumps(raw))
    manifest = controller.run_study(spec, tmp_path / "runs")
    assert manifest["final_state"] == "FAILED", manifest["final_reason"]
    attempt, = manifest["attempts"]
    assert all(level["exit_code"] == 0 for level in attempt["levels"])
    assert "REFERENCE_MISMATCH" in attempt["failure_codes"]
