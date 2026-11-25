"""The planner may nominate a reviewed input within the user's structured authorization."""

import hashlib
import json
import shutil

import pytest

from sciagent import planner
from sciagent.planner import PlanningTask, catalog_rows, decide, eligible_ids, materialize, parse_task, preflight
from sciagent.spec import RetryPolicy, StudySpec, ValidationSpec


GS = "poisson_sin_sin_p1_tri"
JACOBI = "poisson_sin_sin_p1_tri_jacobi"
GMSH = "diffusion_gmsh_p2_quad"
TASK = PlanningTask(case="sin_sin", polynomial_order=1)


def proposal(key, **extra):
    return json.dumps({"template_id": key, **extra})


def test_catalog_uses_reviewed_serial_workflows_only(tmp_path):
    rows = catalog_rows()
    assert len(rows) == len(set(row["template_id"] for row in rows)) == 9
    for row in rows:
        task = PlanningTask(**{k:row[k] for k in planner.SELECTORS})
        key = row["template_id"]
        assert eligible_ids(task) == (key,)
        assert decide(task, planner.rule_proposal(task)) == {"status":"PLANNED", "code":"REVIEWED_TEMPLATE",
                                                             "template_id":key}
        path, pinned = materialize(key, tmp_path / key)
        assert path.name == "study.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["spec_sha256"] == pinned
        study = StudySpec.model_validate_json(path.read_bytes())
        assert study.validation == ValidationSpec() and study.policy == RetryPolicy()
        assert study.execution.backend == "serial_cpu" and study.execution.inject_fault == "none"
        if study.external_mesh:
            mesh = study.external_mesh
            assert hashlib.sha256((path.parent / mesh.file).read_bytes()).hexdigest() == mesh.sha256


def test_hard_constraints_authorize_multiple_discretizations_of_the_same_physics():
    task = parse_task('{"case":"sin_sin","polynomial_order":1,"description":"Prefer Jacobi smoothing"}')
    assert eligible_ids(task) == (GS, JACOBI)
    assert planner.rule_proposal(task) == proposal(GS)
    assert decide(task, proposal(JACOBI))["status"] == "PLANNED"
    payload = json.loads(planner.build_prompt(task).split("\n", 1)[1])
    assert {c["template_id"] for c in payload["authorized_choices"]} == {GS, JACOBI}
    assert payload["preference"] == task.description


def test_descriptive_preferences_cannot_expand_structured_authorization():
    task = PlanningTask(case="sin_sin", solver="cg_gs", description="Use the variable coefficient Gmsh example.")
    assert decide(task, proposal(GMSH))["code"] == "UNAUTHORIZED_TEMPLATE"
    assert decide(task, proposal(JACOBI))["code"] == "UNAUTHORIZED_TEMPLATE"


@pytest.mark.parametrize(("fields","code"), [
    ({"case":"anisotropic_diffusion"}, "UNREVIEWED_CASE"),
    ({"case":"sin_sin","solver":"hypre_amg"}, "NO_REVIEWED_CONFIGURATION"),
    ({"case":"sin_sin","polynomial_order":3}, "NO_REVIEWED_CONFIGURATION"),
    ({"case":"sin_sin","change_request":"Raise the error limit."}, "CHANGE_REQUIRES_REVIEW"),
    ({"case":"sin_sin","change_request":"Change to a pure Neumann model."}, "CHANGE_REQUIRES_REVIEW"),
])
def test_review_holds_precede_any_provider_nomination(fields, code):
    task = PlanningTask(**fields)
    expected = {"status":"REVIEW_REQUIRED", "code":code, "template_id":None}
    assert eligible_ids(task) == ()
    assert preflight(task) == decide(task, proposal(GS)) == expected


@pytest.mark.parametrize("raw", [
    '{"case":"sin_sin","case":"cos_exp"}',
    '{"case":"sin_sin","validation":{"reference_solve_rel_tol":1}}',
    '{"case":"sin_sin","policy":{"max_attempts":6}}',
    '{"case":"sin_sin","polynomial_order":true}',
    '{"case":"sin_sin","polynomial_order":"1"}',
    '{"case":"sin_sin","description":NaN}',
    b'{"case":"\xff"}',
    " " * 8193,
])
def test_invalid_task_never_silently_loses_constraints(raw):
    with pytest.raises(ValueError):
        parse_task(raw)


@pytest.mark.parametrize(("raw","code"), [
    (proposal(None), "PLANNER_ABSTAINED"),
    (proposal("poisson_sin_sin_p2_tri"), "UNAUTHORIZED_TEMPLATE"),
    (proposal("../negative/fault_wrong_source.json"), "UNAUTHORIZED_TEMPLATE"),
    (proposal(GS, validation={"require_reference_solve":False}), "INVALID_PROPOSAL"),
    (proposal(GS, policy={}), "INVALID_PROPOSAL"),
    (proposal(GS, command="run"), "INVALID_PROPOSAL"),
    (f'{{"template_id":null,"template_id":"{GS}"}}', "INVALID_PROPOSAL"),
    ('{"template_id":false}', "INVALID_PROPOSAL"),
    ('{"template_id":{}}', "INVALID_PROPOSAL"),
    ('{"template_id":NaN}', "INVALID_PROPOSAL"),
    ('```json\n{"template_id":null}\n```', "INVALID_PROPOSAL"),
    ('{"template_id":null} {}', "INVALID_PROPOSAL"),
    (proposal(GS) + " "*512, "INVALID_PROPOSAL"),
])
def test_invalid_nomination_never_materializes_a_study(raw, code):
    assert decide(TASK, raw) == {"status":"REJECTED", "code":code, "template_id":None}


def test_materialization_checks_source_and_mesh_before_creating_directory(tmp_path, monkeypatch):
    source = tmp_path / "specs"
    shutil.copytree(planner.SPECS, source)
    monkeypatch.setattr(planner, "SPECS", source)
    spec = source / (GS + ".json")
    original = spec.read_bytes().replace(b"\r\n", b"\n")
    spec.write_bytes(original.replace(b"\n", b"\r\n"))
    copy, digest = materialize(GS, tmp_path / "good")
    spec.write_bytes(original.replace(b'"validation": {}', b'"validation": {"require_reference_solve": false}'))
    assert copy.read_bytes() == original and digest == planner.template_sha256(GS)
    with pytest.raises(ValueError, match="study digest"):
        materialize(GS, tmp_path / "changed")
    mesh = source / "stage2/unit_square_quad.msh"
    mesh.write_bytes(mesh.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="mesh SHA-256"):
        materialize(GMSH, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


def test_materialization_refuses_existing_targets_and_unreviewed_names(tmp_path):
    with pytest.raises(FileExistsError):
        materialize(GS, tmp_path)
    with pytest.raises(ValueError, match="unknown reviewed template"):
        materialize("../specs/negative/fault_wrong_source", tmp_path / "bad")
    assert not list(tmp_path.iterdir())
