"""Unit tests: schema validation and immutability of the data contracts."""

import json

import pytest
from pydantic import ValidationError

from sciagent.spec import ExecutionSpec, StudySpec, load_study, sha256_of
from verifier import expr
from verifier.verifier import check_mms_consistency


def _load(specs_dir, name):
    return json.loads((specs_dir / name).read_text())


@pytest.mark.parametrize("name", [
    "poisson_sin_sin_p1_tri.json", "poisson_sin_sin_p2_tri.json", "poisson_cos_exp_mixed_p1_quad.json",
    "poisson_cos_exp_mixed_p2_quad_pa.json", "retry_demo_sin_sin_p1_tri.json",
])
def test_positive_specs_parse(specs_dir, name):
    s = load_study(specs_dir / name)
    assert s.schema_version == "0.2"
    assert check_mms_consistency(s.problem, s.validation.mms_consistency_rel_tol).passed


@pytest.mark.parametrize("name,fragment", [
    ("missing_boundary.json", "missing"),
    ("duplicate_attribute.json", "duplicate boundary attribute"),
    ("unsupported_bc_value.json", "from_exact_solution"),
    ("unknown_field.json", "Extra inputs are not permitted"),
    ("non_doubling_levels.json", "doublings"),
])
def test_negative_specs_rejected_by_schema(specs_dir, name, fragment):
    with pytest.raises(ValidationError) as ei:
        StudySpec.model_validate(_load(specs_dir / "negative", name))
    assert fragment in str(ei.value)


def test_inconsistent_mms_source_detected(specs_dir):
    s = StudySpec.model_validate(_load(specs_dir / "negative", "inconsistent_mms_source.json"))
    c = check_mms_consistency(s.problem, s.validation.mms_consistency_rel_tol)
    assert not c.passed and c.code == "MMS_INCONSISTENT"


def test_malicious_expression_rejected(specs_dir):
    s = StudySpec.model_validate(_load(specs_dir / "negative", "malicious_expression.json"))
    c = check_mms_consistency(s.problem, 1e-6)
    assert not c.passed and c.code == "INVALID_EXPRESSION"


@pytest.mark.parametrize("bad", ["__import__('os')", "x.real", "[x for x in y]", "open('f')", "lambda: 1", "x if y else 1"])
def test_expression_whitelist(bad):
    with pytest.raises(expr.ExpressionError):
        expr.compile_expr(bad)


def test_expression_values_and_derivatives():
    import numpy as np
    u = expr.compile_expr("sin(pi*x)*sin(pi*y)")
    x = np.array([0.3, 0.7]); y = np.array([0.2, 0.9])
    assert np.allclose(u(x, y), np.sin(np.pi * x) * np.sin(np.pi * y), atol=0, rtol=1e-15)
    gx, gy = expr.gradient(u, x, y)
    assert np.allclose(gx, np.pi * np.cos(np.pi * x) * np.sin(np.pi * y), rtol=0, atol=1e-9)
    assert np.allclose(expr.neg_laplacian(u, x, y), 2 * np.pi**2 * u(x, y), rtol=0, atol=1e-7)


def test_models_are_frozen(specs_dir):
    s = load_study(specs_dir / "poisson_sin_sin_p1_tri.json")
    with pytest.raises(ValidationError):
        s.problem.source = "0"
    with pytest.raises(ValidationError):
        s.validation.l2_order_min_offset = 5.0


def test_hash_is_stable_and_sensitive(specs_dir):
    a = load_study(specs_dir / "poisson_sin_sin_p1_tri.json")
    b = load_study(specs_dir / "poisson_sin_sin_p1_tri.json")
    assert sha256_of(a.problem) == sha256_of(b.problem)
    c = StudySpec.model_validate({**a.model_dump(), "validation": {**a.validation.model_dump(), "min_levels": 4}})
    assert sha256_of(c.validation) != sha256_of(a.validation)
    assert sha256_of(c.problem) == sha256_of(a.problem)


def test_execution_bounds():
    with pytest.raises(ValidationError):
        ExecutionSpec(mesh_levels=(8, 16, 32), rel_tol=0.5)
    with pytest.raises(ValidationError):
        ExecutionSpec(mesh_levels=(0, 0))
    assert ExecutionSpec(polynomial_order=2).estimated_dofs(8) == 17 ** 2


@pytest.mark.parametrize("section,field", [
    ("validation", "reference_solve_rel_tol"), ("policy", "max_total_wall_seconds"),
])
def test_nonfinite_thresholds_rejected(specs_dir, section, field):
    raw = _load(specs_dir, "poisson_sin_sin_p1_tri.json")
    raw[section][field] = float("inf")
    with pytest.raises(ValidationError):
        StudySpec.model_validate(raw)


@pytest.mark.parametrize("literal", ["9" * 370, "1e400"])
def test_unrepresentable_expression_literal(literal):
    with pytest.raises(expr.ExpressionError):
        expr.compile_expr(literal)
