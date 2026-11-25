"""Compatibility of the compiled case catalog with the Poisson adapter."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import requires_mfem
from sciagent import mfem_runner
from sciagent.spec import load_study


def binary_catalog(monkeypatch, payload):
    """Exercise the public registry parser without requiring a compiled solver."""
    result = SimpleNamespace(returncode=0, stdout=json.dumps(payload))
    monkeypatch.setattr(mfem_runner.subprocess, "run", lambda *args, **kwargs: result)
    return mfem_runner.list_cases(Path(__file__))


def legacy_entry(problem):
    return {"id": "old-id", "exact_solution": problem.exact_solution, "source": problem.source}


def test_existing_formula_only_binary_still_binds(monkeypatch, specs_dir):
    problem = load_study(specs_dir / "poisson_sin_sin_p1_tri.json").problem
    entry = legacy_entry(problem)
    entry["source"] += " "
    catalog = binary_catalog(monkeypatch, {"cases": [entry]})
    assert mfem_runner.bind_case(problem, catalog) == "old-id"


@pytest.mark.parametrize("duplicate", ["id", "signature"])
def test_duplicate_entries_are_rejected_before_binding(monkeypatch, specs_dir, duplicate):
    problem = load_study(specs_dir / "poisson_sin_sin_p1_tri.json").problem
    first = legacy_entry(problem)
    second = ({**first, "source": "1"} if duplicate == "id"
              else {**first, "id": "other", "source": " " + first["source"]})
    catalog = {"cases": [first, second]}
    with pytest.raises(mfem_runner.BuildError, match="duplicate"):
        binary_catalog(monkeypatch, catalog)
    assert mfem_runner.bind_case(problem, catalog) is None


@pytest.mark.parametrize("metadata", [
    {"formulation": {}}, {"registry_version": True}, {"registry_version": 3},
    {"registry_version": 1}, {"registry_version": 1, "formulation": {"equation": "-laplacian(u) = f"}},
])
def test_invalid_versioned_metadata_fails_before_binding(monkeypatch, specs_dir, metadata):
    problem = load_study(specs_dir / "poisson_sin_sin_p1_tri.json").problem
    catalog = {"cases": [legacy_entry(problem)], **metadata}
    with pytest.raises(mfem_runner.BuildError, match="invalid --list-cases"):
        binary_catalog(monkeypatch, catalog)
    assert mfem_runner.bind_case(problem, catalog) is None


@pytest.mark.mfem
@requires_mfem
def test_native_formulation_binds_poisson_problems_and_checks_trace_and_operator(specs_dir):
    catalog = mfem_runner.list_cases()
    assert catalog["registry_version"] == 2
    form = catalog["formulation"]
    assert form == {"equation": "-laplacian(u) = f", "domain": "unit_square_2d", "space": "scalar_H1",
                    "bilinear_integrator": "DiffusionIntegrator", "dirichlet_trace": "u_exact",
                    "neumann_trace": "grad(u_exact).n"}
    for file, expected_id in (("poisson_sin_sin_p1_tri.json", "sin_sin"),
                              ("poisson_cos_exp_mixed_p1_quad.json", "cos_exp")):
        problem = load_study(specs_dir / file).problem
        assert mfem_runner.bind_case(problem, catalog) == expected_id
        for key in form:
            changed = {**catalog, "formulation": {**form, key: "other"}}
            assert mfem_runner.bind_case(problem, changed) is None, key


def test_variable_case_requires_a_versioned_coefficient_and_conormal(monkeypatch, specs_dir):
    problem = load_study(specs_dir / "stage2/diffusion_mixed_p1_tri.json").problem
    entry = legacy_entry(problem)
    default = {"equation": "-laplacian(u) = f", "domain": "unit_square_2d", "space": "scalar_H1",
               "bilinear_integrator": "DiffusionIntegrator", "dirichlet_trace": "u_exact",
               "neumann_trace": "grad(u_exact).n"}
    override = {**default, "equation": problem.equation, "kappa": problem.kappa,
                "neumann_trace": "kappa*grad(u_exact).n"}
    catalog = {"registry_version": 2, "formulation": default, "cases": [{**entry, "formulation": override}]}
    assert mfem_runner.bind_case(problem, binary_catalog(monkeypatch, catalog)) == entry["id"]
    for key in override:
        wrong = {**entry, "formulation": {**override, key: "wrong"}}
        assert mfem_runner.bind_case(problem, {**catalog, "cases": [wrong]}) is None, key
    for old in ({"cases": [entry]}, {"registry_version": 1, "formulation": default, "cases": [entry]},
                {**catalog, "cases": [entry]}):
        assert mfem_runner.bind_case(problem, old) is None
    with pytest.raises(mfem_runner.BuildError, match="case-specific"):
        binary_catalog(monkeypatch, {**catalog, "registry_version": 1})
    incomplete = {key: value for key, value in override.items() if key != "kappa"}
    with pytest.raises(mfem_runner.BuildError, match="case-specific"):
        binary_catalog(monkeypatch, {**catalog, "cases": [{**entry, "formulation": incomplete}]})


def test_v1_registry_binds_legacy_poisson(specs_dir):
    problem = load_study(specs_dir / "poisson_sin_sin_p1_tri.json").problem
    form = {"equation": problem.equation, "domain": problem.domain, "space": "scalar_H1",
            "bilinear_integrator": "DiffusionIntegrator", "dirichlet_trace": "u_exact",
            "neumann_trace": "grad(u_exact).n"}
    catalog = {"registry_version": 1, "formulation": form, "cases": [legacy_entry(problem)]}
    assert mfem_runner.bind_case(problem, catalog) == "old-id"
