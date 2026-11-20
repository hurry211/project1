"""Generate the example and negative-test spec files under specs/.

Run: python scripts/make_specs.py   (idempotent; overwrites only specs/*.json)
"""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "specs"

SIN_SIN = {"exact_solution": "sin(pi*x)*sin(pi*y)", "source": "2*pi**2*sin(pi*x)*sin(pi*y)"}
COS_EXP = {"exact_solution": "cos(pi*x)*exp(y)", "source": "(pi**2 - 1)*cos(pi*x)*exp(y)"}
SIDES = [(1, "bottom"), (2, "right"), (3, "top"), (4, "left")]


def bcs(neumann=()):
    return [{"attribute": a, "label": l, "kind": "neumann" if l in neumann else "dirichlet",
             "value": "from_exact_solution"} for a, l in SIDES]


def study(pid, case, execution, neumann=(), validation=None, policy=None):
    return {
        "schema_version": "0.2",
        "problem": {"id": pid, "equation": "-laplacian(u) = f", "domain": "unit_square_2d",
                    **case, "boundaries": bcs(neumann)},
        "execution": execution,
        "validation": validation or {},
        "policy": policy or {},
    }


def main():
    out = {}
    p1tri = {"mesh_type": "tri", "polynomial_order": 1, "mesh_levels": [8, 16, 32, 64],
             "backend": "serial_cpu", "assembly": "full", "solver": "cg_gs"}
    out["poisson_sin_sin_p1_tri"] = study("poisson_unit_square_mms", SIN_SIN, p1tri)
    out["poisson_sin_sin_p2_tri"] = study("poisson_unit_square_mms_p2", SIN_SIN,
                                          {**p1tri, "polynomial_order": 2, "mesh_levels": [4, 8, 16, 32]})
    out["poisson_sin_sin_p1_tri_jacobi"] = study("poisson_unit_square_mms_jacobi", SIN_SIN,
                                                 {**p1tri, "solver": "cg_jacobi"})
    out["poisson_cos_exp_mixed_p1_quad"] = study(
        "poisson_mixed_bc_mms", COS_EXP,
        {"mesh_type": "quad", "polynomial_order": 1, "mesh_levels": [8, 16, 32, 64],
         "assembly": "full", "solver": "cg_gs"}, neumann=("top",))
    out["poisson_cos_exp_mixed_p2_quad"] = study(
        "poisson_mixed_bc_mms_p2", COS_EXP,
        {"mesh_type": "quad", "polynomial_order": 2, "mesh_levels": [4, 8, 16, 32],
         "assembly": "full", "solver": "cg_gs"}, neumann=("top",))
    out["poisson_cos_exp_mixed_p1_quad_pa"] = study(
        "poisson_mixed_bc_mms_pa", COS_EXP,
        {"mesh_type": "quad", "polynomial_order": 1, "mesh_levels": [8, 16, 32, 64],
         "assembly": "partial", "solver": "cg_jacobi"}, neumann=("top",))
    out["poisson_cos_exp_mixed_p2_quad_pa"] = study(
        "poisson_mixed_bc_mms_p2_pa", COS_EXP,
        {"mesh_type": "quad", "polynomial_order": 2, "mesh_levels": [4, 8, 16, 32],
         "assembly": "partial", "solver": "cg_jacobi"}, neumann=("top",))
    # Bounded retry demonstration: the accuracy target is fixed in advance; the
    # controller may only append refined mesh levels.
    out["retry_demo_sin_sin_p1_tri"] = study(
        "poisson_unit_square_mms_retry", SIN_SIN, {**p1tri, "mesh_levels": [4, 8, 16]},
        validation={"max_l2_error_finest": 1.0e-4}, policy={"max_attempts": 4})
    out["budget_demo_sin_sin_p1_tri"] = study(
        "poisson_unit_square_mms_budget", SIN_SIN, {**p1tri, "mesh_levels": [4, 8, 16]},
        validation={"max_l2_error_finest": 1.0e-9}, policy={"max_attempts": 2})

    neg = {}
    base = out["poisson_sin_sin_p1_tri"]
    mixed = out["poisson_cos_exp_mixed_p1_quad"]
    s = copy.deepcopy(base); s["problem"]["boundaries"] = s["problem"]["boundaries"][:3]
    neg["missing_boundary"] = s
    s = copy.deepcopy(base); s["problem"]["boundaries"][3]["attribute"] = 1
    neg["duplicate_attribute"] = s
    s = copy.deepcopy(base); s["problem"]["boundaries"][0]["value"] = "0"
    neg["unsupported_bc_value"] = s
    s = copy.deepcopy(base); s["problem"]["turbulence_model"] = "k-epsilon"
    neg["unknown_field"] = s
    s = copy.deepcopy(base); s["execution"]["mesh_levels"] = [8, 12, 16]
    neg["non_doubling_levels"] = s
    s = copy.deepcopy(base); s["problem"]["source"] = "pi**2*sin(pi*x)*sin(pi*y)"
    neg["inconsistent_mms_source"] = s
    s = copy.deepcopy(base); s["problem"]["source"] = "__import__('os').system('true')"
    neg["malicious_expression"] = s
    s = copy.deepcopy(base)
    s["problem"]["exact_solution"] = "x*(1 - x)*y*(1 - y)"
    s["problem"]["source"] = "2*(x*(1 - x) + y*(1 - y))"
    neg["unbound_case"] = s
    s = copy.deepcopy(base); s["execution"]["backend"] = "mpi_cpu"
    neg["unsupported_backend_mpi"] = s
    s = copy.deepcopy(base); s["execution"]["backend"] = "cuda"
    neg["unsupported_backend_cuda"] = s
    s = copy.deepcopy(base); s["execution"]["solver"] = "umfpack"
    neg["unsupported_solver_umfpack"] = s
    s = copy.deepcopy(base); s["execution"].update(assembly="partial", solver="cg_gs")
    neg["unsupported_pa_with_gs"] = s
    s = copy.deepcopy(base); s["execution"]["polynomial_order"] = 3
    neg["not_verified_order3"] = s
    # Swapped labels: attribute 1 (y=0 in the mesh) is declared "top" and 3 "bottom".
    s = copy.deepcopy(mixed)
    b = s["problem"]["boundaries"]
    b[0]["label"], b[2]["label"] = "top", "bottom"
    b[0]["kind"], b[2]["kind"] = "neumann", "dirichlet"
    neg["swapped_boundary_labels"] = s
    # Attribute 5 does not exist in the mesh; attribute 4 is left unassigned.
    s = copy.deepcopy(base); s["problem"]["boundaries"][3]["attribute"] = 5
    neg["unassigned_mesh_attribute"] = s
    for fault in ("wrong_source", "drop_dirichlet_attr", "nan_value", "early_stop_lie", "self_report_lie"):
        src = mixed if fault == "wrong_source" else base
        s = copy.deepcopy(src); s["execution"]["inject_fault"] = fault
        neg[f"fault_{fault}"] = s
    # Under-resolved but otherwise correct: orders are not yet asymptotic.
    s = copy.deepcopy(base); s["execution"]["mesh_levels"] = [1, 2, 4]
    s["validation"] = {"l2_order_min_offset": 0.05, "l2_order_max_offset": 0.05,
                       "h1_order_min_offset": 0.05, "h1_order_max_offset": 0.05}
    neg["preasymptotic_strict_band"] = s
    # Resolved meshes (quadrature-consistent) but a deliberately strict band:
    # the order gate must be the only failing check.
    s = copy.deepcopy(base); s["execution"]["mesh_levels"] = [4, 8, 16]
    s["validation"] = {"l2_order_min_offset": 0.02, "l2_order_max_offset": 0.02}
    neg["strict_band_resolved"] = s

    SPECS.mkdir(exist_ok=True)
    (SPECS / "negative").mkdir(exist_ok=True)
    for name, obj in out.items():
        (SPECS / f"{name}.json").write_text(json.dumps(obj, indent=2) + "\n")
    for name, obj in neg.items():
        (SPECS / "negative" / f"{name}.json").write_text(json.dumps(obj, indent=2) + "\n")
    print(f"wrote {len(out)} specs and {len(neg)} negative specs")


if __name__ == "__main__":
    main()
