"""Generate immutable native controls and certify them before scoring the verifier."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sciagent.controller import run_study
from sciagent.spec import StudySpec

from .mutations import read_json, require, write_json
from .reference import Reference, exact, relative_difference

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = Path("attempts/attempt_01")


@dataclass
class CleanLevel:
    n: int
    ref: Reference
    field: np.ndarray
    order: np.ndarray
    solution: np.ndarray


@dataclass
class Control:
    run: Path
    digest: str
    validation: object
    levels: dict
    certificate: dict


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_controls(output):
    """Save three-level studies and their manifest digests in a separate index."""
    (output / "requests").mkdir(parents=True, exist_ok=False)
    index = {}
    for path in sorted((ROOT / "specs").glob("poisson_*.json")):
        spec = read_json(path)
        spec["execution"]["mesh_levels"] = spec["execution"]["mesh_levels"][:3]
        request = output / "requests" / path.name
        write_json(request, spec)
        manifest = run_study(request, output / "runs")
        run = Path(manifest["run_dir"])
        # The clean label comes from the oracle below, never from final_state.
        require((run / ATTEMPT / "levels").is_dir(), f"{path.stem}: {manifest['final_reason']}")
        index[path.stem] = {"path": run.relative_to(output).as_posix(),
                            "manifest_sha256": sha(run / "manifest.json")}
    write_json(output / "index.json", index)


def load_controls(directory):
    controls = {}
    index = read_json(directory / "index.json")
    names = {p.stem for p in (ROOT / "specs").glob("poisson_*.json")}
    require(set(index) == names and len(names) == 7, "need all seven canonical controls")
    for name, item in sorted(index.items()):
        run = directory / item["path"]
        require(sha(run / "manifest.json") == item["manifest_sha256"], "control manifest changed")
        manifest = read_json(run / "manifest.json")
        inventory = manifest["artifact_sha256"]
        require(all(sha(run / p) == h for p, h in inventory.items()), "control inventory changed")
        spec = read_json(ROOT / "specs" / f"{name}.json")
        spec["execution"]["mesh_levels"] = spec["execution"]["mesh_levels"][:3]
        require(read_json(run / "request/study_spec.json") == spec, "control request differs from canonical spec")
        study = StudySpec.model_validate(spec)
        v = study.validation
        levels, cert, errors = {}, [], []
        for n in study.execution.mesh_levels:
            path = run / ATTEMPT / "levels" / f"n{n:04d}"
            ref = Reference.from_spec(name, spec["execution"], n)
            field, order = ref.native_field(path)
            ref.check_topology(path, order, n)
            sol = ref.solve()
            diff = relative_difference(field, sol)
            norms = ref.errors(field)
            x, y = ref.basis.doflocs
            fixed = (x == 0) | (x == 1) | (y == 0) | ((y == 1) & (not ref.mixed))
            boundary_error = float(np.max(np.abs(field[fixed] - exact(ref.mixed, x[fixed], y[fixed]))))
            result = read_json(path / "result.json")
            gaps = [abs(result[k] / h - 1) for k, h in zip(
                ("self_reported_l2_error", "self_reported_h1_semi_error"), norms)]
            require(diff <= v.reference_solve_rel_tol and max(gaps) <= v.self_report_rel_tol
                    and boundary_error <= v.dirichlet_nodal_abs_tol,
                    f"{name}/{n}: oracle disagrees with native field or reported norms")
            errors.append(norms)
            cert.append({"n": n, "dofs": int(ref.basis.N), "relative_nodal_difference": diff,
                         "l2": norms[0], "h1_semi": norms[1], "dirichlet_max_difference": boundary_error,
                         "relative_self_report_differences": gaps})
            levels[n] = CleanLevel(n, ref, field, order, sol)
        ns = list(levels)
        p = study.execution.polynomial_order
        orders = np.log(np.array(errors[-2]) / errors[-1]) / np.log(ns[-1] / ns[-2])
        require(np.all(np.diff(errors, axis=0) < 0)
                and p+1-v.l2_order_min_offset <= orders[0] <= p+1+v.l2_order_max_offset
                and p-v.h1_order_min_offset <= orders[1] <= p+v.h1_order_max_offset,
                f"{name}: independent convergence outside original bands")
        certificate = {"manifest_sha256": item["manifest_sha256"], "levels": cert,
                       "finest_orders": orders.tolist(), "solver_binary_sha256": manifest["environment"]["solver_binary_sha256"]}
        controls[name] = Control(run, item["manifest_sha256"], v, levels, certificate)
    return controls
