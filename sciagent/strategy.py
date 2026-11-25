"""Exhaustive, bounded CPU strategy baseline for a frozen Poisson study."""

from itertools import product
import json
import random
import statistics
import time
from pathlib import Path

from . import capability, environment, mfem_runner
from .controller import run_study
from .evidence import sha256_file
from .spec import POISSON, StudySpec


def search(study: StudySpec, output: Path, repeats: int, *, solvers=("cg_jacobi",)) -> dict:
    """Run reviewed candidates and choose the least median-cost feasible one."""
    if (study.external_mesh is not None or study.problem.equation != POISSON or study.execution.mesh_type != "quad"
            or study.execution.backend != "serial_cpu" or study.execution.inject_fault != "none"):
        raise ValueError("search supports fault-free serial CPU Poisson on generated Cartesian quadrilaterals")
    if study.validation.max_l2_error_finest is None or not study.validation.require_reference_solve:
        raise ValueError("search requires a frozen L2 target and independent reference solves")
    if not 1 <= repeats <= 10:
        raise ValueError("repeats must be in [1, 10]")

    if not solvers or len(set(solvers)) != len(solvers) or set(solvers) - {"cg_gs", "cg_jacobi", "cg_none"}:
        raise ValueError("select distinct reviewed serial CG preconditioners")
    extended = study.schema_version == "0.4"
    catalog = mfem_runner.list_cases(include_e3=extended)
    bound_case = mfem_runner.bind_case(study.problem, catalog)
    if not bound_case:
        raise ValueError("search requires a manufactured case from the current binary")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    env = environment.probe()
    candidates = []
    eligible = []
    for order, assembly, solver in product((1, 2), ("full", "partial"), solvers):
        key = f"q{order}-{assembly}-{solver.replace('_', '-')}"
        execution = study.execution.model_copy(update={
            "polynomial_order": order, "assembly": assembly, "solver": solver})
        decision = capability.evaluate(execution, env, study.problem.equation, exact_solution=study.problem.exact_solution,
                                       case_id=bound_case, catalog=catalog, e3_verified=extended)
        dofs = execution.estimated_dofs(execution.mesh_levels[-1])
        row = {"id": key, "capability": decision.__dict__, "initial_finest_dofs": dofs, "trials": []}
        candidates.append(row)
        if decision.status != "tested":
            row["excluded"] = decision.code
        elif dofs > min(study.policy.max_dofs_per_level, study.validation.reference_solve_max_dofs):
            row["excluded"] = "INITIAL_DOF_BUDGET"
        else:
            spec = output / f"{key}.json"
            spec.write_text(study.model_copy(update={"execution": execution}).model_dump_json(indent=2))
            eligible.append((spec, row))

    rng = random.Random(20261125)
    for i in range(repeats):
        schedule = eligible.copy()
        rng.shuffle(schedule)
        for position, (spec, row) in enumerate(schedule):
            t0 = time.perf_counter()
            manifest = run_study(spec, output / "runs")
            elapsed = time.perf_counter() - t0
            run = Path(manifest["run_dir"])
            attempts = manifest["attempts"]
            errors = attempts[-1].get("errors") if attempts else None
            row["trials"].append({"state": manifest["final_state"], "code": manifest["final_code"],
                                   "repeat": i, "position": position,
                                   "seconds": elapsed, "finest_errors": errors[-1] if errors else None,
                                   "run_dir": run.relative_to(output).as_posix(),
                                   "manifest_sha256": sha256_file(run / "manifest.json")})
    feasible = []
    for _, row in eligible:
        passed = all(trial["state"] == "ACCEPTED" for trial in row["trials"])
        row["feasible"] = passed
        row["median_seconds"] = statistics.median(t["seconds"] for t in row["trials"])
        if passed:
            feasible.append(row)
    chosen = min(feasible, key=lambda row: (row["median_seconds"], row["id"]), default=None)
    if chosen:
        for row in feasible:
            differences = [candidate["seconds"] - reference["seconds"]
                           for candidate, reference in zip(row["trials"], chosen["trials"], strict=True)]
            row["paired_seconds_minus_winner"] = differences
            row["median_paired_seconds_minus_winner"] = statistics.median(differences)
    report = {"solvers": list(solvers), "schedule_seed": 20261125, "winner": chosen["id"] if chosen else None,
              "criterion": "minimum median end-to-end seconds among all-repetition accepted candidates",
              "l2_target": study.validation.max_l2_error_finest, "repeats": repeats,
              "search_wall_seconds": time.perf_counter() - started, "environment": env,
              "candidates": candidates}
    (output / "search.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
