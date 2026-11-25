"""Command-line entry point: ``python -m sciagent <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import capability, controller, environment, mfem_runner, mpi_study
from .evidence import sha256_file


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sciagent", description="Deterministic MFEM verification workflow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env", help="print the environment probe as JSON")
    sub.add_parser("build", help="build cpp/poisson_mms against the pinned MFEM install")
    sub.add_parser("cases", help="list manufactured cases compiled into the solver")
    sub.add_parser("plan-catalog", help="list reviewed planner templates and their hashes")
    from benchmarks.e1_select import selection_arguments
    tune = sub.add_parser("select-solver", help="select and verify a solver for a calibrated serial E1 workload")
    selection_arguments(tune)
    plan = sub.add_parser("plan", help="select an authorized template and optionally execute it")
    plan.add_argument("task", type=Path, help="JSON task with a case, hard selectors and optional preferences")
    plan.add_argument("--directory", type=Path, required=True, help="fresh planning archive")
    plan.add_argument("--mode", choices=("rules", "llm", "bounded"), default="rules")
    plan.add_argument("--model", help="installed local Ollama model name")
    plan.add_argument("--model-digest", help="full SHA-256 of the local model")
    plan.add_argument("--proposal-directory", type=Path, help="completed live planning preview for this exact task")
    plan.add_argument("--proposal-manifest-sha256", help="externally retained digest of the planning preview")
    plan.add_argument("--execute", action="store_true", help="run the selected template and independently verify it")
    pv = sub.add_parser("verify-plan", help="reconstruct the plan and replay any numerical attempts offline")
    pv.add_argument("directory", type=Path)
    pv.add_argument("--manifest-sha256", help="externally retained planning manifest digest")
    matrix = sub.add_parser("matrix", help="print reviewed build and solver coverage as JSON lines")
    matrix.add_argument("--case", choices=capability.ALL_CASE_IDS)
    matrix.add_argument("--mesh-source", choices=capability.MESH_SOURCES, default="cartesian")
    matrix.add_argument("--backend", choices=("serial_cpu", "mpi_cpu"), default="serial_cpu")
    mpisel = sub.add_parser("select-mpi", help="run a verified backend and rank count from a pinned MPI calibration")
    mpisel.add_argument("--calibration", type=Path, required=True)
    mpisel.add_argument("--report-sha256", required=True)
    mpisel.add_argument("--directory", type=Path, required=True)
    mpi = sub.add_parser("mpi-study", help="verify MPI against serial and measure strong and weak scaling")
    mpi.add_argument("--directory", type=Path, required=True, help="new campaign directory")
    mpi.add_argument("--repetitions", type=int, default=3, help="timing samples per configuration (3 to 15)")
    mpiv = sub.add_parser("verify-mpi", help="replay an MPI campaign from archived fields and timing records")
    mpiv.add_argument("directory", type=Path)
    mpiv.add_argument("--manifest-sha256", help="externally retained campaign manifest digest")
    family = sub.add_parser("pde-study", help="verify a manufactured tensor-grid heat, wave or elasticity study")
    family.add_argument("--family", choices=("heat", "wave", "elasticity"), required=True)
    family.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    family.add_argument("--order", type=int, choices=(1, 2), required=True)
    family.add_argument("--levels", type=int, nargs=3, default=[4, 8, 16])
    family.add_argument("--directory", type=Path, required=True)
    fr = sub.add_parser("verify-pde", help="independently recompute a pinned manufactured field study")
    fr.add_argument("directory", type=Path)
    fr.add_argument("--manifest-sha256", required=True)
    r = sub.add_parser("run", help="run a study spec end to end")
    r.add_argument("spec")
    r.add_argument("--runs-dir", default="runs")
    s = sub.add_parser("search", help="compare verified serial CPU Poisson strategies")
    s.add_argument("spec")
    s.add_argument("--output", required=True, help="new directory for candidate specs, runs and search.json")
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--solvers", choices=("cg_gs", "cg_jacobi", "cg_none"), nargs="+", default=["cg_jacobi"])
    q = sub.add_parser("quickstart", help="probe tools, run the P1 triangle benchmark and replay its evidence")
    q.add_argument("--runs-dir", default="runs")
    v = sub.add_parser("verify", help="re-verify a stored attempt directory from artifacts only")
    v.add_argument("attempt_dir")
    v.add_argument("--manifest-sha256", help="expected manifest digest saved outside the run directory")
    a = ap.parse_args(argv)

    if a.cmd == "env":
        print(json.dumps(environment.probe(), indent=2))
    elif a.cmd == "build":
        rec = mfem_runner.build_solver(environment.PROJECT_ROOT / "build" / "logs")
        print(json.dumps(rec, indent=2))
        return 0 if rec["ok"] else 1
    elif a.cmd == "cases":
        print(json.dumps(mfem_runner.list_cases(), indent=2))
    elif a.cmd == "plan-catalog":
        from .planner import catalog_rows
        print(json.dumps(catalog_rows(), indent=2, ensure_ascii=False))
    elif a.cmd == "select-solver":
        from benchmarks.e1_select import submit
        try:
            return submit(a)
        except (OSError, ValueError) as exc:
            print(json.dumps({"eligible": False, "reason": str(exc)}))
            return 2
    elif a.cmd in ("plan", "verify-plan"):
        from . import planner, planning_run
        try:
            if a.cmd == "plan":
                with a.task.open("rb") as src:
                    task = planner.parse_task(src.read(8193))
                report = planning_run.run(task, a.directory, mode=a.mode, model=a.model,
                                          model_digest=a.model_digest, execute=a.execute,
                                          proposal_directory=a.proposal_directory,
                                          proposal_manifest_sha256=a.proposal_manifest_sha256)
            else:
                report = planning_run.replay(a.directory, a.manifest_sha256)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            print(json.dumps({"state": "FAILED", "reason": str(exc)}, ensure_ascii=False))
            return 1
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        if a.cmd == "verify-plan":
            return 0  # A correctly replayed refusal is a valid archive.
        return 0 if report["state"] in ("PLANNED", "ACCEPTED") else 1
    elif a.cmd == "matrix":
        if a.backend == "mpi_cpu":
            reviewed = a.mesh_source == "cartesian" and a.case in (None, "sin_sin")
            print(json.dumps({"status": "tested", **mpi_study.reviewed_configuration()} if reviewed else
                             {"status": "not_verified", "case": a.case, "mesh_source": a.mesh_source,
                              "backend": a.backend}))
        else:
            for row in capability.matrix_as_rows(a.case, a.mesh_source):
                print(json.dumps(row))
    elif a.cmd == "select-mpi":
        from benchmarks.mpi_scaling import select
        try:
            report = select(a.calibration.resolve(), a.report_sha256, a.directory.resolve())
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            report = {"verdict": "REJECTED", "reason": str(exc)}
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["verdict"] == "ACCEPTED" else 1
    elif a.cmd in ("mpi-study", "verify-mpi"):
        try:
            if a.cmd == "mpi-study":
                report = mpi_study.run(a.directory, a.repetitions)
            else:
                report = mpi_study.replay(a.directory, a.manifest_sha256)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report = {"state": "FAILED", "reason": str(exc)}
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["state"] == "ACCEPTED" else 1
    elif a.cmd in ("pde-study", "verify-pde"):
        from . import field_study
        import subprocess
        try:
            if a.cmd == "pde-study":
                cfg = field_study.request(a.family, a.dimension, a.order, a.levels)
                result = field_study.capture(a.directory, cfg)
            else:
                result = field_study.replay(a.directory, a.manifest_sha256)
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
            result = {"verdict": "REJECTED", "reason": str(exc)}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["verdict"] == "ACCEPTED" else 1
    elif a.cmd == "run":
        m = controller.run_study(a.spec, a.runs_dir)
        print(controller.summarize(m))
        return 0 if m["final_state"] == "ACCEPTED" else 1
    elif a.cmd == "search":
        from .spec import load_study
        from .strategy import search

        try:
            report = search(load_study(a.spec), Path(a.output), a.repeats, solvers=a.solvers)
        except (mfem_runner.BuildError, OSError, ValueError) as exc:
            s.error(str(exc))
        print(json.dumps({"winner": report["winner"], "report": str(Path(a.output) / "search.json")}))
        return 0 if report["winner"] else 1
    elif a.cmd == "quickstart":
        spec = environment.PROJECT_ROOT / "specs" / "poisson_sin_sin_p1_tri.json"
        manifest = controller.run_study(spec, a.runs_dir)
        run = Path(manifest["run_dir"])
        env = json.loads((run / "environment.json").read_text())
        result = {"tools": {"python": env["python"], "compiler": env["cxx"], "cmake": env["cmake"],
                            "mfem": env["mfem"].get("version"), "solver_available": bool(env["solver_binary_sha256"])},
                  "run_dir": str(run), "state": manifest["final_state"], "code": manifest["final_code"],
                  "reason": manifest["final_reason"], "manifest_sha256": sha256_file(run / "manifest.json")}
        if manifest["final_state"] == "ACCEPTED":
            attempt = run / "attempts" / f"attempt_{manifest['attempts'][-1]['attempt']:02d}"
            replay = controller.reverify(attempt, result["manifest_sha256"])
            result.update(attempt_dir=str(attempt), replay=replay["verdict"], integrity=replay["integrity"],
                          finest_orders=manifest["attempts"][-1]["orders"][-1])
        print(json.dumps(result, indent=2))
        return 0 if result.get("replay") == "ACCEPTED" else 1
    elif a.cmd == "verify":
        out = controller.reverify(a.attempt_dir, a.manifest_sha256)
        print(json.dumps({"verdict": out["verdict"], "failure_codes": out["failure_codes"],
                          "integrity": out["integrity"],
                          "levels": [{"n": lv["n"], "l2": lv["l2_error"], "h1_semi": lv["h1_semi_error"],
                                      **({"physics_checks": [c for c in lv["checks"] if c["name"].startswith("layered_")]}
                                         if out["convergence"].get("mode") == "exact_in_q1" else {})}
                                     for lv in out["levels"]], "orders": out["convergence"]["pairs"]}, indent=2))
        return 0 if out["verdict"] == "ACCEPTED" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
