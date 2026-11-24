"""Command-line entry point: ``python -m sciagent <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import capability, controller, environment, mfem_runner
from .evidence import sha256_file


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sciagent", description="Deterministic MFEM verification workflow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env", help="print the environment probe as JSON")
    sub.add_parser("build", help="build cpp/poisson_mms against the pinned MFEM install")
    sub.add_parser("cases", help="list manufactured cases compiled into the solver")
    matrix = sub.add_parser("matrix", help="print reviewed build and solver coverage as JSON lines")
    matrix.add_argument("--case", choices=capability.CASE_IDS)
    matrix.add_argument("--mesh-source", choices=capability.MESH_SOURCES, default="cartesian")
    r = sub.add_parser("run", help="run a study spec end to end")
    r.add_argument("spec")
    r.add_argument("--runs-dir", default="runs")
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
    elif a.cmd == "matrix":
        for row in capability.matrix_as_rows(a.case, a.mesh_source):
            print(json.dumps(row))
    elif a.cmd == "run":
        m = controller.run_study(a.spec, a.runs_dir)
        print(controller.summarize(m))
        return 0 if m["final_state"] == "ACCEPTED" else 1
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
                          "levels": [{"n": lv["n"], "l2": lv["l2_error"], "h1_semi": lv["h1_semi_error"]}
                                     for lv in out["levels"]], "orders": out["convergence"]["pairs"]}, indent=2))
        return 0 if out["verdict"] == "ACCEPTED" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
