"""Command-line entry point: ``python -m sciagent <command>``."""

from __future__ import annotations

import argparse
import json
import sys

from . import capability, controller, environment, mfem_runner


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sciagent", description="Deterministic MFEM verification workflow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env", help="print the environment probe as JSON")
    sub.add_parser("build", help="build cpp/poisson_mms against the pinned MFEM install")
    sub.add_parser("cases", help="list manufactured cases compiled into the solver")
    sub.add_parser("matrix", help="print the capability matrix")
    r = sub.add_parser("run", help="run a study spec end to end")
    r.add_argument("spec")
    r.add_argument("--runs-dir", default="runs")
    v = sub.add_parser("verify", help="re-verify a stored attempt directory from artifacts only")
    v.add_argument("attempt_dir")
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
        for row in capability.matrix_as_rows():
            print(json.dumps(row))
    elif a.cmd == "run":
        m = controller.run_study(a.spec, a.runs_dir)
        print(controller.summarize(m))
        return 0 if m["final_state"] == "ACCEPTED" else 1
    elif a.cmd == "verify":
        out = controller.reverify(a.attempt_dir)
        print(json.dumps({"verdict": out["verdict"], "failure_codes": out["failure_codes"],
                          "levels": [{"n": lv["n"], "l2": lv["l2_error"], "h1_semi": lv["h1_semi_error"]}
                                     for lv in out["levels"]], "orders": out["convergence"]["pairs"]}, indent=2))
        return 0 if out["verdict"] == "ACCEPTED" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
