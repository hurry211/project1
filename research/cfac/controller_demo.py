"""Archive small CFAC decisions and optionally replay a previous run."""
import argparse
import json
from fractions import Fraction
from pathlib import Path

from cfac.controller import run
from cfac.prototype import uniform_mesh
from cfac.spec import Problem


def problems():
    mesh = uniform_mesh(2)
    return {name: Problem(mesh, (1,) * len(mesh.triangles), 1, Fraction(epsilon), budget)
            for name, epsilon, budget in [
                ("initial_pair", "1/4", 32), ("refine_once", "1/8", 32),
                ("after_correction", "1/9", 20), ("tight_floor", "1/100", 20),
                ("one_action", "1/8", 1), ("no_actions", "1/8", 0)]}


def deterministic(value):
    """Only measured clock readings may differ between journal replays."""
    if isinstance(value, list):
        return [deterministic(v) for v in value]
    if isinstance(value, dict):
        return {k: deterministic(v) for k, v in value.items() if k not in ("elapsed_ns", "cost_ns")}
    return value


def campaign(*, allow_rt1=False):
    if allow_rt1:
        mesh = uniform_mesh(2)
        # This policy demonstration follows the fixed-field threshold study.
        # Include both a threshold that separates the two spaces and a miss.
        cases = {name: Problem(mesh, (1,) * len(mesh.triangles), 1, eps, 24)
                 for name, eps in [("separating", Fraction(3, 20)), ("tighter", Fraction(1, 8))]}
        anchors = {name: problem.digest() for name, problem in cases.items()}
        return {name: {policy: run(p, anchors[name], allow_rt1=enabled)
                       for policy, enabled in (("RT0", False), ("RT0_then_RT1", True))}
                for name, p in cases.items()}
    cases = problems()
    # The caller freezes these before the prover sees a problem. A replay reader
    # can re-create the originals without taking a digest from the saved trace.
    anchors = {name: p.digest() for name, p in cases.items()}
    return {name: run(p, anchors[name]) for name, p in cases.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rt1", action="store_true",
                        help="compare RT0 control with optional RT1 trials at two frozen tolerances")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--replay", type=Path)
    args = parser.parse_args()
    data = campaign(allow_rt1=args.rt1)
    if args.replay:
        if deterministic(data) != deterministic(json.loads(args.replay.read_bytes())):
            parser.exit(1, "Mismatch in the deterministic journal\n")
        print("Reproduced all mathematical observations, actions, gains and terminal states.")
    else:
        with args.output.open("x", encoding="utf-8", newline="\n") as out:
            json.dump(data, out, indent=2, sort_keys=True)
            out.write("\n")


if __name__ == "__main__":
    main()
