"""Recompute exact floor evidence on small, frozen problems."""
import argparse
from dataclasses import asdict
from fractions import Fraction as Q
import json
from pathlib import Path

from cfac.evaluation import four_terms
from cfac.floor import assess
from cfac.prototype import ReferenceSystem, uniform_mesh
from cfac.spec import Problem


def outward(interval, scale=10**10):
    """Short rational outer enclosure for storage of a longer series bound."""
    lo, hi = interval
    return [Q((lo * scale).__floor__(), scale), Q((hi * scale).__ceil__(), scale)]


def measured_cases():
    mesh = uniform_mesh(3)
    problem = Problem(mesh, (1,) * len(mesh.triangles), 2, Q(1, 10), 12)
    anchor = problem.digest()
    system = ReferenceSystem(problem)
    primal = system.primal_step(system.zero_primal())
    flux = system.local_dual_sweep(system.feasible_flux)
    loose = assess(problem, anchor, primal, flux, witness=True)
    tight = assess(problem, anchor, primal, flux, tight=True, witness=True)
    split = four_terms(system, primal, flux, cutoff=31)
    if tight.floor.lower != tight.floor.upper or tight.floor.lower != split.pair_floor_sq:
        raise ArithmeticError("discrete floor disagrees with independent assembly")
    if tight.certificate.gap_sq != split.pair_floor_sq + split.primal_algebraic_sq + split.dual_algebraic_sq:
        raise ArithmeticError("four-term identity failed")

    # A globally affine source with a nonzero projection residual on each cell.
    means = tuple(sum(mesh.vertices[i][0] for i in t) / 3 for t in mesh.triangles)
    original = Problem(mesh, means, 2, Q(1, 10), 12, original_affine=(0, 1, 0))
    projected = Problem(mesh, means, 2, Q(1, 10), 12)
    original_anchor, projected_anchor = original.digest(), projected.digest()
    projected_system = ReferenceSystem(projected)
    v, q = projected_system.zero_primal(), projected_system.feasible_flux
    orig = assess(original, original_anchor, v, q, tight=True)
    proj = assess(projected, projected_anchor, v, q, tight=True)
    changed = list(q)
    changed[0] += Q(1, 100)
    defect = assess(projected, projected_anchor, v, changed)
    if orig.floor is not None or defect.floor is not None or proj.floor is None:
        raise ArithmeticError("floor scope check failed")
    if not (orig.certificate.upper_sq > proj.certificate.upper_sq and defect.certificate.defect_sq > 0):
        raise ArithmeticError("missing correction")
    if not defect.certificate.gap_sq < projected.epsilon**2 < defect.certificate.upper_sq or defect.certificate.accepted:
        raise ArithmeticError("defect-only threshold crossing must not accept the field")
    return {
        "constant_load": {
            "n": 3, "load": 1, "diffusion": 2, "epsilon": "1/10",
            "primal": primal, "flux": flux, "problem_digest": anchor,
            "loose": asdict(loose), "tight": asdict(tight),
            "separate_reference": {
                "series_cutoff": 31, "primal_disc_sq": outward(split.primal_discretization_sq),
                "dual_disc_sq": outward(split.dual_discretization_sq),
                "primal_alg_sq": split.primal_algebraic_sq, "dual_alg_sq": split.dual_algebraic_sq,
                "pair_floor_sq": split.pair_floor_sq,
            },
        },
        "affine_load": {
            "description": "original f=x, projected f_h=cell mean of x, on the same n=3 mesh",
            "primal": v, "flux": q, "changed_first_flux_moment": changed[0],
            "original_digest": original_anchor, "projected_digest": projected_anchor,
            "original": asdict(orig), "projected": asdict(proj), "perturbed_projected": asdict(defect),
        },
    }


def serialise(value):
    if isinstance(value, Q):
        return str(value)
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    raw = json.dumps(measured_cases(), default=serialise, indent=2, sort_keys=True) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as f:
        f.write(raw)


if __name__ == "__main__":
    main()
