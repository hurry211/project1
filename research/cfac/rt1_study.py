"""Fixed-P1 certificate comparison on predeclared unit-square problems.

The mesh, diffusion, load and input P1 field are identical between the three
coarse certificates. The fourth certificate uses newly solved refined P1 and
RT0 fields. Physical moments depend on geometry and load. We construct them
once per case, then offer the same coefficients to independent checks under
each separately frozen tolerance. Times cover this shared-work design only.
"""
import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

from cfac.checker import check
from cfac.enrichment import EnrichmentRequest, RTMoments
from cfac.evaluation import optimal_fields
from cfac.floor import assess
from cfac.prototype import ReferenceSystem, uniform_mesh
from cfac.rt1_checker import RT1Checker
from cfac.spec import Problem


GRID = tuple(Fraction(k, 200) for k in range(10, 43))
# Declared before running the RT1 constructor. Both load and diffusion are constant.
CASES = (("primary", 2, 1, "galerkin"), ("replication", 3, 1, "galerkin"),
         ("inexact", 2, 1, "zero"), ("null", 2, 0, "zero"))
SCHEMA = "cfac-rt1-fixed-field-v1"


def _freeze(n, load):
    mesh = uniform_mesh(n)
    loads = (load,) * len(mesh.triangles)
    targets = [Problem(mesh, loads, 1, eps, 100) for eps in GRID]
    return [(p, p.digest()) for p in targets]


def _values(items):
    return list(map(str, items))


def _construct(coarse, fine, mode):
    # The producer and discrete optimizers are excluded from receipt replay.
    from cfac.rt1_reconstruction import RT1Reconstructor

    costs = {}
    tic = perf_counter_ns()
    p, anchor = coarse
    system = ReferenceSystem(p)
    costs["coarse_assembly_lift"] = perf_counter_ns() - tic
    tic = perf_counter_ns()
    vstar, qstar = optimal_fields(system)
    v = vstar if mode == "galerkin" else system.zero_primal()
    costs["coarse_discrete_optimizers"] = perf_counter_ns() - tic
    tic = perf_counter_ns()
    seed = check(p, anchor, v, system.feasible_flux)
    request = EnrichmentRequest(p, anchor, seed.state_digest, v, system.feasible_flux, 1)
    candidate = RT1Reconstructor().reconstruct(request)
    candidate.validate_layout(request)
    costs["patch_reconstruction_and_parent_check"] = perf_counter_ns() - tic
    tic = perf_counter_ns()
    fine_system = ReferenceSystem(fine[0])
    vf, qf = optimal_fields(fine_system)
    costs["refined_assembly_lift_and_discrete_optimizers"] = perf_counter_ns() - tic
    fields = {"primal": _values(v), "tree": _values(system.feasible_flux),
              "optimal_rt0": _values(qstar), "refined_primal": _values(vf),
              "refined_rt0": _values(qf), "rt1_face": [_values(x) for x in candidate.face_moments],
              "rt1_cell": [_values(x) for x in candidate.cell_moments]}
    return fields, costs


def _inspect(coarse, fine, fields, mode):
    p, anchor = coarse
    v, tree = fields["primal"], fields["tree"]
    lift = check(p, anchor, v, tree)
    best = assess(p, anchor, v, fields["optimal_rt0"], tight=True)
    if best.floor is None or best.floor.lower != best.floor.upper or best.dual_alg.upper != 0:
        raise ValueError("coarse RT0 optimality or exact floor proof failed")
    if mode == "galerkin" and best.primal_alg.upper != 0:
        raise ValueError("coarse primal fails the Galerkin equations")
    if mode == "zero" and any(Fraction(x) for x in v):
        raise ValueError("nonzero primal in zero-field control")
    request = EnrichmentRequest(p, anchor, lift.state_digest, v, tree, 1)
    proposal = RTMoments(anchor, lift.state_digest, 1, fields["rt1_face"], fields["rt1_cell"])
    retained = proposal.digest()
    high = RT1Checker().verify(request, proposal, retained)
    refined = assess(*fine, fields["refined_primal"], fields["refined_rt0"], tight=True)
    if refined.floor is None or refined.dual_alg.upper != 0 or refined.primal_alg.upper != 0:
        raise ValueError("refined discrete optimizer failed independent residual checks")
    certs = {"tree": lift, "optimal_rt0": best.certificate, "patch_rt1": high,
             "refined_optimal_rt0": refined.certificate}
    for cert in certs.values():
        if cert.status != "CERTIFIED" or not cert.exact_equilibrium or cert.defect_sq != 0:
            raise ValueError("missing exact-equilibrium certificate")
    quantities = {name + "_upper_sq": str(c.upper_sq) for name, c in certs.items()}
    quantities.update(rt0_pair_floor_sq=str(best.floor.lower), primal_alg_sq=str(best.primal_alg.upper),
                      rt1_original_residual_sq=str(high.original_residual_sq))
    row = {"epsilon": str(p.epsilon), "coarse_digest": anchor, "refined_digest": fine[1],
           "rt1_digest": retained, "rt0_floor_verdict": best.verdict,
           "accepted": {name: c.accepted for name, c in certs.items()}}
    row["avoids_refinement"] = (best.floor.lower > p.epsilon**2 and high.accepted
                                and refined.certificate.accepted)
    return quantities, row


def _observations(coarse, fine, fields, mode):
    common, rows = None, []
    for target, control in zip(coarse, fine, strict=True):
        quantities, row = _inspect(target, control, fields, mode)
        if common is not None and common != quantities:
            raise ArithmeticError("varying tolerance changed a squared energy")
        common = quantities
        rows.append(row)
    return {"squared_energies": common, "rows": rows}


def study():
    records = {}
    for name, n, load, mode in CASES:
        start = perf_counter_ns()
        coarse, fine = _freeze(n, load), _freeze(2*n, load)
        setup_ns = perf_counter_ns() - start
        fields, costs = _construct(coarse[0], fine[0], mode)
        tic = perf_counter_ns()
        observations = _observations(coarse, fine, fields, mode)
        costs.update(freeze=setup_ns, checks_at_all_tolerances=perf_counter_ns()-tic)
        records[name] = {"n": n, "load": load, "primal_mode": mode, "fields": fields,
                         **observations, "cost_ns": costs, "elapsed_ns": perf_counter_ns() - start}
    return {"schema": SCHEMA, "cases": records}


def replay(data):
    """Re-check archived fields against the predeclared problems without reconstruction."""
    if data["schema"] != SCHEMA or set(data["cases"]) != {c[0] for c in CASES}:
        raise ValueError("incorrect schema or case inventory")
    for name, n, load, mode in CASES:
        record = data["cases"][name]
        if (record["n"], record["load"], record["primal_mode"]) != (n, load, mode):
            raise ValueError(f"{name}: design mismatch")
        coarse, fine = _freeze(n, load), _freeze(2*n, load)
        observed = _observations(coarse, fine, record["fields"], mode)
        for key, value in observed.items():
            if json.dumps(record[key], sort_keys=True) != json.dumps(value, sort_keys=True):
                raise ValueError(f"{name}: mismatched {key}")
        costs = record["cost_ns"]
        if (set(costs) != {"coarse_assembly_lift", "coarse_discrete_optimizers",
                           "patch_reconstruction_and_parent_check", "refined_assembly_lift_and_discrete_optimizers",
                           "freeze", "checks_at_all_tolerances"}
                or any(type(t) is not int or t < 0 for t in costs.values())
                or type(record["elapsed_ns"]) is not int or sum(costs.values()) > record["elapsed_ns"]):
            raise ValueError(f"{name}: invalid timing record")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--replay", type=Path)
    parser.add_argument("--sha256", help="independently retained digest of the replay file")
    args = parser.parse_args()
    if args.replay:
        raw = args.replay.read_bytes()
        if args.sha256 and sha256(raw).hexdigest() != args.sha256:
            parser.error("receipt digest mismatch")
        data = json.loads(raw)
        replay(data)
    else:
        if args.sha256:
            parser.error("--sha256 requires --replay")
        data = study()
        with args.output.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
    for name, record in data["cases"].items():
        successes = [row["epsilon"] for row in record["rows"] if row["avoids_refinement"]]
        print(name, "refinement avoided at", successes)


if __name__ == "__main__":
    main()
