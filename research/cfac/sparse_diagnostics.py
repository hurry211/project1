"""Measure fixed-primal RT1 corrections and sparse scaling on frozen data."""

import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

from cfac.checker import check
from cfac.controller import _refine
from cfac.enrichment import EnrichmentRequest, RTMoments
from cfac.floor import assess
from cfac.mfem_primal import native_pcg
from cfac.prototype import uniform_mesh
from cfac.rt1_checker import RT1Checker
from cfac.rt1_reconstruction import RT1Reconstructor
from cfac.sparse_control import attribute_rt1
from cfac.sparse_primal import PrimalSystem
from cfac.sparse_rt1 import optimize_rt1
from cfac.spec import Problem
from .futile_limits import missed_fields


def peak():
    return int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines()
                    if line.startswith("VmHWM:")))


def recover(binary, directory):
    comparisons = {}
    for case, entry, p, state in missed_fields():
        parent = EnrichmentRequest(p, p.digest(), state["state_digest"], state["primal"], state["flux"], 1)
        details = {"problem_digest": p.digest(), "journal": entry["journal_id"],
                   "primal_unchanged_digest": parent.parent_state_digest, "epsilon_sq": str(p.epsilon**2),
                   "rt1_candidate_comparisons": []}
        checker = RT1Checker()
        configurations = ("patch", "eager", "geometric", "predicted", "no_coarse", "patch_predicted")
        for block in range(6):
            for i in range(len(configurations)):
                name = configurations[(i + block) % len(configurations)]
                started = perf_counter()
                if name == "patch":
                    proposal = RT1Reconstructor().reconstruct(parent)
                    cert = checker.verify(parent, proposal, proposal.digest())
                    if proposal.digest() != entry["archived_candidate_digest"]:
                        raise ArithmeticError("historical patch output changed")
                    costs = {"iterations": 0, "checks": 1}
                else:
                    seed = RT1Reconstructor().reconstruct(parent) if name == "patch_predicted" else None
                    schedule = name if name in ("eager", "geometric") else "predicted"
                    proposal, cert, costs = optimize_rt1(parent, schedule=schedule, coarse=name != "no_coarse", seed=seed)
                elapsed = perf_counter() - started
                result = dict(block=block, position=i, policy=name, seconds=elapsed, upper_sq=str(cert.upper_sq),
                              accepted=cert.accepted, exact_equilibrium=cert.exact_equilibrium,
                              candidate_digest=proposal.digest(), **costs)
                if block == 0:
                    result["face_moments"] = [list(map(str, row)) for row in proposal.face_moments]
                    result["cell_moments"] = [list(map(str, row)) for row in proposal.cell_moments]
                    result["four_terms"] = attribute_rt1(p, parent.primal, parent.rt0_flux, proposal)
                details["rt1_candidate_comparisons"].append(result)
        timing = perf_counter()
        native, record = native_pcg(p, p.digest(), parent.primal, binary, directory / case / "resume",
                                    iterations=32)
        details["mfem_resume"] = dict(seconds=perf_counter() - timing, **record)
        resumed = check(p, p.digest(), native, parent.rt0_flux)
        request = EnrichmentRequest(p, p.digest(), resumed.state_digest, native, parent.rt0_flux, 1)
        proposed, cert, cost = optimize_rt1(request)
        details["mfem_resume"].update(accepted=cert.accepted, upper_sq=str(cert.upper_sq), **cost)
        details["mfem_resume"]["packet"] = {"primal": list(map(str, native)),
                                            "faces": [list(map(str, row)) for row in proposed.face_moments],
                                            "cells": [list(map(str, row)) for row in proposed.cell_moments],
                                            "candidate_digest": proposed.digest()}
        comparisons[case] = details
    return comparisons


def scale(n, binary, directory):
    mesh = uniform_mesh(n)
    p = Problem(mesh, [1]*len(mesh.triangles), 1, Q(4, 5*n), 128)
    start = perf_counter()
    field, native = native_pcg(p, p.digest(), [0]*len(mesh.vertices), binary, directory / "primal",
                               iterations=4)
    record = {"n": n, "free_p1_dofs": (n-1)**2, "triangles": len(mesh.triangles),
              "native_proposal_seconds": perf_counter() - start, "native": native}
    start = perf_counter()
    system = PrimalSystem(p, p.digest())
    record["sparse_p1_setup_seconds"] = perf_counter() - start
    record["p1_matrix_nonzeros"] = system.matrix.nnz
    start = perf_counter()
    floor = assess(p, p.digest(), field, system.flux, sparse_iterations=6)
    record.update(floor_seconds=perf_counter() - start, floor_verdict=floor.verdict, floor_peak_rss_kib=peak(),
                  primal_alg_sq=[float(floor.primal_alg.lower), float(floor.primal_alg.upper)],
                  dual_alg_sq=[float(floor.dual_alg.lower), float(floor.dual_alg.upper)])
    return record


def replay_recovery(path, retained_digest):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != retained_digest:
        raise ValueError("diagnostic receipt differs from the retained digest")
    receipt = json.loads(raw)["results"]
    cases = list(missed_fields())
    if set(receipt) != {name for name, _, _, _ in cases}:
        raise ValueError("unexpected diagnostic cases")
    checker = RT1Checker()
    for name, source, p, state in cases:
        row = receipt[name]
        if row["problem_digest"] != p.digest() or row["journal"] != source["journal_id"]:
            raise ValueError("diagnostic source does not match the historical field")
        request = EnrichmentRequest(p, p.digest(), state["state_digest"], state["primal"], state["flux"], 1)
        for trial in row["rt1_candidate_comparisons"]:
            if trial["block"] != 0:
                continue
            candidate = RTMoments(p.digest(), request.parent_state_digest, 1,
                                  trial["face_moments"], trial["cell_moments"])
            cert = checker.verify(request, candidate, trial["candidate_digest"])
            if not cert.exact_equilibrium or (cert.accepted, str(cert.upper_sq)) != (trial["accepted"], trial["upper_sq"]):
                raise ValueError("diagnostic RT1 field failed its independent checker")
        resumed = row["mfem_resume"]
        packet = resumed["packet"]
        parent = check(p, p.digest(), packet["primal"], state["flux"])
        request = EnrichmentRequest(p, p.digest(), parent.state_digest, packet["primal"], state["flux"], 1)
        candidate = RTMoments(p.digest(), request.parent_state_digest, 1, packet["faces"], packet["cells"])
        cert = checker.verify(request, candidate, packet["candidate_digest"])
        if not cert.exact_equilibrium or (cert.accepted, str(cert.upper_sq)) != (resumed["accepted"], resumed["upper_sq"]):
            raise ValueError("native resumed field failed its independent checker")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path)
    ap.add_argument("--directory", type=Path)
    ap.add_argument("--scale", type=int)
    ap.add_argument("--replay", type=Path)
    ap.add_argument("--sha256")
    args = ap.parse_args()
    if args.replay:
        replay_recovery(args.replay, args.sha256)
        print("Verified recovered RT1 fields and both native resumed fields")
        return
    if not args.binary or not args.directory:
        ap.error("a native binary and a new output directory are required to capture")
    args.directory.mkdir(parents=True, exist_ok=False)
    rows = scale(args.scale, args.binary, args.directory) if args.scale else recover(args.binary, args.directory)
    receipt = {"affinity": sorted(os.sched_getaffinity(0)), "peak_rss_kib": peak(), "results": rows}
    (args.directory / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
