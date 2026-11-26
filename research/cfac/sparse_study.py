"""Run and independently replay the phase-one sparse CFAC comparison."""

import argparse
import csv
from fractions import Fraction
import gzip
from hashlib import sha256
import json
import lzma
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

from cfac.baselines import run_policy
from cfac.checker import check
from cfac.prototype import uniform_mesh
from cfac.spec import Mesh, Problem
from . import futile_study, stopping_study


DESIGN = Path(__file__).with_name("sparse_design.json")
SCREEN_DESIGN = Path(__file__).with_name("sparse_screen_design.json")
POLICIES = ("fixed", "classical", "cfac", "cfac_reserve", "cfac_retry", "fixed_sparse",
            "classical_patch", "classical_sparse", "cfac_geometric", "cfac_no_coarse", "cfac_sparse")
LEGACY = frozenset(POLICIES[:5])


def read_receipt(path, retained_digest=None):
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if retained_digest is not None and digest != retained_digest:
        raise ValueError("receipt differs from the retained digest")
    unpacked = lzma.decompress(raw) if path.suffix == ".xz" else gzip.decompress(raw)
    return json.loads(unpacked), digest


def transfer_case(case):
    n = case["n"]
    loads = []
    for j in range(n):
        for i in range(n):
            if case["load"] == "alternating":
                pair = (2, -1) if (i + j) % 2 else (-1, 2)
            elif case["load"] == "notch":
                pair = (3, 1) if i < 2 and j != 1 else (-1, -1)
            elif case["load"] == "pair":
                pair = (4, 4) if (i, j) == (1, 3) else (-3, -2) if (i, j) == (4, 0) else (0, 0)
            elif case["load"] == "islands":
                pair = (4, 2) if (i, j) in {(0, 4), (3, 1), (4, 3)} else (0, -1)
            elif case["load"] == "ring":
                pair = (2, 2) if max(i, j) == 3 and i != 0 else (-1, 0)
            elif case["load"] == "quadrants":
                pair = (4, 1) if i < 3 and j < 3 else (0, -2)
            elif case["load"] == "cross":
                pair = (-4, -3) if i == 2 else (3, 4) if j == 4 else (0, 0)
            elif case["load"] == "staggered":
                pair = (5, 0) if i in (1, 4) and j in (1, 4) else (0, -1)
            else:
                raise ValueError("unknown frozen load")
            loads.extend(pair)
    return dict(case, loads=loads, max_actions=128)


def serialize(result):
    from cfac.controller import _problem_record
    from cfac.sparse_control import _request

    p, field, flux, rt1 = (result.pop(k) for k in ("problem", "field", "rt0", "rt1"))
    request = _request(p, field, flux) if rt1 is not None else None
    digest = request.parent_state_digest if request else check(p, p.digest(), field, flux).state_digest
    result["delivery"] = {"problem": _problem_record(p), "problem_digest": p.digest(),
                          "primal": list(map(str, field)), "flux": list(map(str, flux)), "parent_digest": digest}
    if rt1 is not None:
        result["delivery"].update(rt1_digest=rt1.digest(),
                                  faces=[list(map(str, v)) for v in rt1.face_moments],
                                  cells=[list(map(str, v)) for v in rt1.cell_moments])


def worker(case, policy, budget):
    problem = stopping_study.problem_for(case)
    started = perf_counter()
    if policy in LEGACY:
        journal = run_policy(problem, problem.digest(), policy=policy)
        packet = stopping_study.delivery(journal)
        result = {"status": packet["status"], "accepted": packet["accepted"],
                  "delivery": packet if packet["accepted"] else None,
                  "actions": [e["action"] for e in journal["trace"]]}
    else:
        from cfac.sparse_control import run_sparse

        result = run_sparse(problem, problem.digest(), policy=policy, iterations=budget["iterations_per_stream"],
                            refinements=budget["child_refinements"])
        serialize(result)
    peak = next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines()
                if line.startswith("VmHWM:"))
    result.update(worker_seconds=perf_counter() - started, peak_rss_kib=int(peak))
    return result


def verify_trial(case, policy, trial):
    if trial["status"] in ("TIMEOUT", "WORKER_ERROR"):
        if trial["accepted"]:
            raise ValueError("a failed worker cannot deliver an accepted solution")
        return
    packet = trial["delivery"]
    if policy in LEGACY:
        if packet is None:
            return
        from cfac.controller import _refine

        p = stopping_study.problem_for(case)
        data = packet["problem"]
        child = Problem(Mesh(data["vertices"], data["triangles"]), data["loads"], data["diffusion"],
                        data["epsilon"], data["max_actions"], data["original_affine"])
        if p != child and _refine(p, (Fraction(0),)*len(p.mesh.vertices))[0] != child:
            raise ValueError("legacy field differs from the study PDE")
        cert = stopping_study._certificate(child, packet["state"])
        if not cert.accepted or not trial["accepted"]:
            raise ValueError("legacy accepted field failed independent replay")
        return
    from cfac.controller import _refine
    from cfac.enrichment import EnrichmentRequest, RTMoments
    from cfac.rt1_checker import RT1Checker

    p = stopping_study.problem_for(case)
    if p.digest() != packet["problem_digest"]:
        p = _refine(p, (Fraction(0),)*len(p.mesh.vertices))[0]
    if p.digest() != packet["problem_digest"]:
        raise ValueError("delivered state differs from frozen mesh sequence")
    parent = check(p, p.digest(), packet["primal"], packet["flux"])
    if packet["parent_digest"] != parent.state_digest:
        raise ValueError("bad P1/RT0 state binding")
    cert = parent
    if "rt1_digest" in packet:
        request = EnrichmentRequest(p, p.digest(), parent.state_digest, packet["primal"], packet["flux"], 1)
        candidate = RTMoments(p.digest(), request.parent_state_digest, 1, packet["faces"], packet["cells"])
        cert = RT1Checker().verify(request, candidate, packet["rt1_digest"])
    if cert.status != "CERTIFIED" or (cert.accepted, str(cert.upper_sq)) != (trial["accepted"], trial["upper_sq"]):
        raise ValueError("independent check differs from delivered certificate")


def tickets(design, cohort):
    if cohort == "development":
        cases = stopping_study.design(futile_study.PROTOCOL)["cases"]
        repetitions = design["budget"]["paired_blocks"]
    else:
        cases = [transfer_case(case) for case in design["held_out"]]
        repetitions = design["budget"]["holdout_blocks"]
    policies = POLICIES
    if cohort == "screen":
        policies = ("classical", "fixed_sparse", "classical_sparse", "cfac_sparse", "cfac_screened")
    for case, block, position, policy in stopping_study.schedule(
            {"cases": cases, "policies": policies, "blocks": repetitions}):
        yield case, block, position, policy


def collect(case, policy, budget):
    command = [sys.executable, "-m", __name__.replace("__main__", "research.cfac.sparse_study"), "--worker"]
    start = perf_counter()
    try:
        child = subprocess.run(command, input=json.dumps([case, policy, budget]), capture_output=True, text=True,
                               timeout=budget["timeout_s"])
        if child.returncode == 0:
            result = json.loads(child.stdout)
        else:
            result = {"status": "WORKER_ERROR", "accepted": False, "stderr": child.stderr[-3000:]}
    except subprocess.TimeoutExpired:
        result = {"status": "TIMEOUT", "accepted": False}
    result["seconds"] = perf_counter() - start
    return result


def cpu_model():
    return next(row.split(":", 1)[1].strip() for row in Path("/proc/cpuinfo").read_text().splitlines()
                if row.startswith("model name"))


def write_csv(path, rows):
    columns = ("case", "block", "position", "policy", "status", "accepted", "seconds", "peak_rss_kib",
               "upper_sq", "primal_iterations", "rt1_iterations", "levels", "certificate_checks",
               "stopping_checks", "floor_checks", "flux_reconstructions")
    with path.open("x", encoding="utf-8", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        table.writeheader()
        for row in rows:
            work = row.get("work", {})
            table.writerow({**row, **work, "levels": len(work["levels"]) if work else None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cohort", choices=("development", "transfer", "screen"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--sha256", help="externally retained digest of the archived receipt")
    parser.add_argument("--resume", action="store_true", help="validate and resume an incomplete journal")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(*json.load(sys.stdin)), allow_nan=False))
        return
    receipt = None
    if args.replay:
        if args.sha256 is None:
            raise ValueError("an externally retained digest is required for replay")
        receipt, _ = read_receipt(args.replay, args.sha256)
    cohort = receipt["cohort"] if receipt else args.cohort
    raw = (SCREEN_DESIGN if cohort == "screen" else DESIGN).read_bytes()
    digest = sha256(raw).hexdigest()
    design = json.loads(raw)
    if args.replay:
        if receipt["design_sha256"] != digest:
            raise ValueError("receipt uses a different design")
        for row, (case, block, position, policy) in zip(receipt["rows"], tickets(design, receipt["cohort"]), strict=True):
            if (row["case"], row["block"], row["position"], row["policy"]) != (case["id"], block, position, policy):
                raise ValueError("receipt contains unexpected comparisons")
            verify_trial(case, policy, row)
        print(f"Verified {len(receipt['rows'])} submissions")
        return
    if args.cohort is None or args.output is None:
        parser.error("select a cohort and output directory, or replay")
    journal_path = args.output / "rows.jsonl"
    receipt_path = args.output / "receipt.json.gz"
    plan = list(tickets(design, args.cohort))
    if args.resume:
        if receipt_path.exists():
            raise ValueError("use replay on a complete campaign")
        rows = [json.loads(line) for line in journal_path.read_bytes().splitlines()]
        for row, (case, block, position, policy) in zip(rows, plan[:len(rows)], strict=True):
            if (row["case"], row["block"], row["position"], row["policy"]) != (case["id"], block, position, policy):
                raise ValueError("cannot resume an unrelated campaign")
            verify_trial(case, policy, row)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        rows = []
    with journal_path.open("a" if args.resume else "x", encoding="utf-8") as journal:
        for case, block, position, policy in plan[len(rows):]:
            row = dict(case=case["id"], block=block, position=position, policy=policy,
                       **collect(case, policy, design["budget"]))
            rows.append(row)
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            print(case["id"], block, policy, row["status"], flush=True)
    receipt = {"cohort": args.cohort, "design_sha256": digest, "cpu": cpu_model(),
               "platform": platform.platform(), "python": platform.python_version(),
               "affinity": sorted(os.sched_getaffinity(0)), "rows": rows}
    packed = gzip.compress(json.dumps(receipt).encode(), mtime=0)
    with receipt_path.open("xb") as output:
        output.write(packed)
    write_csv(args.output / "trials.csv", rows)
    print(sha256(packed).hexdigest())


if __name__ == "__main__":
    main()
