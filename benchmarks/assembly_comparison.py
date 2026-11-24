"""Compare quadrilateral full and partial assembly with an identical Jacobi smoother."""

import argparse
import json
import os
from pathlib import Path
import random
from statistics import median
import subprocess

import numpy as np

from benchmarks import e1, stage2
from benchmarks.e1_analysis import _interval
from sciagent import capability, controller
from sciagent.evidence import sha256_file
from sciagent.mfem_runner import solver_binary
from sciagent.spec import StudySpec, sha256_of


def matrix(directory):
    """Certify field, topology, norms and rates against regenerated scikit-fem meshes."""
    rows = []
    for case in stage2.CASES:
        for source in capability.MESH_SOURCES:
            for p in (1, 2):
                for assembly in ("full", "partial"):
                    key = ("quad", p, assembly, "cg_jacobi")
                    parent = directory / f"{case}_{source}_q{p}_{assembly}"
                    parent.mkdir(parents=True)
                    path = stage2.case_request(case, source, key, parent)
                    manifest = controller.run_study(path, parent / "studies")
                    row = stage2.certify(Path(manifest["run_dir"]), case, source, key, None)
                    rows.append(row)
                    attempt = Path(manifest["run_dir"]) / "attempts/attempt_01/levels"
                    if assembly == "full":
                        full = attempt
                    else:
                        row["paired_with_full"] = matched_fields(full, attempt)
    return rows


def matched_fields(full, partial):
    results = []
    for n in (8, 16, 32):
        a, b = (path / f"n{n:04d}" for path in (full, partial))
        for filename in ("elements.txt", "element_dofs.txt", "boundary.txt"):
            stage2.check(sha256_file(a / filename) == sha256_file(b / filename), "paired mesh or DOFs differ")
        x, y = (np.loadtxt(path / "dof_values.txt") for path in (a, b))
        stage2.check(np.array_equal(x[:, 0], y[:, 0]), "field numbering differs")
        gap = float(np.max(np.abs(x[:, 1] - y[:, 1])) / np.max(np.abs(x[:, 1])))
        stage2.check(gap <= 1e-7, "assembly variants disagree on the nodal solution")
        iterations = [json.loads((path / "result.json").read_bytes())["solver_iterations"] for path in (a, b)]
        results.append({"n": n, "relative_field_gap": gap, "full_partial_iterations": iterations})
    return results


def diagonal_check(binary):
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    gaps = {}
    for line in result.stdout.splitlines():
        p, variable, partial, raw = line.split()
        gaps[f"q{p}_variable{variable}_partial{partial}"] = float(raw)
    expected = {f"q{p}_variable{v}_partial{a}" for p in (1, 2) for v in (0, 1) for a in (0, 1)}
    stage2.check(set(gaps) == expected and all(0 <= v <= 1e-12 for v in gaps.values()),
                 "Jacobi action differs from the independently extracted matrix inverse diagonal")
    return {"probe_sha256": sha256_file(binary), "relative_max_norm_by_case": gaps}


def timing(directory, p, cpu, repetitions, seed, binary_hash):
    historical = json.loads((e1.ROOT / capability.REPORT).read_bytes())["rows"]
    control = historical[f"cos_exp_cartesian_quad_{p}_partial_cg_jacobi"]
    workload = {"case": "cos_exp", "levels": [8, 16, 32], "l2_target": 1.05 * control["finest_l2"],
                "h1_target": 1.05 * control["finest_h1_semi"]}
    rng = random.Random(seed + p)
    samples, requests = [], {}
    for asm in ("full", "partial"):
        parent = directory / "requests" / f"q{p}_{asm}"
        parent.mkdir(parents=True)
        path = stage2.case_request("cos_exp", "cartesian", ("quad", p, asm, "cg_jacobi"), parent)
        requests[asm] = StudySpec.model_validate_json(path.read_bytes())
    a, b = [study.model_dump() for study in requests.values()]
    a["execution"]["assembly"] = b["execution"]["assembly"]
    stage2.check(a == b, "paired requests differ outside the assembly choice")
    for block in range(-1, repetitions):
        order = ["full", "partial"]
        rng.shuffle(order)
        for asm in order:
            study = requests[asm]
            parent = directory / "timing" / f"q{p}_b{block}_{asm}"
            result = e1.run_trial(parent, workload, asm, binary_sha256=binary_hash, cpu=cpu, timeout=900,
                                  study=study, expected_request_sha256=sha256_of(study))
            row = {"block": block, "assembly": asm, "eligible": result["eligible"], "code": result["code"],
                   "external_seconds": result["elapsed_seconds"]}
            if result["eligible"]:
                profile = result["profile"]
                row.update(manifest_sha256=result["manifest_sha256"], archive=result["archive"],
                           worker_peak_rss_kib=profile["worker_peak_rss_kib"],
                           max_native_peak_rss_kib=max(x["peak_rss_kib"] for x in profile["native_levels"]),
                           solve_seconds=sum(x["solve_seconds"] for x in profile["native_levels"]),
                           verify_seconds=sum(t for k, t in profile["phase_seconds"].items() if k.startswith("verify.")),
                           iterations=[level["iterations"] for level in result["levels"]])
            samples.append(row)
    # Audit after acquisition so the extra oracle and replay never enter just one arm's timing.
    for s in samples:
        if s["eligible"]:
            parent = directory / "timing" / f"q{p}_b{s['block']}_{s['assembly']}"
            archive = parent / s["archive"]
            replay = controller.reverify(archive / "attempts/attempt_01", expected_digest=s["manifest_sha256"])
            stage2.check(replay["integrity"]["passed"] and replay["verdict"] == "ACCEPTED",
                         "digest-pinned independent replay failed")
    paired = {}
    for s in samples:
        if s["block"] >= 0 and s["eligible"]:
            paired.setdefault(s["block"], {})[s["assembly"]] = s
    diffs = [pair["partial"]["external_seconds"] - pair["full"]["external_seconds"]
             for pair in paired.values() if len(pair) == 2]
    # Keep failures and their charged times visible. Report a paired timing result only for a complete panel.
    complete = all(s["eligible"] for s in samples)
    medians = {}
    if complete:
        for arm in ("full", "partial"):
            rows = [s for s in samples if s["assembly"] == arm and s["block"] >= 0]
            medians[arm] = {k: median(s[k] for s in rows) for k in
                            ("external_seconds", "solve_seconds", "verify_seconds", "worker_peak_rss_kib",
                             "max_native_peak_rss_kib")}
    return {"workload": workload, "request_sha256": {k: sha256_of(s) for k, s in requests.items()},
            "complete": complete, "samples": samples, "medians": medians,
            "paired_partial_minus_full_median_seconds": median(diffs) if complete else None,
            "paired_bootstrap_95_interval_seconds": _interval(diffs) if complete else None}


def run(directory, cpu, repetitions, seed, probe):
    if cpu not in os.sched_getaffinity(0) or repetitions < 2:
        raise ValueError("select an available CPU and at least two paired repetitions")
    directory.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {cpu})
    binary_hash = sha256_file(solver_binary())
    report = {"cpu": cpu, "seed": seed, "repetitions": repetitions, "warmup_pairs": 1,
              "solver_binary_sha256": binary_hash, "diagonal": diagonal_check(probe),
              "coverage": matrix(directory / "coverage")}
    report["timing"] = {f"q{p}": timing(directory, p, cpu, repetitions, seed, binary_hash) for p in (1, 2)}
    e1.write_json(directory / "report.json", report)
    return all(r["complete"] for r in report["timing"].values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--diagonal-probe", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.directory, args.cpu, args.repetitions, args.seed, args.diagonal_probe) else 1)
