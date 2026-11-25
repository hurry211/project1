"""Audit completed paired-revision studies from externally pinned design and report files."""
import argparse
import json
from pathlib import Path
import random
from statistics import median

from benchmarks import e1
from benchmarks.e1_analysis import _interval
from benchmarks.e1_audit import _check, _child, _pinned, _same_json
from benchmarks.verified_revision import digest_json, ordered_pairs
from sciagent.controller import reverify
from sciagent.evidence import inspect_manifest, sha256_file
from sciagent.spec import sha256_of


def _schedule(design):
    workloads = design["workloads"]
    _check(bool(workloads) and any(all(e1.workload_definitions(scale).get(name) == work
                                       for name, work in workloads.items()) for scale in ("memo", "compact")),
           "unknown workload specification")
    orders = design["orders_including_warmup"]
    _check(set(orders) == set(workloads), "schedule workloads differ")
    blocks = len(orders[next(iter(workloads))]) - 1
    _check(blocks >= 2, "insufficient paired blocks")
    rng = random.Random(design["seed"])
    for name in workloads:
        expected = [next(ordered_pairs(2, rng)), *ordered_pairs(blocks, rng)]
        _check(orders[name] == expected, "order differs from seeded schedule")
        for block, order in enumerate(expected, start=-1):
            for arm in order:
                yield name, block, arm


def _check_receipts(folder, row):
    trial = json.loads(_child(folder, "trial.json").read_bytes())
    worker = json.loads(_child(folder, "worker.json").read_bytes())
    _check(_same_json(trial, {**worker, "arm": "gs", "elapsed_seconds": row["elapsed_seconds"],
                              "request_sha256": row["request_sha256"], "external_phases": trial["external_phases"]}),
           "trial and worker receipts disagree")
    fields = ("state", "code", "eligible", "manifest_sha256", "archive", "profile", "elapsed_seconds", "request_sha256")
    _check(_same_json([trial[k] for k in fields], [row[k] for k in fields]), "report differs from receipt")
    accounting = trial["external_phases"]
    parent = {k: accounting["monotonic_boundaries_seconds"][k] for k in ("submitted", "before_spawn", "completed")}
    clocks = json.loads(_child(folder, "worker-clocks.json").read_bytes())
    _check(accounting["status"] == "MEASURED" and
           _same_json(e1.external_phases(parent, clocks, worker["profile"]), accounting) and
           parent["completed"] - parent["submitted"] == row["elapsed_seconds"], "endpoint timing differs")
    return worker


def _science(archive, manifest, actual, saved, worker, replay):
    relative = Path("attempts/attempt_01")
    attempt = archive / relative
    # run_dir and the saved level paths were written at acquisition. Compare
    # against that root first, so unpacking the archive elsewhere is supported.
    captured = Path(manifest["run_dir"]) / relative
    _check(saved["verdict"] == worker["verifier_verdict"] == "ACCEPTED" and
           saved["failure_codes"] == worker["failure_codes"] == [], "verifier receipt differs")
    if replay:
        recomputed = reverify(attempt, expected_digest=worker["manifest_sha256"])
        _check(recomputed.pop("integrity")["passed"] is True, "independent replay failed")
        for level in recomputed["levels"]:
            subdir = Path("levels") / f"n{level['n']:04d}"
            _check(Path(level["level_dir"]) == attempt / subdir, "unexpected replay level path")
            level["level_dir"] = str(captured / subdir)
        _check(_same_json(saved, recomputed), "complete replay verdict differs")
    expected_inputs = {"frozen/problem.json", "frozen/validation.json", "execution.json"}
    pins = saved["input_sha256"]
    for level, summary, native in zip(saved["levels"], worker["levels"], worker["profile"]["native_levels"], strict=True):
        n = level["n"]
        subdir = Path("levels") / f"n{n:04d}"
        _check(Path(level["level_dir"]) == captured / subdir, "unexpected captured level path")
        level["level_dir"] = subdir.as_posix()
        names = ("dof_values.txt", "element_dofs.txt", "elements.txt", "boundary.txt", "result.json")
        expected_inputs.update((subdir / fn).as_posix() for fn in names)
        for fn in names:
            key = (subdir / fn).as_posix()
            _check(pins[key] == actual[(relative / key).as_posix()], "level digest differs")
        result_key = (subdir / "result.json").as_posix()
        raw = json.loads((attempt / result_key).read_bytes())
        expected = {"n": n, "iterations": raw["solver_iterations"], "final_norm": raw["solver_final_norm"],
                    "relative_residual": raw["diagnostics"]["algebraic_residual"]["relative_l2"],
                    "solve_seconds": raw["time_solve_ms"] / 1000, "native_wall_seconds": native["process_seconds"],
                    "l2_error": level["l2_error"], "h1_semi_error": level["h1_semi_error"]}
        _check(native["n"] == n and raw["solver"] == e1.ARMS["gs"] and _same_json(expected, summary),
               "native/field receipt differs")
        for kind in ("setup", "solve", "error"):
            _check(raw.pop(f"time_{kind}_ms") / 1000 == native[f"{kind}_seconds"], "native timer differs")
        for key in ("time_rhs_ms", "time_assembly_ms", "time_preconditioner_ms", "time_cg_ms", "time_export_ms", "peak_rss_kib"):
            raw.pop(key, None)
        pins[result_key] = digest_json(raw)
    _check(set(pins) == expected_inputs and
           all(pins[k] == actual[k] for k in ("frozen/problem.json", "frozen/validation.json")) and
           pins["execution.json"] == actual[(relative / "execution.json").as_posix()], "verifier input set differs")
    return digest_json(saved)


def audit(directory, design_sha256, report_sha256):
    directory = directory.resolve(strict=True)
    design = _pinned(_child(directory, "design.json"), design_sha256)
    report = _pinned(_child(directory, "report.json"), report_sha256)
    journal = _child(directory, "rows.jsonl")
    rows = [json.loads(line) for line in journal.read_bytes().splitlines()]
    _check(report["design_sha256"] == design_sha256 and sha256_file(journal) == report["journal_sha256"] and
           _same_json(rows, report["records"]), "report, journal or design differs")
    times = {w: {"baseline": [], "candidate": []} for w in design["workloads"]}
    fingerprints, replayed = {}, []
    for row, (name, block, arm) in zip(rows, _schedule(design), strict=True):
        label = f"{name}_{block:03d}_{arm}"
        _check(_same_json([row[k] for k in ("workload", "block", "revision", "directory")], [name, block, arm, label]),
               "row differs from seeded schedule")
        folder = _child(directory, label)
        worker = _check_receipts(folder, row)
        work = design["workloads"][name]
        request = e1.make_request(work, "gs")
        frozen = {k: sha256_of(getattr(request, k)) for k in ("problem", "validation", "policy")}
        expected = {"arm": "gs", "workload": work, "cpu": design["cpu"], "binary_sha256": design["binary_sha256"],
                    "request_sha256": sha256_of(request), "frozen_sha256": frozen}
        _check(_same_json(json.loads(_child(folder, "job.json").read_bytes()), expected) and
               sha256_file(_child(folder, "request.json")) == row["request_sha256"] == expected["request_sha256"] ==
               design["request_sha256"][name], "job or request differs from design")
        archive = _child(folder, row["archive"])
        _check(archive.parent == folder / "studies" and archive.name.endswith("_request"), "unexpected archive path")
        manifest, actual, changed = inspect_manifest(archive, row["manifest_sha256"])
        _check(manifest.get("evidence_version") == 1 and not changed, "manifest inventory differs")
        verdict = json.loads((archive / "attempts/attempt_01/verdict.json").read_bytes())
        eligible, code = e1.qualification(manifest, verdict, frozen=frozen, binary_sha256=expected["binary_sha256"],
                                          request_matches=actual["request/study_spec.json"] == row["request_sha256"],
                                          workload=work)
        _check(eligible and eligible is row["eligible"] and code == row["code"] and row["state"] == "ACCEPTED"
               and worker["binary_sha256"] == expected["binary_sha256"] and worker["frozen_sha256"] == frozen,
               "study failed qualification")
        digest = _science(archive, manifest, actual, verdict, worker, replay=block == 0)
        _check(digest == row["numerical_sha256"] == fingerprints.setdefault(name, digest), "numerical evidence differs")
        if block == 0:
            replayed.append({"directory": label, "manifest_sha256": row["manifest_sha256"]})
        if block >= 0:
            times[name][arm].append(row["elapsed_seconds"])
    scores = {}
    for name, timed in times.items():
        delta = [c - b for c, b in zip(timed["candidate"], timed["baseline"], strict=True)]
        scores[name] = {"median_seconds": {arm: median(v) for arm, v in timed.items()},
                        "median_paired_delta_seconds": median(delta), "paired_95pct_seconds": _interval(delta)}
    _check(_same_json(scores, report["scores"]), "published paired summary differs")
    return {"design_sha256": design_sha256, "report_sha256": report_sha256, "journal_sha256": report["journal_sha256"],
            "inventories_checked": len(rows), "replayed": replayed, "numerical_sha256": fingerprints, "scores": scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--design-sha256", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.design_sha256, args.report_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
