"""Check all E1 inventories and worker receipts, then replay selected numerical studies."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from benchmarks.e1 import ARMS, block_schedule, external_phases, make_request, qualification, workload_definitions
from sciagent.controller import reverify
from sciagent.evidence import inspect_manifest, sha256_file
from sciagent.spec import sha256_of


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _child(root, relative):
    relative = Path(relative)
    _check(not relative.is_absolute() and bool(relative.parts) and ".." not in relative.parts,
           "archive path outside campaign")
    path = root / relative
    _check(path.resolve() == path, "unexpected link inside campaign path")
    return path


def _pinned(path, digest):
    data = path.read_bytes()
    _check(hashlib.sha256(data).hexdigest() == digest, f"{path.stem} digest mismatch")
    return json.loads(data)


def _same_json(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def validate_schedule(data, config):
    workloads = workload_definitions(config["scale"])
    _check(config.get("schema_version") == 1 and config["workloads"] and config["workloads"] == data["workloads"]
           and all(k in workloads and w == workloads[k] for k, w in data["workloads"].items()),
           "workload differs from protocol")
    warmups, repetitions = config["warmups"], config["repetitions"]
    _check(type(warmups) is int and warmups >= 0 and type(repetitions) is int and repetitions > 0
           and type(config["seed"]) is int, "invalid acquisition schedule")
    expected = block_schedule(config["workloads"], repetitions, warmups, config["seed"])
    _check(data["completed_trials"] == len(data["records"]) and
           data["scheduled_trials"] == len(data["workloads"]) * (repetitions + warmups) * len(ARMS),
           "campaign trial counts disagree")
    stopped = set()
    for row in data["records"]:
        slot = next((s for s in expected if (s[0], s[2]) not in stopped), None)
        name, block, arm = row["workload"], row["block"], row["arm"]
        _check(type(block) is int and slot == (name, block, arm) and row["warmup"] is (block < 0)
               and row["directory"] == f"{name}_{'warm' if block < 0 else 'block'}{abs(block):03d}_{arm}",
               "trial assignment differs from registered schedule")
        _check(type(row["eligible"]) is bool and type(row["elapsed_seconds"]) in (float, int)
               and math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0, "invalid timing receipt")
        if row["code"] == "TIMEOUT":
            stopped.add((name, arm))
    status = data["status"]
    _check(status in {"COMPLETE", "BUDGET_EXHAUSTED", "PARTIAL_TIMEOUT"}
           and sorted(stopped) == sorted(tuple(x) for x in data["stopped_arms"]), "unsupported campaign status")
    if status != "BUDGET_EXHAUSTED":
        _check(not any((s[0], s[2]) not in stopped for s in expected)
               and bool(stopped) == (status == "PARTIAL_TIMEOUT"), "incomplete schedule for campaign status")
    if status == "COMPLETE":
        _check(not any(r["code"] == "STORAGE_LIMIT" for r in data["records"]), "unexpected storage stop")


def audit(directory, campaign_sha256, all_trials=False):
    directory = Path(directory).resolve()
    data = _pinned(_child(directory, "campaign.json"), campaign_sha256)
    config = _pinned(_child(directory, "config.json"), data["config_sha256"])
    validate_schedule(data, config)
    journal = _child(directory, "rows.jsonl")
    if data["records"] or journal.exists():
        entries = [json.loads(line) for line in journal.read_bytes().splitlines()]
        _check(_same_json(entries, data["records"]), "acquisition journal differs from pinned campaign")
    audited = []
    represented = set()
    failed = 0
    checked = 0
    for row in data["records"]:
        key = (row["workload"], row["arm"])
        w = config["workloads"][key[0]]
        req = make_request(w, key[1])
        request_digest = sha256_of(req)
        frozen = {k: sha256_of(getattr(req, k)) for k in ("problem", "validation", "policy")}
        trial = _child(directory, row["directory"])
        _check(row["request_sha256"] == config["request_sha256"][key[0]][key[1]] == request_digest ==
               sha256_file(_child(trial, "request.json")), "trial request differs from protocol")
        job = json.loads(_child(trial, "job.json").read_bytes())
        expected_job = {"arm": key[1], "workload": w, "cpu": config["cpu"], "binary_sha256": config["binary_sha256"],
                        "request_sha256": request_digest, "frozen_sha256": frozen}
        _check(_same_json(job, expected_job), "trial job differs from protocol")
        if not row["eligible"]:
            failed += 1
            continue
        _check(row["state"] == "ACCEPTED" and row["code"] == "OK", "unexpected eligible state")
        receipt = json.loads(_child(trial, "worker.json").read_bytes())
        parent_fields = {"arm", "elapsed_seconds", "request_sha256", "external_phases", "workload",
                         "block", "warmup", "directory"}
        _check(_same_json(receipt, {k: v for k, v in row.items() if k not in parent_fields}),
               "worker receipt differs from pinned campaign")
        if "external_phases" in row:
            accounting = row["external_phases"]
            clock = accounting["monotonic_boundaries_seconds"]
            parent = {k: clock[k] for k in ("submitted", "before_spawn", "completed")}
            local = json.loads(_child(trial, "worker-clocks.json").read_bytes())
            _check(accounting["status"] == "MEASURED" and
                   _same_json(external_phases(parent, local, receipt["profile"]), accounting) and
                   parent["completed"] - parent["submitted"] == row["elapsed_seconds"],
                   "external clocks differ from pinned campaign")
        archive = _child(trial, row["archive"])
        _check(archive.parent == trial / "studies" and archive.name.endswith("_request"), "unexpected study location")
        pin = row["manifest_sha256"]
        _check(isinstance(pin, str) and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin),
               "missing or invalid study manifest digest")
        manifest, _, changed = inspect_manifest(archive, pin)
        _check(manifest.get("evidence_version") == 1 and not changed, "study inventory failed integrity checks")
        attempt = archive / "attempts/attempt_01"
        verdict = json.loads((attempt / "verdict.json").read_bytes())
        replay_this = all_trials or (not row["warmup"] and key not in represented)
        if replay_this:
            verdict = reverify(attempt, expected_digest=row["manifest_sha256"])
            _check(verdict.get("integrity", {}).get("passed") is True, "replay failed integrity checks")
        eligible, code = qualification(
            manifest, verdict, frozen=frozen, binary_sha256=config["binary_sha256"],
            request_matches=sha256_file(archive / "request/study_spec.json") == request_digest, workload=w)
        _check(eligible, f"archived numerical verdict failed {code}")
        _check(row["frozen_sha256"] == frozen and row["binary_sha256"] == config["binary_sha256"]
               and row["verifier_verdict"] == verdict["verdict"] and row["failure_codes"] == verdict["failure_codes"],
               "worker decision differs from archived verifier evidence")
        _check(len(verdict["levels"]) == len(row["levels"]), "timed workload level count differs")
        for v, captured in zip(verdict["levels"], row["levels"], strict=True):
            result = json.loads((attempt / "levels" / f"n{v['n']:04d}" / "result.json").read_bytes())
            _check((v["n"], v["l2_error"], v["h1_semi_error"]) ==
                   (captured["n"], captured["l2_error"], captured["h1_semi_error"]), "archived field metrics differ")
            _check(result["solver"] == ARMS[key[1]] and result["solver_iterations"] == captured["iterations"] and
                   result["time_solve_ms"] / 1000 == captured["solve_seconds"] and
                   result["solver_final_norm"] == captured["final_norm"] and
                   result["diagnostics"]["algebraic_residual"]["relative_l2"] == captured["relative_residual"],
                   "native record differs")
        checked += 1
        if replay_this:
            audited.append({"directory": row["directory"], "manifest_sha256": row["manifest_sha256"],
                            "levels": len(verdict["levels"])})
            represented.add(key)
    return {"campaign_sha256": campaign_sha256, "config_sha256": data["config_sha256"],
            "campaign_status": data["status"], "mode": "all eligible" if all_trials else "first eligible scored per arm",
            "eligible": len(data["records"]) - failed, "ineligible": failed, "inventories_checked": checked,
            "replayed": len(audited), "study_archives": audited}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--campaign-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    result = audit(args.directory, args.campaign_sha256, args.all)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
