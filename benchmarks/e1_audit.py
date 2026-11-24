"""Recheck the numerical evidence of a relocated E1 campaign outside its timed endpoint."""
import argparse
import json
import math
from pathlib import Path

from benchmarks.e1 import ARMS, make_request, qualification, workload_definitions
from sciagent.controller import reverify
from sciagent.evidence import sha256_file
from sciagent.spec import sha256_of


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _child(root, relative):
    path = (root / relative).resolve()
    _check(path != root and path.is_relative_to(root), "archive path outside campaign")
    return path


def audit(directory, campaign_sha256, all_trials=False):
    directory = Path(directory).resolve()
    report = directory / "campaign.json"
    _check(sha256_file(report) == campaign_sha256, "campaign digest mismatch")
    data = json.loads(report.read_bytes())
    config_path = directory / "config.json"
    _check(sha256_file(config_path) == data["config_sha256"], "configuration digest mismatch")
    config = json.loads(config_path.read_bytes())
    workloads = workload_definitions(config["scale"])
    _check(config["workloads"] == data["workloads"] and all(w == workloads[k] for k, w in data["workloads"].items()),
           "workload differs from protocol")
    audited = []
    represented = set()
    failed = 0
    for row in data["records"]:
        elapsed = row["elapsed_seconds"]
        _check(math.isfinite(elapsed) and elapsed >= 0, "invalid external timing")
        if not row["eligible"]:
            failed += 1
            continue
        key = (row["workload"], row["arm"])
        if not all_trials and (row["warmup"] or key in represented):
            continue
        w = workloads[row["workload"]]
        req = make_request(w, row["arm"])
        request_digest = sha256_of(req)
        _check(row["state"] == "ACCEPTED" and row["code"] == "OK" and
               row["request_sha256"] == config["request_sha256"][key[0]][key[1]] == request_digest,
               "receipt disagrees with protocol")
        trial = _child(directory, row["directory"])
        archive = _child(trial, row["archive"])
        attempt = archive / "attempts/attempt_01"
        replay = reverify(attempt, expected_digest=row["manifest_sha256"])
        _check(replay.get("integrity", {}).get("passed") is True, "replay failed integrity checks")
        manifest = json.loads((archive / "manifest.json").read_bytes())
        frozen = {key: sha256_of(getattr(req, key)) for key in ("problem", "validation", "policy")}
        eligible, code = qualification(
            manifest, replay, frozen=frozen, binary_sha256=config["binary_sha256"],
            request_matches=sha256_file(archive / "request/study_spec.json") == request_digest, workload=w)
        _check(eligible, f"replayed evidence failed {code}")
        _check(len(replay["levels"]) == len(row["levels"]), "timed workload level count differs")
        for v, captured in zip(replay["levels"], row["levels"], strict=True):
            result = json.loads((attempt / "levels" / f"n{v['n']:04d}" / "result.json").read_bytes())
            _check((v["n"], v["l2_error"], v["h1_semi_error"]) ==
                   (captured["n"], captured["l2_error"], captured["h1_semi_error"]), "archived field metrics differ")
            _check(result["solver"] == ARMS[key[1]] and result["solver_iterations"] == captured["iterations"] and
                   result["time_solve_ms"] / 1000 == captured["solve_seconds"], "native record differs")
        audited.append({"directory": row["directory"], "manifest_sha256": row["manifest_sha256"],
                        "levels": len(replay["levels"])})
        represented.add(key)
    return {"campaign_sha256": campaign_sha256, "config_sha256": data["config_sha256"],
            "campaign_status": data["status"], "mode": "all eligible" if all_trials else "first eligible scored per arm",
            "eligible": len(data["records"]) - failed, "ineligible": failed,
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
