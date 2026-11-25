"""Authenticate a retained adaptive campaign, reproduce pairing and replay its numerical evidence."""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmarks.e3_adaptive import STRATEGIES, build_held_out_tasks, paired_results
from sciagent.evidence import artifact_hashes, sha256_file
from sciagent.spec import ProblemSpec, ValidationSpec
from verifier.amr import verify_amr_run
from verifier.verifier import verify_attempt


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(data):
    return json.dumps(data, allow_nan=False, sort_keys=True)


def verify_field(directory):
    task = json.loads((directory / "task.json").read_bytes())
    previous = json.loads((directory / "verdict.json").read_bytes())
    if task["candidate"]["mesh_topology"] == "amr":
        current = verify_amr_run(directory / "amr_run", ProblemSpec.model_validate(task["problem"]),
                                 ValidationSpec.model_validate(task["validation"]))
    else:
        attempt = directory / "run/attempts/attempt_01"
        current = verify_attempt(attempt, require_run_records=True)
        for result in (previous, current):
            for level in result["levels"]:
                key = Path(level["level_dir"])
                expected = Path("levels") / f"n{level['n']:04d}"
                _require(key.parts[-2:] == expected.parts, "unexpected level path")
                level["level_dir"] = expected.as_posix()
    _require(_json(previous) == _json(current), f"numerical replay differs at {directory.name}")
    return current["verdict"]


def audit(report_path, pin, raw, *, all_fields=False, workers=1):
    _require(sha256_file(report_path) == pin, "report digest differs")
    report = json.loads(report_path.read_bytes())
    design = report["design"]
    _require(sha256_file(raw / "design.json") == report["design_sha256"]
             and design == json.loads((raw / "design.json").read_bytes()), "design differs")
    _require(sha256_file(raw / "inventory.json") == report["inventory_sha256"], "inventory digest differs")
    expected = json.loads((raw / "inventory.json").read_bytes())
    inventory = artifact_hashes(raw)
    inventory.pop("inventory.json")
    _require(expected == inventory and len(expected) == report["raw_files"], "retained files differ")
    tasks = build_held_out_tasks()
    rng = np.random.default_rng(design["seed"])
    schedule = []
    for rep in range(design["repeats"]):
        for task in tasks:
            arms = list(STRATEGIES)
            rng.shuffle(arms)
            schedule.extend((rep, task, arm, position) for position, arm in enumerate(arms))
    rows = report["trials"]
    replayed = Counter()
    for row, (rep, task, arm, pos) in zip(rows, schedule, strict=True):
        label = f"rep{rep}_{task.task_id}_{arm}"
        _require((row["repeat"], row["task_id"], row["strategy"], row["order_position"], row["directory"]) ==
                 (rep, task.task_id, arm, pos, label), "schedule differs")
        receipt = json.loads((raw / f"{label}.json").read_bytes())
        for key in ("accepted", "attempts", "peak_rss_mb", "external_wall_seconds"):
            _require(receipt[key] == row[key], "trial receipt differs")
        _require(row["total_end_to_end_seconds"] == receipt["external_wall_seconds"]
                 and row["target_l2"] == task.validation.max_l2_error_finest, "endpoint/target differs")
        if not row["attempts"]:
            _require(not row["accepted"], "missing accepted study")
            continue
        for i, attempt in enumerate(row["attempts"], 1):
            directory = raw / label / f"try_{i:02d}_{attempt['candidate_id']}"
            frozen = json.loads((directory / "task.json").read_bytes())
            _require(frozen["task_id"] == task.task_id and frozen["problem"] == task.problem.model_dump(mode="json")
                     and frozen["policy"] == task.policy.model_dump(mode="json")
                     and frozen["validation"] == task.validation.model_dump(mode="json"), "task contract differs")
            verdict = json.loads((directory / "verdict.json").read_bytes())
            levels = verdict.get("levels", verdict.get("steps"))
            if levels:
                _require(levels[-1]["l2_error"] == attempt["l2_error"], "finest error differs")
            if attempt["ok"]:
                _require(verdict["verdict"] == "ACCEPTED" and attempt["l2_error"] <= row["target_l2"]
                         and attempt["finest_dofs"] <= task.policy.max_dofs_per_level, "unqualified accepted field")
        _require(row["accepted"] == row["attempts"][-1]["ok"], "terminal state differs")
        if not all_fields and rep == 0 and arm in ("fixed_baseline", "ai_assisted"):
            replayed[verify_field(directory)] += 1
    if all_fields:
        directories = [path.parent for path in sorted(raw.rglob("task.json"))]
        if workers == 1:
            replayed.update(map(verify_field, directories))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                replayed.update(pool.map(verify_field, directories))
    _require(report["paired_speedups"] == paired_results(rows), "matched-pair scores differ")
    for arm in STRATEGIES:
        count = sum(r["accepted"] for r in rows if r["strategy"] == arm)
        _require(count == report["strategy_summary"][arm]["accepted_count"], "strategy count differs")
    return {"report_sha256": pin, "inventory_sha256": report["inventory_sha256"], "verified_files": len(expected),
            "trial_receipts_checked": len(rows), "full_verdict_replays": dict(replayed),
            "mode": "all" if all_fields else "first scored fixed and model result on each task"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    print(json.dumps(audit(args.report, args.report_sha256, args.raw, all_fields=args.all, workers=args.workers), indent=2))
