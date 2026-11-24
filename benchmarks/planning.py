"""Paired trial of local model selection and the deterministic selection rule."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from sciagent import planner

DATA = Path(__file__).with_name("data")
TASK_SHA = "37ee1eeab7bea6f17640a9e1f0d6b75316d87072cf510bc389317399286519e0"
CONTROL_SHA = "ad939acbaa41608631721e159e72636670e0be618ccd11191c119adbc1112018"
MODES = ("rules", "llm")
FIELDS = ("state", "code", "template_id", "mode", "model", "model_digest", "task_sha256", "child_manifest_sha256", "proposal_calls",
          "proposal_retries", "model_calls", "numerical_retries", "usage", "manifest_sha256")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pinned_json(path, digest):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, f"digest mismatch for {path}")
    return json.loads(raw)


def protocol():
    tasks = pinned_json(DATA / "planning_tasks.json", TASK_SHA)["tasks"]
    controls = pinned_json(DATA / "planning_gate_controls.json", CONTROL_SHA)
    lookup = {row["id"]: row for row in tasks}
    checks = []
    for item in controls:
        task = planner.PlanningTask.model_validate(lookup[item["task_id"]]["task"])
        outcome = planner.decide(task, json.dumps(item["candidate"]))
        checks.append({"task_id": item["task_id"], "code": outcome["code"], "state": outcome["status"],
                       "passed": outcome["status"] == item["expected_state"] and outcome["template_id"] is None})
    return tasks, checks


def score(tasks, trials):
    """Count successful verification, preference fulfillment and safety separately."""
    gold = {row["id"]: row for row in tasks}
    require(set(gold) == {r["id"] for r in trials} and len(gold) == len(trials), "incomplete trial panel")
    complete = correct = review_ok = false_accepts = false_rejects = 0
    review_accepted = preference_hits = preference_wrong = 0
    states = Counter()
    for r in trials:
        row = gold[r["id"]]
        state, selected = r["state"], r["template_id"]
        eligible = row["expected_state"] == "ACCEPTED"
        accepted = state == "ACCEPTED"
        authorized = accepted and selected in row["permitted_ids"]
        hit = authorized and selected == row["expected_template_id"]
        complete += authorized
        correct += hit
        review_ok += not eligible and state == "REVIEW_REQUIRED"
        review_accepted += not eligible and accepted
        false_accepts += accepted and not authorized
        false_rejects += eligible and state in ("REVIEW_REQUIRED", "REJECTED")
        preference_hits += row["group"] == "preference" and hit
        preference_wrong += authorized and not hit
        states[state] += 1
    n_eligible = sum(r["expected_state"] == "ACCEPTED" for r in tasks)
    n_review = len(tasks) - n_eligible
    n_preference = sum(r["group"] == "preference" for r in tasks)
    for r in trials:
        for name in ("wall_seconds", "external_replay_seconds"):
            seconds = r[name]
            require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
                    f"invalid measured {name}")
    wall = [r["wall_seconds"] for r in trials]
    tokens = {}
    active = [r for r in trials if r["model_calls"] > 0]
    for key in ("prompt_tokens", "completion_tokens"):
        reported = [r["usage"].get(key) for r in active]
        available = [x for x in reported if x is not None]
        require(all(type(t) is int and t >= 0 for t in available), "invalid measured token count")
        tokens[key] = {"known_sum": sum(available), "unreported_trials": len(active) - len(available),
                       "total": sum(available) if len(available) == len(active) else None}
    return {
        "all_tasks": len(tasks), "eligible_tasks": n_eligible, "review_controls": n_review,
        "preference_tasks": n_preference, "states": dict(sorted(states.items())),
        "authorized_verified_completions": complete, "intended_verified_completions": correct,
        "correct_review_holds": review_ok, "correct_actions": correct + review_ok,
        "preference_hits": preference_hits, "authorized_wrong_preference": preference_wrong,
        "unauthorized_acceptances_all_tasks": false_accepts, "false_accepts_on_review_controls": review_accepted,
        "false_rejects_on_eligible_tasks": false_rejects,
        "eligible_without_verified_completion": n_eligible - complete,
        "proposal_calls": sum(r["proposal_calls"] for r in trials), "model_calls": sum(r["model_calls"] for r in trials),
        "proposal_retries": sum(r["proposal_retries"] for r in trials),
        "numerical_retries": sum(r["numerical_retries"] for r in trials),
        "provider_errors": sum(r["code"] == "PROVIDER_ERROR" for r in active),
        "model_responses_rejected_by_code": dict(sorted(Counter(r["code"] for r in active if r["state"] == "REJECTED").items())),
        "tokens": tokens, "hosted_api_charge_usd": 0.0, "local_electricity_cost_usd": None,
        "workflow_seconds_total": sum(wall), "workflow_seconds_median_per_task": statistics.median(wall),
        "external_replay_seconds_total": sum(r["external_replay_seconds"] for r in trials),
    }


def panel(directory, model, digest, prior=None):
    from sciagent import planning_run

    tasks, checks = protocol()
    trials = []
    for i, row in enumerate(tasks):
        task = planner.PlanningTask.model_validate(row["task"])
        for mode in (MODES if i % 2 == 0 else reversed(MODES)):
            path = directory / row["id"] / mode
            identity = {"model": model if mode == "llm" else None, "model_digest": digest if mode == "llm" else None}
            if prior is None:
                record = planning_run.run(task, path, mode=mode, execute=True, **identity)
            else:
                record = prior["trials"][len(trials)]
                require(record["id"] == row["id"] and record["mode"] == mode, "trial ordering changed")
            started = time.monotonic()
            proof = planning_run.replay(path, record["manifest_sha256"])
            replay_elapsed = time.monotonic() - started
            require(proof["integrity"] is True and proof["mode"] == mode, f"incomplete proof for {path}")
            require(all(proof[k] == v for k, v in identity.items()), "wrong provider or model digest")
            require(planner.parse_task((path / "task.json").read_bytes()) == task,
                    "archived request differs from the fixed task")
            require(all(proof[k] == record[k] for k in FIELDS), f"run/replay mismatch for {path}")
            trial = {"id": row["id"], **{k: proof[k] for k in FIELDS}, "wall_seconds": record["wall_seconds"],
                     "external_replay_seconds": record["external_replay_seconds"] if prior is not None else replay_elapsed}
            trials.append(trial)
            print(row["id"], mode, trial["state"], trial["template_id"], flush=True)
    metrics = {m: score(tasks, [t for t in trials if t["mode"] == m]) for m in MODES}
    return {"version": 1, "task_corpus_sha256": TASK_SHA, "gate_controls_sha256": CONTROL_SHA,
            "model": model, "model_digest": digest, "gate_controls": checks, "metrics": metrics, "trials": trials}


def validate_comparison(result):
    metrics = result["metrics"]
    require(all(check["passed"] for check in result["gate_controls"]) and
            all(scores["unauthorized_acceptances_all_tasks"] == 0 for scores in metrics.values()),
            "a scope control failed")
    require(metrics["llm"]["model_calls"] > metrics["llm"]["provider_errors"],
            "no completed live model comparison")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, required=True, help="fresh capture root, or existing archives for replay")
    ap.add_argument("--report", type=Path, required=True, help="new output JSON")
    ap.add_argument("--model")
    ap.add_argument("--model-digest")
    ap.add_argument("--replay-report", type=Path)
    ap.add_argument("--report-sha256", help="externally retained digest of the original report")
    args = ap.parse_args()
    if args.report.exists():
        ap.error("the output report already exists")
    if args.replay_report:
        if args.model or args.model_digest or not args.report_sha256:
            ap.error("pass --report-sha256 for offline replay, without model arguments")
        prior = pinned_json(args.replay_report, args.report_sha256)
        result = panel(args.directory, prior["model"], prior["model_digest"], prior)
        require(result == prior, "the independently reconstructed planning report differs")
    else:
        if not args.model or not args.model_digest or args.report_sha256:
            ap.error("provide --model and --model-digest for live capture")
        args.directory.mkdir(parents=True, exist_ok=False)
        result = panel(args.directory, args.model, args.model_digest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    validate_comparison(result)


if __name__ == "__main__":
    main()
