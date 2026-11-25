"""Measure repeat requests with independent MFEM execution and pinned proposal reuse."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from benchmarks import planning
from sciagent import planner, planning_run
from sciagent.evidence import sha256_file
from sciagent.mfem_runner import solver_binary


def summarize(tasks, rows, previews):
    metrics = {}
    for mode in ("rules", "llm", "bounded"):
        subset = [r for r in rows if r["mode"] == mode]
        seeds = previews if mode == "bounded" else []
        metrics[mode] = {
            "model_calls_including_previews": sum(r["model_calls"] for r in subset + seeds),
            "seconds_including_previews": sum(r["wall_seconds"] for r in subset + seeds),
            "preference_matches": sum(r["preference_hit"] for r in subset),
            "states": dict(Counter(r["state"] for r in subset)),
            "routes": dict(Counter(r["route"] for r in subset)),
            "tasks_with_changing_choices": [task["id"] for task in tasks if len(
                {r["template_id"] for r in subset if r["id"] == task["id"]}) > 1],
        }
    return metrics


def capture(directory, model, digest, repeats=3):
    if repeats < 2:
        raise ValueError("at least two repeats required")
    directory.mkdir(parents=True, exist_ok=False)
    tasks, controls = planning.protocol()
    rng = random.Random(20261125)
    modes = ["rules", "llm", "bounded"]
    identity = {"model": model, "model_digest": digest}
    previews, rows = [], []
    for item in tasks:
        task = planner.PlanningTask.model_validate(item["task"])
        pin, source = None, None
        if len(planner.eligible_ids(task)) > 1:
            path = directory / item["id"] / "preview"
            receipt = planning_run.run(task, path, mode="llm", execute=False, **identity)
            previews.append({"id": item["id"], **{k: receipt[k] for k in
                             ("state", "template_id", "model_calls", "usage", "wall_seconds", "manifest_sha256")}})
            if receipt["state"] == "PLANNED":
                pin, source = receipt["manifest_sha256"], path
        for rep in range(repeats):
            order = list(modes)
            rng.shuffle(order)
            for mode in order:
                path = directory / item["id"] / f"{rep}-{mode}"
                kwargs = {} if mode == "rules" else identity.copy()
                if mode == "bounded" and pin:
                    kwargs.update(proposal_directory=source, proposal_manifest_sha256=pin)
                r = planning_run.run(task, path, mode=mode, execute=True, **kwargs)
                replay = planning_run.replay(path, r["manifest_sha256"])
                keys = ("state", "code", "template_id", "child_manifest_sha256", "manifest_sha256",
                        "model_calls", "usage")
                if not all(r[k] == replay[k] for k in keys):
                    raise ValueError("archived planner or numerical decision differs")
                if r["state"] == "ACCEPTED" and r["template_id"] not in item["permitted_ids"]:
                    raise ValueError("selected outside authorized templates")
                rows.append({"id": item["id"], "repeat": rep, "mode": mode, "route": r.get("proposal_route", mode),
                             "wall_seconds": r["wall_seconds"], **{k: r[k] for k in keys},
                             "preference_hit": item["expected_state"] == "ACCEPTED" and
                                               r["template_id"] == item["expected_template_id"] and
                                               r["state"] == "ACCEPTED"})
            print(item["id"], rep, flush=True)
    metrics = summarize(tasks, rows, previews)
    return {"model": model, "model_digest": digest, "binary_sha256": sha256_file(solver_binary()),
            "task_panel_sha256": planning.TASK_SHA, "order_seed": 20261125, "repeats": repeats,
            "gate_controls": controls, "previews": previews, "trials": rows, "metrics": metrics}


def audit(directory, report, digest):
    """Recheck the schedule, live proposal bindings, child solves and published metrics."""
    result = planning.pinned_json(report, digest)
    tasks, controls = planning.protocol()
    if (result["task_panel_sha256"] != planning.TASK_SHA or result["gate_controls"] != controls
            or result["order_seed"] != 20261125 or result["repeats"] < 2):
        raise ValueError("planner benchmark differs from the fixed task panel")
    rng = random.Random(result["order_seed"])
    expected_orders = []
    for task in tasks:
        for repeat in range(result["repeats"]):
            order = ["rules", "llm", "bounded"]
            rng.shuffle(order)
            expected_orders.extend((task["id"], repeat, mode) for mode in order)
    if [(r["id"], r["repeat"], r["mode"]) for r in result["trials"]] != expected_orders:
        raise ValueError("trial order differs from registered schedule")
    by_id = {item["id"]: item for item in tasks}
    expected_previews = [t["id"] for t in tasks if len(planner.eligible_ids(
        planner.PlanningTask.model_validate(t["task"]))) > 1]
    if [row["id"] for row in result["previews"]] != expected_previews:
        raise ValueError("preview schedule differs")
    seeds = {}
    for r in result["previews"] + result["trials"]:
        preview = "repeat" not in r
        mode = "llm" if preview else r["mode"]
        path = directory / r["id"] / ("preview" if preview else f"{r['repeat']}-{mode}")
        expected = by_id[r["id"]]
        task = planner.parse_task((path / "task.json").read_bytes())
        if task != planner.PlanningTask.model_validate(expected["task"]):
            raise ValueError("archived task differs from benchmark task")
        checked = planning_run.replay(path, r["manifest_sha256"])
        for key in ("state", "template_id", "model_calls", "usage"):
            if checked[key] != r[key]:
                raise ValueError(f"planner replay differs at {key}")
        expected_model = None if mode == "rules" else result["model"]
        expected_digest = None if mode == "rules" else result["model_digest"]
        config = json.loads((path / "config.json").read_bytes())
        if (checked["mode"] != mode or checked["model"] != expected_model
                or checked["model_digest"] != expected_digest or config["execute"] != (not preview)):
            raise ValueError("planner identity or execution mode differs")
        if preview:
            if r["state"] == "PLANNED":
                seeds[r["id"]] = r["manifest_sha256"]
            continue
        pin = seeds.get(r["id"]) if mode == "bounded" else None
        if config.get("proposal_manifest_sha256") != pin:
            raise ValueError("repeat proposal differs from the recorded live preview")
        hit = checked["state"] == "ACCEPTED" and checked["template_id"] == expected["expected_template_id"]
        if (r["preference_hit"] != hit or checked["child_manifest_sha256"] != r["child_manifest_sha256"]
                or checked["code"] != r["code"] or checked.get("proposal_route", mode) != r["route"]):
            raise ValueError("numerical result or planning route differs")
    metrics = summarize(tasks, result["trials"], result["previews"])
    if metrics != result["metrics"]:
        raise ValueError("planner summary differs from receipts")
    return {"report_sha256": digest, "verified_previews": len(seeds), "verified_trials": len(expected_orders),
            "matched_metrics": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--model-digest")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--audit-sha256", help="replay a retained report and complete run directory")
    args = parser.parse_args()
    if args.audit_sha256:
        print(json.dumps(audit(args.directory, args.report, args.audit_sha256), indent=2))
        raise SystemExit(0)
    if not args.model or not args.model_digest or args.report.exists():
        parser.error("capture requires a pinned local model and a fresh report path")
    result = capture(args.directory, args.model, args.model_digest, args.repeats)
    with args.report.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
