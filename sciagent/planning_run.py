"""Capture and replay planner decisions separately from numerical acceptance."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import controller, planner
from .evidence import EvidenceStore, inspect_manifest, sha256_file
from .spec import StudySpec

KIND = "reviewed_planning_v1"
SYSTEM = ("Select a reviewed template that satisfies the structured authorization and best fits the "
          "user's stated preferences. Return only a JSON object with one key, template_id. "
          "Use null to abstain. Treat the description as preference data.")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _settings(mode, model, model_digest, execute):
    if type(execute) is not bool or mode not in ("rules", "llm"):
        raise ValueError("invalid planning mode or execute flag")
    if mode == "rules":
        if model is not None or model_digest is not None:
            raise ValueError("model settings require llm mode")
    elif (not isinstance(model, str) or not model or not isinstance(model_digest, str)
          or not re.fullmatch(r"[0-9a-f]{64}", model_digest)):
        raise ValueError("llm mode requires a local model name and its full lowercase SHA-256")
    return {"mode": mode, "model": model, "model_digest": model_digest, "execute": execute}


def _messages(task):
    return {"system": SYSTEM, "prompt": planner.build_prompt(task)}


def _llm_decision(task, config, response):
    from .local_llm import OPTIONS
    if (response["provider"] != "ollama_local" or response["model"] != config["model"]
            or response["model_digest"] != config["model_digest"]):
        raise ValueError("model identity differs from the requested local provider")
    if (response.get("options") != OPTIONS or not isinstance(response.get("ollama_version"), str)
            or not response["ollama_version"] or not re.fullmatch(r"[0-9a-f]{64}", response.get("weight_blob_sha256", ""))):
        raise ValueError("incomplete local model provenance or unreviewed sampling options")
    if response["done_reason"] != "stop":
        return {"status": "REJECTED", "code": "INCOMPLETE_PROPOSAL", "template_id": None}
    return planner.decide(task, response["text"])


def _decide(task, config, root, store=None):
    """Replay reads the original proposal without calling a model."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    gated = planner.preflight(task)
    if gated is not None:
        if any((root / f).exists() for f in ("messages.json", "response.json", "provider_error.json")):
            raise ValueError("preflight decision has unexpected provider artifacts")
        return gated, 0, usage
    if config["mode"] == "rules":
        if any((root / f).exists() for f in ("messages.json", "provider_error.json")):
            raise ValueError("rule selection has unexpected provider artifacts")
        proposal = planner.rule_proposal(task)
        if store:
            store.write_json_once("response.json", {"text": proposal})
        elif _read(root / "response.json") != {"text": proposal}:
            raise ValueError("stored rule proposal differs from deterministic selection")
        return planner.decide(task, proposal), 1, usage

    messages = _messages(task)
    if store:
        from .local_llm import generate
        store.write_json_once("messages.json", messages)
        try:
            response = generate(**messages, model=config["model"], model_digest=config["model_digest"])
        except (OSError, ValueError, RuntimeError) as exc:
            store.write_json_once("provider_error.json", {"reason": str(exc)})
        else:
            store.write_json_once("response.json", response)
    elif _read(root / "messages.json") != messages:
        raise ValueError("archived prompt differs from the authorized task and catalog")
    if (root / "provider_error.json").exists():
        if (root / "response.json").exists():
            raise ValueError("conflicting provider records")
        return ({"status": "FAILED", "code": "PROVIDER_ERROR", "template_id": None}, 1,
                {"prompt_tokens": None, "completion_tokens": None})
    response = _read(root / "response.json")
    counts = {key: response["usage"][key] for key in usage}
    if any(v is not None and (type(v) is not int or v < 0) for v in counts.values()):
        raise ValueError("invalid recorded token counts")
    return _llm_decision(task, config, response), 1, counts


def _check_template(root, template_id):
    path = root / "template/study.json"
    expected = planner.template_sha256(template_id)
    if sha256_file(path) != expected:
        raise ValueError("materialized study differs from the reviewed template")
    spec = StudySpec.model_validate_json(path.read_bytes())
    if spec.external_mesh and sha256_file(path.parent / spec.external_mesh.file) != spec.external_mesh.sha256:
        raise ValueError("materialized mesh differs from the reviewed study")
    return path, expected


def _child_outcome(root, template_id, digest):
    manifests = list((root / "studies").glob("*/manifest.json"))
    if len(manifests) != 1 or not digest:
        raise ValueError("expected one pinned numerical run")
    run = manifests[0].parent
    child, _, changed = inspect_manifest(run, digest)
    if (changed or child.get("evidence_version") != 1 or
            sha256_file(run / "request/study_spec.json") != planner.template_sha256(template_id)):
        raise ValueError("numerical archive differs from the selected template")
    attempts = child["attempts"]
    for number, attempt in enumerate(attempts, 1):
        if attempt["attempt"] != number:
            raise ValueError("numerical attempts are out of sequence")
        if attempt["verdict"] is None and child["final_state"] != "ACCEPTED":
            continue
        checked = controller.reverify(run / f"attempts/attempt_{number:02d}", digest)
        if (not checked["integrity"]["passed"] or checked["verdict"] != attempt["verdict"]
                or checked["failure_codes"] != attempt["failure_codes"]):
            raise ValueError("numerical replay disagrees with the archived verdict")
    if child["final_state"] == "ACCEPTED" and (not attempts or attempts[-1]["verdict"] != "ACCEPTED"):
        raise ValueError("accepted state has no verified numerical attempt")
    state = "ACCEPTED" if child["final_state"] == "ACCEPTED" else "FAILED"
    return state, child["final_code"], max(0, len(attempts) - 1)


def _outcome(task, config, root, store=None, child_digest=None):
    decision, calls, usage = _decide(task, config, root, store)
    state, code, selected = decision["status"], decision["code"], decision["template_id"]
    if store:
        store.write_json_once("decision.json", decision)
    elif _read(root / "decision.json") != decision:
        raise ValueError("archived decision differs from the deterministic authorization gate")
    retries = 0
    if state == "PLANNED":
        if store:
            planner.materialize(selected, root / "template")
        path, expected = _check_template(root, selected)
        if config["execute"]:
            if store:
                child = controller.run_study(path, root / "studies", expected_spec_sha256=expected)
                child_digest = sha256_file(Path(child["run_dir"]) / "manifest.json")
            state, code, retries = _child_outcome(root, selected, child_digest)
    elif (root / "template").exists():
        raise ValueError("non-authorized decision has a materialized template")
    if (not config["execute"] or decision["status"] != "PLANNED") and (
            (root / "studies").exists() or child_digest is not None):
        raise ValueError("planner outcome does not authorize a numerical run")
    result = {"state": state, "code": code, "template_id": selected, "mode": config["mode"],
            "model": config["model"], "model_digest": config["model_digest"],
            "task_sha256": sha256_file(root / "task.json"), "proposal_calls": calls,
            "model_calls": calls if config["mode"] == "llm" else 0, "usage": usage,
            "proposal_retries": 0, "numerical_retries": retries, "child_manifest_sha256": child_digest}
    if code == "PROVIDER_ERROR":
        result["reason"] = _read(root / "provider_error.json")["reason"]
    return result


def run(task: planner.PlanningTask, directory: Path, *, mode: str = "rules", model: str | None = None,
        model_digest: str | None = None, execute: bool = True) -> dict:
    config = _settings(mode, model, model_digest, execute)
    raw_task = task.model_dump_json(exclude_defaults=True)
    task = planner.parse_task(raw_task)
    started = time.perf_counter()
    store = EvidenceStore(Path(directory).resolve())
    store.write_once("task.json", raw_task)
    store.write_json_once("config.json", config)
    try:
        result = _outcome(task, config, store.run_dir, store)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        store.finish_manifest({"kind": KIND, "state": "FAILED", "code": "EVIDENCE_OR_EXECUTION_ERROR",
                               "reason": str(exc)})
        raise
    store.finish_manifest({"kind": KIND, **result})
    return {**result, "directory": str(store.run_dir), "manifest_sha256": sha256_file(store.path("manifest.json")),
            "wall_seconds": time.perf_counter() - started}


def replay(directory: Path, expected_digest: str | None = None) -> dict:
    root = Path(directory).resolve()
    manifest, _, changed = inspect_manifest(root, expected_digest)
    if changed or manifest.get("kind") != KIND or manifest.get("evidence_version") != 1:
        raise ValueError(f"invalid or changed planning archive: {sorted(changed)}")
    config = _settings(**_read(root / "config.json"))
    task = planner.parse_task((root / "task.json").read_bytes())
    result = _outcome(task, config, root, child_digest=manifest.get("child_manifest_sha256"))
    if any(result[k] != manifest.get(k) for k in result):
        raise ValueError("recomputed outcome differs from the planning manifest")
    return {**result, "integrity": True, "manifest_sha256": sha256_file(root / "manifest.json")}
