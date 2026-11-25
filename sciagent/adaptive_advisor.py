"""Auditable nominations among the already feasible adaptive solver configurations."""
import json

from . import local_llm
from .planner import _strict_object


SYSTEM = ("Select one finite element configuration from the supplied shortlist. Consider the L2 target, "
          "memory budget, error margin and time estimate. Return only a JSON object with exactly one "
          "key candidate_id and a string copied verbatim from the shortlist. Use null to abstain.")


def choose(features, hardware, previews, *, model, digest):
    """Retain both successful and declined proposals; the numerical gate owns acceptance."""
    ids = [item["candidate_id"] for item in previews]
    if not 1 <= len(ids) <= 5 or any(type(key) is not str for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("expected one to five distinct reviewed candidate IDs")
    prompt = json.dumps({"problem": features, "hardware": hardware, "candidates": previews},
                        sort_keys=True, allow_nan=False, separators=(",", ":"))
    audit = {"model": model, "model_digest": digest, "system": SYSTEM, "prompt": prompt,
             "used_live_llm": False, "provider_completed": False}
    try:
        response = local_llm.generate(SYSTEM, prompt, model=model, model_digest=digest)
    except local_llm.LocalProviderError as exc:
        return None, {**audit, "status": "provider_error", "reason": str(exc)}
    audit.update(response=response, provider_completed=True)
    if response["done_reason"] != "stop":
        return None, {**audit, "status": "incomplete"}
    try:
        proposal = _strict_object(response["text"], 512)
    except ValueError:
        return None, {**audit, "status": "invalid_json"}
    if set(proposal) != {"candidate_id"}:
        return None, {**audit, "status": "invalid_schema"}
    choice = proposal["candidate_id"]
    if choice is None:
        return None, {**audit, "status": "abstained"}
    if type(choice) is not str or choice not in ids:
        return None, {**audit, "status": "outside_shortlist"}
    return choice, {**audit, "status": "nominated", "used_live_llm": True}
