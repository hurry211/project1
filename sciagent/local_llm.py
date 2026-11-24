"""Optional candidate generation through a local Ollama daemon."""

import http.client
import json
import re
import time

OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 128, "num_ctx": 4096}
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class LocalProviderError(ValueError):
    """The local provider could not certify or complete a model invocation."""


def _api(path, payload=None):
    # A direct socket to a literal loopback address avoids proxy configuration
    # and automatic redirect handling. Only fixed Ollama API paths call this.
    connection = http.client.HTTPConnection("127.0.0.1", 11434, timeout=120)
    try:
        body = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        connection.request("POST" if body is not None else "GET", path, body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise LocalProviderError(f"Ollama returned HTTP {response.status} for {path}")
        data = response.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise LocalProviderError(f"Ollama response exceeds the size limit for {path}")
        result = json.loads(data)
    except (OSError, http.client.HTTPException, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalProviderError(f"Could not read {path} from local Ollama: {exc}") from exc
    finally:
        connection.close()
    if not isinstance(result, dict) or "error" in result:
        raise LocalProviderError(f"Invalid Ollama response for {path}")
    return result


def _pinned_entry(path, model, model_digest):
    models = _api(path).get("models")
    if not isinstance(models, list):
        raise LocalProviderError("Ollama did not return a model inventory")
    entries = [entry for entry in models if isinstance(entry, dict) and entry.get("name") == model]
    if len(entries) != 1 or entries[0].get("digest") != model_digest:
        raise LocalProviderError(f"The Ollama model in {path} differs from the requested SHA-256 digest")
    return entries[0]


def _counter(record, name):
    value = record.get(name)
    if value is not None and (type(value) is not int or value < 0):
        raise LocalProviderError(f"Invalid Ollama measurement: {name}")
    return value


def generate(system: str, prompt: str, *, model: str, model_digest: str) -> dict:
    """Return raw model text and measured usage. The planner checks the proposal."""
    if (not isinstance(model, str) or len(model) > 190
            or not re.fullmatch(r"(?:[\w.-]+/)*[\w.-]+:[\w.-]+", model, re.ASCII)
            or model.lower().endswith((":cloud", "-cloud")) or not isinstance(model_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", model_digest)):
        raise LocalProviderError("Supply a local model tag and its full lowercase SHA-256 digest")
    if not isinstance(system, str) or not isinstance(prompt, str):
        raise LocalProviderError("The model requires text system and user messages")
    started = time.perf_counter()
    version = _api("/api/version").get("version")
    if not isinstance(version, str) or not version:
        raise LocalProviderError("Missing Ollama version")
    _pinned_entry("/api/tags", model, model_digest)
    show = _api("/api/show", {"model": model})
    if (show.get("remote_model") or show.get("remote_host")
            or not isinstance(show.get("details"), dict) or show["details"].get("format") != "gguf"
            or not isinstance(show.get("model_info"), dict) or not show["model_info"].get("general.architecture")):
        raise LocalProviderError("Ollama did not describe a locally installed GGUF model")
    # A local model's generated Modelfile identifies its weight blob.
    blob = re.search(r"^FROM\s+[^\r\n]*[/\\]blobs[/\\]sha256-([0-9a-f]{64})[ \t]*\r?$",
                     str(show.get("modelfile", "")), re.MULTILINE)
    if not blob:
        raise LocalProviderError("Missing local GGUF weight blob in Ollama model metadata")
    record = _api("/api/generate", {"model": model, "system": system, "prompt": prompt, "format": "json",
                                    "stream": False, "keep_alive": "2m", "options": dict(OPTIONS)})
    running = _pinned_entry("/api/ps", model, model_digest)
    _pinned_entry("/api/tags", model, model_digest)
    if (record.get("done") is not True or record.get("model") != model
            or not isinstance(record.get("response"), str) or record.get("done_reason") not in {"stop", "length"}):
        raise LocalProviderError("Ollama did not return a completed response from the requested model")
    timings = {f"{k}_ns": _counter(record, k) for k in
               ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")}
    timings["client_wall_seconds"] = time.perf_counter() - started
    return {"provider": "ollama_local", "ollama_version": version, "model": model, "model_digest": model_digest,
            "weight_blob_sha256": blob.group(1), "options": dict(OPTIONS), "text": record["response"],
            "runtime": {"size_vram_bytes": _counter(running, "size_vram"), "context_length": _counter(running, "context_length")},
            "done_reason": record["done_reason"], "usage": {"prompt_tokens": _counter(record, "prompt_eval_count"),
                                                           "completion_tokens": _counter(record, "eval_count")},
            "timings": timings, "cost": {"hosted_api_charge_usd": 0.0, "electricity_cost_usd": None}}
