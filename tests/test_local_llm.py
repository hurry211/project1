import json

import pytest

from sciagent import local_llm


DIGEST = "a" * 64
MODEL = "example:local"


def fixture_api(monkeypatch, *, mutate_show=None, mutate_result=None, change_digest=False):
    calls = []

    def api(path, payload=None):
        calls.append((path, payload))
        if path == "/api/version":
            return {"version": "test"}
        if path in ("/api/tags", "/api/ps"):
            digest = "b" * 64 if change_digest and len(calls) > 3 else DIGEST
            return {"models": [{"name": MODEL, "digest": digest}]}
        if path == "/api/show":
            return {"model_info": {"general.architecture": "qwen2"}, "details": {"format": "gguf"},
                    "modelfile": f"FROM /models/blobs/sha256-{DIGEST}\n", **(mutate_show or {})}
        assert path == "/api/generate"
        return {"model": MODEL, "response": '{"template_id":null}', "done": True, "done_reason": "stop",
                "prompt_eval_count": 21, "eval_count": 6, "eval_duration": 130, **(mutate_result or {})}

    monkeypatch.setattr(local_llm, "_api", api)
    return calls


def test_usage_sampling_and_raw_response(monkeypatch):
    calls = fixture_api(monkeypatch)
    result = local_llm.generate("system", "task", model=MODEL, model_digest=DIGEST)
    assert result["text"] == '{"template_id":null}'
    assert result["usage"] == {"prompt_tokens": 21, "completion_tokens": 6}
    assert result["timings"]["eval_duration_ns"] == 130
    assert result["timings"]["load_duration_ns"] is None
    assert result["cost"]["electricity_cost_usd"] is None
    assert result["model_digest"] == DIGEST
    api_call, = [payload for path, payload in calls if path == "/api/generate"]
    assert api_call == {"model": MODEL, "prompt": "task", "system": "system", "format": "json",
                        "stream": False, "keep_alive": "2m", "options": local_llm.OPTIONS}


@pytest.mark.parametrize("metadata", [
    {"remote_model": "remote"}, {"remote_host": "provider"}, {"details": None},
    {"model_info": {}}, {"modelfile": "FROM supplied_name:latest"},
])
def test_only_a_local_gguf_receives_prompt(monkeypatch, metadata):
    calls = fixture_api(monkeypatch, mutate_show=metadata)
    with pytest.raises(local_llm.LocalProviderError):
        local_llm.generate("system", "task", model=MODEL, model_digest=DIGEST)
    assert all(path != "/api/generate" for path, _ in calls)


@pytest.mark.parametrize("model,digest", [(MODEL, "this is not a digest"), ("abc:cloud", DIGEST), ("https://url", "")])
def test_invalid_identity_before_request(monkeypatch, model, digest):
    calls = fixture_api(monkeypatch)
    with pytest.raises(local_llm.LocalProviderError):
        local_llm.generate("system", "task", model=model, model_digest=digest)
    assert not calls


def test_digest_matches_before_and_after_inference(monkeypatch):
    calls = fixture_api(monkeypatch, change_digest=True)
    with pytest.raises(local_llm.LocalProviderError, match="digest"):
        local_llm.generate("system", "task", model=MODEL, model_digest=DIGEST)
    assert [k for k, _ in calls][-2:] == ["/api/generate", "/api/ps"]
    calls = fixture_api(monkeypatch)
    with pytest.raises(local_llm.LocalProviderError, match="digest"):
        local_llm.generate("system", "task", model=MODEL, model_digest="b" * 64)
    assert all(path != "/api/generate" for path, _ in calls)


@pytest.mark.parametrize("record", [
    {"done": False}, {"model": "other"}, {"done_reason": "error"}, {"eval_count": -1}, {"eval_count": True}
])
def test_unusable_generation_record_rejected(monkeypatch, record):
    fixture_api(monkeypatch, mutate_result=record)
    with pytest.raises(local_llm.LocalProviderError):
        local_llm.generate("system", "task", model=MODEL, model_digest=DIGEST)


def test_transport_ignores_proxy_and_refuses_redirect(monkeypatch):
    class Connection:
        def __init__(self, host, port, *, timeout):
            assert (host, port) == ("127.0.0.1", 11434)

        def request(self, method, path, body, headers):
            assert (method, path, json.loads(body)) == ("POST", "/api/show", {"model": MODEL})

        def getresponse(self):
            class Redirect:
                status = 307

            return Redirect()

        def close(self):
            pass

    monkeypatch.setenv("HTTP_PROXY", "http://example.invalid:80")
    monkeypatch.setattr(local_llm.http.client, "HTTPConnection", Connection)
    with pytest.raises(local_llm.LocalProviderError, match="HTTP 307"):
        local_llm._api("/api/show", {"model": MODEL})
