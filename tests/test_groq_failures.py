"""Groq-specific error handling, tested with fake clients (no network)."""
import pytest

from config import settings
from src.mock_backend import MockBackend
from src.models import BackendError, GroqBackend, LLMClient, RateLimiter
from src.pipeline import Pipeline

PARAMS = {"temperature": 0, "max_completion_tokens": 100, "reasoning_effort": None, "response_format": "json_schema"}
JSON_FAIL = "Error code: 400 - {'error': {'code': 'json_validate_failed', 'failed_generation': ''}}"


class FakeError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def fake_backend(handler):
    backend = object.__new__(GroqBackend)  # skip __init__ (needs the groq package and a key)
    backend.limiter = RateLimiter(None, None)

    class Completions:
        def create(self, **kwargs):
            return handler(kwargs)

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    backend.client = Client()
    return backend


def test_json_validate_failed_is_a_model_failure_not_an_api_error():
    def handler(kwargs):
        raise FakeError(400, JSON_FAIL)

    text, usage, latency = fake_backend(handler).complete("m", [], PARAMS)
    assert text == "" and usage["_json_validate_failed"] and usage["total_tokens"] == 0 and latency >= 0


def test_other_bad_requests_still_raise_backend_error():
    def handler(kwargs):
        raise FakeError(400, "Error code: 400 - unsupported parameter")

    with pytest.raises(BackendError):
        fake_backend(handler).complete("m", [], PARAMS)


def test_response_format_is_passed_to_the_api():
    seen = []

    class U:
        def model_dump(self):
            return {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    class Msg:
        content = '{"a": 1}'

    class Choice:
        message = Msg()

    class Resp:
        usage, choices = U(), [Choice()]

    def handler(kwargs):
        seen.append(kwargs["response_format"])
        return Resp()

    backend = fake_backend(handler)
    backend.complete("m", [], PARAMS)
    backend.complete("m", [], {**PARAMS, "response_format": "json_object"})
    assert seen[0]["type"] == "json_schema" and seen[0]["json_schema"]["strict"] is True
    assert seen[1] == {"type": "json_object"}


def test_format_failure_is_cached_and_replayed(tmp_path):
    calls = {"n": 0}

    def handler(kwargs):
        calls["n"] += 1
        raise FakeError(400, JSON_FAIL)

    client = LLMClient(fake_backend(handler), tmp_path / "c.jsonl")
    msgs = [{"role": "user", "content": "x"}]
    first, second = client.call("small", msgs), client.call("small", msgs)
    assert calls["n"] == 1 and second.cached and second.text == "" and second.usage_raw["_json_validate_failed"]


def test_changing_response_format_changes_the_cache_key(tmp_path, monkeypatch):
    calls = {"n": 0}

    class Counting(MockBackend):
        def complete(self, *a, **k):
            calls["n"] += 1
            return super().complete(*a, **k)

    client = LLMClient(Counting(), tmp_path / "c.jsonl")
    msgs = [{"role": "user", "content": "payment failed ORD-12345"}]
    monkeypatch.setattr(settings, "RESPONSE_FORMAT", "json_schema")
    client.call("small", msgs)
    monkeypatch.setattr(settings, "RESPONSE_FORMAT", "json_object")
    client.call("small", msgs)
    assert calls["n"] == 2


def test_smartroute_escalates_when_the_small_model_fails_to_produce_json(tmp_path):
    class SmallFormatFailure(MockBackend):
        def complete(self, model, messages, params):
            if model == settings.MODELS["small"]:
                return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "_json_validate_failed": True}, 0.3
            return super().complete(model, messages, params)

    pipe = Pipeline(LLMClient(SmallFormatFailure(), tmp_path / "c.jsonl"))
    t = {"id": "t", "messages": ["payment failed order ORD-12345, 99 SAR"], "metadata": {}}
    r = pipe.process(t, "smartroute")
    assert r["escalated"] and r["final_status"] == "validated" and r["prediction"]["intent"] == "payment_failed"
    assert r["calls"][0]["usage_raw"]["_json_validate_failed"]
    baseline = pipe.process(t, "always_small")
    assert baseline["prediction"] is None and baseline["final_status"] == "unverified"  # a wrong answer, not an api_error
