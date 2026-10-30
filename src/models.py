"""Model access with a persistent call cache, rate limiting and per-call accounting.

Design rules (they protect the evaluation):
* Every call returns a CallRecord with tokens, reasoning tokens, latency, cost and attempt count.
* SDK-internal retries are OFF. We retry explicitly and record how many attempts were needed.
* The cache stores the ORIGINAL latency/usage and when it was measured (provenance), and replays it,
  so reruns report the same numbers instead of fake ~0 s latencies. `cached=True` marks a replay.
* Failed calls are never cached. A failure raises BackendError and is reported as api_error,
  which is NOT the same thing as a wrong prediction.
* Time spent waiting for rate limits or retrying is not counted as model latency.
"""
import hashlib
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from src.prompts import build_response_format


class BackendError(RuntimeError):
    """The model call failed (after retries, or because the request was rejected)."""


@dataclass
class CallRecord:
    tier: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_s: float
    cost_usd: float
    cached: bool
    text: str
    measured_at: str = ""  # UTC time of the original live call
    attempts: int = 1
    usage_raw: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class RateLimiter:
    """Sliding 60 s window over requests and tokens. Set rpm/tpm to None to disable."""

    def __init__(self, rpm, tpm, clock=time.monotonic, sleep=time.sleep):
        self.rpm, self.tpm = rpm, tpm
        self.clock, self.sleep = clock, sleep
        self.events = deque()  # (timestamp, tokens)
        self.avg_tokens = 1200.0  # estimate for the next call, updated from observations

    def wait(self):
        waited = 0.0
        if not (self.rpm or self.tpm):
            return waited
        while True:
            now = self.clock()
            while self.events and now - self.events[0][0] >= 60:
                self.events.popleft()
            over_rpm = bool(self.rpm) and len(self.events) >= self.rpm
            used = sum(tokens for _, tokens in self.events)
            over_tpm = bool(self.tpm) and bool(self.events) and used + self.avg_tokens > self.tpm
            if not (over_rpm or over_tpm):
                return waited
            delay = max(0.0, 60 - (now - self.events[0][0])) + 0.05
            self.sleep(delay)
            waited += delay

    def record(self, tokens):
        self.events.append((self.clock(), tokens))
        if tokens:
            self.avg_tokens = 0.7 * self.avg_tokens + 0.3 * tokens


class GroqBackend:
    name = "groq"

    def __init__(self, limiter=None):
        from groq import Groq  # imported lazily so tests run without the package

        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise BackendError("GROQ_API_KEY is not set (see .env.example)")
        self.client = Groq(api_key=key, max_retries=0)  # SDK retries off: we count attempts ourselves
        rl = settings.RATE_LIMITS
        self.limiter = limiter or RateLimiter(rl.get("rpm"), rl.get("tpm"))

    @staticmethod
    def _retry_after(exc):
        headers = getattr(getattr(exc, "response", None), "headers", None)
        try:
            return float(headers.get("retry-after")) if headers else None
        except (TypeError, ValueError):
            return None

    def complete(self, model, messages, params):
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=params["temperature"],
            max_completion_tokens=params["max_completion_tokens"],
            response_format=build_response_format(params.get("response_format", "json_object")),
        )
        if params.get("reasoning_effort"):
            kwargs["reasoning_effort"] = params["reasoning_effort"]
        last = None
        for attempt in range(1, settings.MAX_RETRIES + 1):
            self.limiter.wait()
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - t0  # successful attempt only
                usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(vars(resp.usage))
                usage["_attempts"] = attempt
                self.limiter.record(int(usage.get("total_tokens") or 0))
                return resp.choices[0].message.content or "", usage, latency
            except Exception as exc:  # rate limit, network, 5xx, bad request ...
                last = exc
                code = getattr(exc, "status_code", None)
                if code == 400 and "json_validate_failed" in str(exc):
                    # The MODEL failed to produce valid JSON. That is a model-quality failure, not an API outage:
                    # return empty text so the pipeline sees invalid_json (and escalates or counts it wrong).
                    # Groq reports no usage for this error, so tokens/cost are recorded as 0 and flagged.
                    latency = time.perf_counter() - t0
                    self.limiter.record(int(self.limiter.avg_tokens))
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                             "_json_validate_failed": True, "_usage_unknown": True,
                             "_attempts": attempt, "_error": str(exc)[:500]}
                    return "", usage, latency
                if code in (400, 401, 403, 404, 422):  # not transient: surface it immediately
                    raise BackendError(f"{model} rejected the request ({code}): {exc}") from exc
                delay = min(2 ** attempt * 2, 30)
                retry_after = self._retry_after(exc)
                if retry_after is not None:
                    if retry_after > 300:
                        raise BackendError(f"{model}: rate-limit window of {retry_after:.0f}s is too long to wait") from exc
                    delay = max(delay, retry_after)
                time.sleep(delay)
        raise BackendError(f"{model} failed after {settings.MAX_RETRIES} attempts: {last}")


class LLMClient:
    def __init__(self, backend, cache_path, use_cache=True):
        self.backend = backend
        self.cache_path = Path(cache_path)
        self.use_cache = use_cache
        self._cache = {}
        if use_cache and self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        self._cache[row["key"]] = row
                    except (json.JSONDecodeError, KeyError):
                        continue  # ignore a half-written line

    def _params(self, tier):
        return {
            "temperature": settings.TEMPERATURE,
            "max_completion_tokens": settings.MAX_COMPLETION_TOKENS,
            "reasoning_effort": settings.REASONING_EFFORT.get(tier),
            "response_format": settings.RESPONSE_FORMAT,
        }

    def _key(self, model, messages, params):
        blob = json.dumps(
            {"backend": self.backend.name, "model": model, "messages": messages, "params": params},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cost(self, tier, prompt_tokens, completion_tokens):
        p = settings.PRICES_PER_M[tier]
        return (prompt_tokens * p["in"] + completion_tokens * p["out"]) / 1_000_000

    def call(self, tier, messages):
        model = settings.MODELS[tier]
        params = self._params(tier)
        key = self._key(model, messages, params)

        if self.use_cache and key in self._cache:
            row = self._cache[key]
            return self._record(tier, model, row["text"], row["usage"], row["latency_s"],
                                row.get("measured_at", ""), cached=True)

        text, usage, latency = self.backend.complete(model, messages, params)  # may raise BackendError
        measured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row = {"key": key, "model": model, "text": text, "usage": usage,
               "latency_s": latency, "measured_at": measured_at}
        if self.use_cache:
            self._cache[key] = row
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return self._record(tier, model, text, usage, latency, measured_at, cached=False)

    def _record(self, tier, model, text, usage, latency, measured_at, cached):
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)  # assumed to include reasoning tokens (smoke test checks)
        details = usage.get("completion_tokens_details") or {}
        reasoning = int(details.get("reasoning_tokens") or 0)
        return CallRecord(
            tier=tier, model=model, prompt_tokens=prompt, completion_tokens=completion,
            reasoning_tokens=reasoning, latency_s=float(latency),
            cost_usd=self._cost(tier, prompt, completion), cached=cached, text=text,
            measured_at=measured_at, attempts=int(usage.get("_attempts", 1)), usage_raw=usage,
        )
