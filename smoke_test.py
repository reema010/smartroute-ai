"""Real Groq smoke test. Run this BEFORE generating the full dataset.

    python smoke_test.py                          # real calls, uses RESPONSE_FORMAT from config/settings.py
    python smoke_test.py --format json_object     # compare the other output format
    python smoke_test.py --mock                   # offline check of this script's own logic

It checks: both models answer, output parses and validates, token/reasoning usage is reported,
mixed Arabic/English text works, and your account's real rate limits (response headers).
It writes results/smoke_test.json: paste that file's content (or the console output) to your assistant.
"""
import argparse
import json
import sys

from config import settings
from src.models import BackendError, GroqBackend, LLMClient
from src.prompts import build_messages, build_response_format
from src.pruning import render_raw
from src.router import assess_risk
from src.verification import parse_output, validate

TICKETS = [
    {"id": "smoke1", "messages": ["My payment failed when I renewed. Order ORD-48213, amount 250 SAR."], "expect": "payment_failed"},
    {"id": "smoke2", "messages": ["payment failed بس انخصم المبلغ 90 SAR، رقم الطلب ORD-55012"], "expect": "payment_failed"},
]


def rate_limit_headers(backend, tier, messages):
    """Read x-ratelimit-* headers for this account (Groq only)."""
    try:
        raw = backend.client.chat.completions.with_raw_response.create(
            model=settings.MODELS[tier], messages=messages, temperature=0,
            max_completion_tokens=settings.MAX_COMPLETION_TOKENS,
            response_format=build_response_format(settings.RESPONSE_FORMAT),
            **({"reasoning_effort": settings.REASONING_EFFORT[tier]} if settings.REASONING_EFFORT.get(tier) else {}),
        )
        return {k: v for k, v in raw.headers.items() if k.lower().startswith(("x-ratelimit", "retry-after"))}
    except Exception as exc:
        return {"error": str(exc)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--format", choices=["json_object", "json_schema"], default=None)
    args = ap.parse_args()
    if args.format:
        settings.RESPONSE_FORMAT = args.format
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if args.mock:
        from src.mock_backend import MockBackend
        backend = MockBackend()
    else:
        try:
            backend = GroqBackend()
        except BackendError as exc:
            print(f"FAIL: {exc}")
            return 1

    print(f"Response format: {settings.RESPONSE_FORMAT} | reasoning effort: {settings.REASONING_EFFORT}")
    client = LLMClient(backend, settings.CACHE_DIR / "smoke_unused.jsonl", use_cache=False)
    report = {"mock": args.mock, "calls": [], "settings": {
        "models": settings.MODELS, "reasoning_effort": settings.REASONING_EFFORT,
        "response_format": settings.RESPONSE_FORMAT,
        "max_completion_tokens": settings.MAX_COMPLETION_TOKENS, "rate_limits": settings.RATE_LIMITS}}
    failures = 0

    for tier in ("small", "large"):
        for t in TICKETS:
            raw = render_raw(t)
            try:
                rec = client.call(tier, build_messages(raw))
            except BackendError as exc:
                print(f"FAIL [{tier}/{t['id']}] API problem: {exc}")
                failures += 1
                report["calls"].append({"tier": tier, "id": t["id"], "error": str(exc)})
                continue
            parsed, issue = parse_output(rec.text)
            issues = [issue] if issue else validate(parsed, raw, assess_risk(raw), True)
            format_failure = bool(rec.usage_raw.get("_json_validate_failed"))
            failures += 0 if parsed is not None else 1
            headroom = settings.MAX_COMPLETION_TOKENS - rec.completion_tokens
            print(f"[{tier}/{t['id']}] {rec.latency_s:.2f}s  in={rec.prompt_tokens} out={rec.completion_tokens} "
                  f"reasoning={rec.reasoning_tokens} headroom={headroom}  cost=${rec.cost_usd:.6f}  attempts={rec.attempts}")
            if format_failure:
                print("    FORMAT FAILURE: the model did not produce valid JSON for this request (counted as a model failure, not an API error)")
            print(f"    intent={parsed and parsed.get('intent')} (expected {t['expect']})  urgency={parsed and parsed.get('urgency')}  "
                  f"fields={parsed and parsed.get('fields')}  conf={parsed and parsed.get('confidence')}  issues={issues}")
            if rec.reasoning_tokens > rec.completion_tokens:
                print("    NOTE: reasoning_tokens > completion_tokens, so completion_tokens may EXCLUDE reasoning. Tell your assistant.")
            if headroom < 150 and not format_failure:
                print("    NOTE: little headroom under MAX_COMPLETION_TOKENS; outputs may truncate. Raise it in config/settings.py.")
            report["calls"].append({"tier": tier, "id": t["id"], "latency_s": rec.latency_s, "usage": rec.usage_raw,
                                    "parsed": parsed, "issues": issues, "expected_intent": t["expect"],
                                    "format_failure": format_failure})

    if not args.mock:
        msgs = build_messages(render_raw(TICKETS[0]))
        report["rate_limit_headers"] = {tier: rate_limit_headers(backend, tier, msgs) for tier in ("small", "large")}
        print("\nRate-limit headers:", json.dumps(report["rate_limit_headers"], indent=2))

    out = settings.RESULTS_DIR / ("smoke_test_mock.json" if args.mock else "smoke_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")
    print("SMOKE TEST PASSED" if failures == 0 else f"SMOKE TEST FAILED ({failures} problem(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
