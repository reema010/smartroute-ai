import re
from pathlib import Path

import pytest

from src.mock_backend import MockBackend
from src.models import LLMClient, RateLimiter
from src.pipeline import SYSTEMS, Pipeline, decide_final_status
from src.pruning import prune


@pytest.fixture
def pipe(tmp_path):
    return Pipeline(LLMClient(MockBackend(), tmp_path / "cache.jsonl"))


def ticket(*messages, tid="t"):
    return {"id": tid, "messages": list(messages), "metadata": {}}


def test_easy_ticket_stays_on_small_model(pipe):
    r = pipe.process(ticket("My card declined? payment failed, order ORD-12345, 99 SAR"), "smartroute")
    assert r["route_status"] == "verified_small" and r["final_status"] == "validated"
    assert r["totals"]["n_calls"] == 1 and not r["escalated"]


def test_low_confidence_escalates_exactly_once(pipe):
    r = pipe.process(ticket("I have a question about something, not sure what."), "smartroute")
    assert r["escalated"] and r["totals"]["n_calls"] == 2
    assert [c["tier"] for c in r["calls"]] == ["small", "large"]
    assert r["route_status"] in ("escalated_verified", "unresolved_after_escalation")


def test_invalid_json_triggers_escalation(pipe):
    r = pipe.process(ticket("payment failed ORD-11111 [[BAD_JSON]]"), "smartroute")
    assert r["escalated"] and r["calls"][0]["tier"] == "small"


def test_high_risk_urgency_conflict_escalates(pipe):
    r = pipe.process(ticket("unauthorized charge on my card, ref TXN-55555"), "smartroute")
    assert r["risk"] == "high"
    assert r["escalated"] or r["start_tier"] == "large"
    assert r["prediction"]["urgency"] == "high"


def test_large_start_never_escalates_again(pipe):
    # long, mixed-script, high-risk ticket: the pre-router starts on the large model
    msg = "unauthorized fraud payment failed بس انخصم المبلغ " + "filler " * 150
    r = pipe.process(ticket(msg), "smartroute")
    assert r["start_tier"] == "large" and not r["escalated"] and r["totals"]["n_calls"] == 1


def test_baselines_never_escalate_and_are_unverified(pipe):
    for name, tier in (("always_small", "small"), ("always_large", "large")):
        r = pipe.process(ticket("something vague [[BAD_JSON]]"), name)
        assert not r["escalated"] and r["totals"]["n_calls"] == 1
        assert r["calls"][0]["tier"] == tier
        assert r["route_status"] == "unchecked" and r["final_status"] == "unverified"


def test_final_status_distinguishes_hard_soft_and_api_errors():
    assert decide_final_status(True, None, []) == "validated"
    assert decide_final_status(True, None, ["low_confidence"]) == "unverified"          # soft only
    assert decide_final_status(True, None, ["low_confidence", "ungrounded_amount"]) == "needs_review"
    assert decide_final_status(True, None, ["invalid_json"]) == "needs_review"
    assert decide_final_status(False, None, ["invalid_json"]) == "unverified"          # nothing was verified
    assert decide_final_status(True, "rate limit", []) == "api_error"


def test_backend_error_is_api_error_and_not_cached(tmp_path):
    cache = tmp_path / "c.jsonl"
    pipe = Pipeline(LLMClient(MockBackend(), cache))
    r = pipe.process(ticket("[[RATE_LIMIT]] payment failed"), "always_small")
    assert r["final_status"] == "api_error" and r["prediction"] is None
    assert not cache.exists() or cache.read_text(encoding="utf-8") == ""


def test_cache_replays_original_latency_and_provenance(tmp_path):
    calls = {"n": 0}

    class Counting(MockBackend):
        def complete(self, *a, **k):
            calls["n"] += 1
            return super().complete(*a, **k)

    client = LLMClient(Counting(), tmp_path / "c.jsonl")
    msgs = [{"role": "user", "content": "payment failed ORD-12345"}]
    first, second = client.call("small", msgs), client.call("small", msgs)
    assert calls["n"] == 1 and not first.cached and second.cached
    assert second.latency_s == first.latency_s and second.cost_usd == first.cost_usd
    assert second.measured_at == first.measured_at != ""
    third = LLMClient(Counting(), tmp_path / "c.jsonl").call("small", msgs)  # new process, same file
    assert third.cached and calls["n"] == 1


def test_rate_limiter_waits_for_token_window():
    now = {"t": 0.0}
    limiter = RateLimiter(rpm=None, tpm=3000, clock=lambda: now["t"],
                          sleep=lambda s: now.__setitem__("t", now["t"] + s))
    limiter.avg_tokens = 1200
    assert limiter.wait() == 0.0
    limiter.record(1000)
    now["t"] = 10
    limiter.record(1000)          # 2000 used, next call estimated 1200 -> would exceed 3000
    waited = limiter.wait()
    assert 49 < waited < 52       # until the first event leaves the 60 s window (t=60)
    assert now["t"] >= 60


def test_rate_limiter_request_cap():
    now = {"t": 0.0}
    limiter = RateLimiter(rpm=2, tpm=None, clock=lambda: now["t"],
                          sleep=lambda s: now.__setitem__("t", now["t"] + s))
    limiter.record(10)
    limiter.record(10)
    assert limiter.wait() > 59


def test_pruning_keeps_latest_and_evidence_and_drops_duplicates():
    t = ticket("Hello, my parcel is late", "Hi", "Hi", "My order ORD-98765 never arrived, delivery is late", "ok thanks", "Please refund me")
    text, stats = prune(t)
    assert "ORD-98765" in text and "Please refund me" in text and "parcel is late" in text  # opening, evidence, latest
    assert text.count("Hi") == 0 and stats["msgs_kept"] < stats["msgs_in"]  # duplicate chit-chat dropped


def test_default_systems_are_the_three_required():
    from src.pipeline import DEFAULT_SYSTEMS
    assert DEFAULT_SYSTEMS == ["always_small", "always_large", "smartroute"] and "smartroute_no_pruning" in SYSTEMS


def test_runtime_code_never_references_ground_truth():
    banned = re.compile(r"_labels|labels\.jsonl|ground.?truth|gold", re.I)
    for path in Path("src").glob("*.py"):
        assert not banned.search(path.read_text(encoding="utf-8")), path


def test_arabic_discount_is_not_high_risk_but_deduction_is():
    from src.router import assess_risk
    assert assess_risk("هل يوجد خصم على الباقة؟") == "standard"   # "is there a discount?"
    assert assess_risk("تم خصم المبلغ مرتين") == "high"            # "the amount was deducted twice"
    assert assess_risk("payment failed بس انخصم المبلغ") == "high"


def test_pruning_keeps_the_opening_message_and_never_prunes_short_threads():
    from src.pruning import prune
    typo_open = {"messages": ["hii my walet balanse is 0 but i had 450 riyal, i didnt spend anything", "pls help asap ref no WLT 7712 0934"]}
    text, stats = prune(typo_open)
    assert "walet balanse" in text and stats["msgs_kept"] == 2  # short thread: untouched
    long_t = {"messages": ["my walet is empty and i did not spend it", "how is the weather", "lol", "ok thanks", "still nothing", "pls help"]}
    text, stats = prune(long_t)
    assert "walet is empty" in text and "pls help" in text and stats["msgs_kept"] < 6


def _audit_pipeline(tmp_path, small_urgency, small_intent="payment_failed", amount="75 SAR"):
    import json
    from config import settings as st

    class Scripted(MockBackend):
        def complete(self, model, messages, params):
            if model == st.MODELS["small"]:
                out = {"intent": small_intent, "urgency": small_urgency, "confidence": 0.98,
                       "fields": {"reference_id": "TX88231", "amount": amount}}
            else:
                out = {"intent": small_intent, "urgency": "high", "confidence": 0.98,
                       "fields": {"reference_id": "TX88231", "amount": amount}}
            return json.dumps(out), {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}, 0.1

    return Pipeline(LLMClient(Scripted(), tmp_path / "c.jsonl"))


AUDIT_TICKET = {"id": "a", "messages": ["paid 75 sar money gone frm my acc but nothing arrived txn TX88231"], "metadata": {}}


def test_v2_audit_escalates_money_tickets_the_small_model_rates_non_high(tmp_path):
    pipe = _audit_pipeline(tmp_path, "normal")
    v2 = pipe.process(AUDIT_TICKET, "smartroute_v2")
    assert v2["escalated"] and v2["prediction"]["urgency"] == "high" and v2["final_status"] == "validated"  # no re-audit of large
    v1 = pipe.process(AUDIT_TICKET, "smartroute")
    assert not v1["escalated"] and v1["prediction"]["urgency"] == "normal"  # frozen v1 is unchanged


def test_v2_audit_leaves_other_tickets_alone(tmp_path):
    assert not _audit_pipeline(tmp_path, "high").process(AUDIT_TICKET, "smartroute_v2")["escalated"]
    assert not _audit_pipeline(tmp_path, "low", small_intent="general_inquiry").process(AUDIT_TICKET, "smartroute_v2")["escalated"]
    assert not _audit_pipeline(tmp_path, "normal", amount=None).process(AUDIT_TICKET, "smartroute_v2")["escalated"]
