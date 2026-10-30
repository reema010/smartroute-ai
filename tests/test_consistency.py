"""Locks field names and definitions together so a rename in one place cannot silently break another."""
import json

import pytest

import smoke_test
from config import settings
from evaluation import analyze, check_dataset
from src.mock_backend import MockBackend
from src.models import LLMClient
from src.pipeline import Pipeline
from src.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_response_format
from src.verification import validate

TOP_LEVEL_KEYS = ["intent", "urgency", "fields", "confidence"]


def perfect_output():
    return {"intent": settings.INTENTS[0], "urgency": settings.URGENCY_LEVELS[0],
            "fields": {n: None for n in settings.FIELD_NAMES}, "confidence": 0.99}


def test_prompt_names_every_schema_element():
    for word in TOP_LEVEL_KEYS + settings.INTENTS + settings.URGENCY_LEVELS + settings.FIELD_NAMES:
        assert word in SYSTEM_PROMPT, word


def test_validator_accepts_the_prompt_schema_and_flags_each_missing_key():
    assert validate(perfect_output(), "text", "standard", True) == []
    for key in TOP_LEVEL_KEYS:
        broken = {k: v for k, v in perfect_output().items() if k != key}
        assert f"missing_{key}" in validate(broken, "text", "standard", True)


def test_response_schema_matches_settings():
    assert callable(smoke_test.main)
    assert build_response_format("json_object") == {"type": "json_object"}
    assert build_response_format("json_schema")["json_schema"]["schema"] is RESPONSE_SCHEMA
    props = RESPONSE_SCHEMA["properties"]
    assert list(props) == TOP_LEVEL_KEYS
    assert props["intent"]["enum"] == settings.INTENTS and props["urgency"]["enum"] == settings.URGENCY_LEVELS
    assert list(props["fields"]["properties"]) == settings.FIELD_NAMES


def test_mock_backend_output_matches_the_schema():
    text, _, _ = MockBackend().complete(settings.MODELS["small"], [{"role": "user", "content": "refund ORD-12345 90 SAR"}], {})
    out = json.loads(text)
    assert list(out) == TOP_LEVEL_KEYS and list(out["fields"]) == settings.FIELD_NAMES


def test_scorer_success_means_intent_and_urgency_and_all_fields():
    gold = {"intent": "refund_request", "urgency": "normal", "fields": {"reference_id": "ORD-1234", "amount": "1,200 SAR"}}
    good = {"intent": "refund_request", "urgency": "normal", "fields": {"reference_id": "ord 1234", "amount": "SAR 1200.00"}}
    assert analyze.score_one(good, gold)["success"]  # same normalisation as the validator's grounding
    for key, value in (("intent", "payment_failed"), ("urgency", "high")):
        assert not analyze.score_one({**good, key: value}, gold)["success"]
    assert not analyze.score_one({**good, "fields": {**good["fields"], "amount": "999 SAR"}}, gold)["success"]
    assert not analyze.score_one(None, gold)["success"]
    extra = {**gold, "fields": {**gold["fields"], "unused": "x"}}
    assert analyze.score_one(good, extra)["success"]  # only settings.FIELD_NAMES are scored


def test_sample_dataset_passes_the_checker():
    errors, warnings, summary = check_dataset.check("sample")
    assert errors == [] and summary["n"] == 6
    labels = [json.loads(line) for line in open(settings.DATA_DIR / "sample_labels.jsonl", encoding="utf-8")]
    assert all(list(g["fields"]) == settings.FIELD_NAMES for g in labels)


def _write(tmp_path, tickets, labels, split="dev"):
    (tmp_path / f"{split}_tickets.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tickets), encoding="utf-8")
    (tmp_path / f"{split}_labels.jsonl").write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in labels), encoding="utf-8")


def good_pair(i="a1"):
    return ({"id": i, "messages": ["Payment failed for order ORD-12345, 250 SAR"]},
            {"id": i, "intent": "payment_failed", "urgency": "normal", "fields": {"reference_id": "ORD-12345", "amount": "250 SAR"}, "tags": ["easy"]})


def test_checker_accepts_a_good_dataset_and_catches_each_kind_of_mistake(tmp_path):
    t, g = good_pair()
    _write(tmp_path, [t], [g])
    assert check_dataset.check("dev", tmp_path)[0] == []

    bad_cases = {
        "ungrounded": {**g, "fields": {"reference_id": "ORD-99999", "amount": "250 SAR"}},
        "bad intent": {**g, "intent": "payments"},
        "bad urgency": {**g, "urgency": "urgent"},
        "wrong field keys": {**g, "fields": {"order_id": "ORD-12345", "amount": "250 SAR"}},
        "unknown tag": {**g, "tags": ["weird"]},
    }
    for name, bad in bad_cases.items():
        _write(tmp_path, [t], [bad])
        assert check_dataset.check("dev", tmp_path)[0], name

    _write(tmp_path, [{**t, "intent": "payment_failed"}], [g])  # label leaked into the ticket file
    assert any("leakage" in e for e in check_dataset.check("dev", tmp_path)[0])
    _write(tmp_path, [t], [{**g, "id": "zzz"}])  # ids differ
    assert any("ids differ" in e for e in check_dataset.check("dev", tmp_path)[0])


def test_checker_warns_about_probable_missing_labels(tmp_path):
    t, g = good_pair()
    _write(tmp_path, [t], [{**g, "fields": {"reference_id": None, "amount": None}}])
    errors, warnings, _ = check_dataset.check("dev", tmp_path)
    assert errors == [] and any("id-like" in w for w in warnings) and any("amount-like" in w for w in warnings)


def test_cross_split_check_flags_shared_templates(tmp_path):
    t, g = good_pair("d1")
    _write(tmp_path, [t], [g], "dev")
    t2 = {"id": "t1", "messages": ["Payment failed for order ORD-77777, 910 SAR"]}  # same sentence, new numbers
    _write(tmp_path, [t2], [{**g, "id": "t1"}], "test")
    errors, warnings = check_dataset.check_cross_split(tmp_path)
    assert errors == [] and any("same template" in w for w in warnings)
    _write(tmp_path, [{**t, "id": "t2"}], [{**g, "id": "t2"}], "test")  # identical text
    assert check_dataset.check_cross_split(tmp_path)[0]


def test_metrics_follow_their_definitions(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path)
    tickets = [json.loads(line) for line in open(settings.DATA_DIR / "sample_tickets.jsonl", encoding="utf-8")]
    pipe = Pipeline(LLMClient(MockBackend(), tmp_path / "c.jsonl"))
    out = tmp_path / "mock" / "sample"
    out.mkdir(parents=True)
    rows = [pipe.process(t, "always_large") for t in tickets]
    rows[0] = {**rows[0], "prediction": None, "final_status": "api_error", "route_status": "api_error"}  # simulate an outage
    with open(out / "always_large.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    m = analyze.run("mock", "sample", ["always_large"])[0]

    labels = {g["id"]: g for g in map(json.loads, open(settings.DATA_DIR / "sample_labels.jsonl", encoding="utf-8"))}
    wins = sum(analyze.score_one(r["prediction"], labels[r["id"]])["success"] for r in rows)
    assert m["api_errors"] == 1 and m["n"] == 6
    assert m["task_success"] == pytest.approx(wins / 6)                      # api errors count as failures here
    assert m["success_excl_api_errors"] == pytest.approx(wins / 5)           # and are excluded here
    assert m["cost_per_success_usd"] == pytest.approx(m["total_cost_usd"] / wins)
    assert m["cost_savings_vs_large_pct"] == pytest.approx(0.0)
