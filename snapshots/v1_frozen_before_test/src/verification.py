"""Deterministic output validation and uncertainty signals. No model calls here.

validate() returns a list of issue codes. Any issue means "uncertain": the pipeline escalates
(small model) or flags the ticket for human review (large model).
"""
import json
import re

from config import settings

# Signals that make a result "unverified" rather than "needs_review" if they remain after the final model.
SOFT_ISSUES = {"low_confidence"}


def parse_output(text):
    """Return (parsed_dict | None, issue | None)."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text or "", re.S)
        try:
            parsed = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            parsed = None
    if not isinstance(parsed, dict):
        return None, "invalid_json"
    return parsed, None


def _norm(value):
    return re.sub(r"[\s\-_]", "", str(value).lower())


def _numbers(value):
    return {float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(value).replace(",", ""))}


def _grounded(name, value, raw_text):
    """An extracted value must literally appear in the ticket (cheap hallucination check)."""
    if value in (None, ""):
        return True
    if name == "amount":
        nums = _numbers(value)
        return bool(nums) and nums <= _numbers(raw_text)
    return _norm(value) in _norm(raw_text)


def validate(parsed, raw_text, risk, risk_checks):
    if parsed is None:
        return []
    issues = []
    for key in ("intent", "urgency", "fields", "confidence"):
        if key not in parsed:
            issues.append(f"missing_{key}")
    if "intent" in parsed and parsed["intent"] not in settings.INTENTS:
        issues.append("bad_intent")
    if "urgency" in parsed and parsed["urgency"] not in settings.URGENCY_LEVELS:
        issues.append("bad_urgency")

    fields = parsed.get("fields")
    if "fields" in parsed:
        if not isinstance(fields, dict):
            issues.append("bad_fields")
        else:
            for name in settings.FIELD_NAMES:
                if not _grounded(name, fields.get(name), raw_text):
                    issues.append(f"ungrounded_{name}")

    conf = parsed.get("confidence")
    if "confidence" in parsed:
        if not isinstance(conf, (int, float)) or isinstance(conf, bool):
            issues.append("bad_confidence")
        else:
            level = risk if risk_checks else "standard"
            if conf < settings.CONF_THRESHOLD[level]:
                issues.append("low_confidence")

    if risk_checks and risk == "high" and parsed.get("urgency") in ("low", "normal"):
        issues.append("urgency_conflict")  # risk keywords present but urgency not high
    return issues
