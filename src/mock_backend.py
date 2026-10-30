"""Offline stand-in for Groq. Used ONLY to test the plumbing (routing, escalation, caching,
logging). Its answers say nothing about real model quality; never report mock results.
"""
import json
import re

from config import settings
from src.models import BackendError

_KEYWORDS = [  # priority order for the 'large' mock
    ("unauthorized_transaction", ["unauthorized", "fraud", "غير مصرح", "احتيال"]),
    ("payment_failed", ["payment failed", "card declined", "فشل الدفع"]),
    ("refund_request", ["refund", "money back", "استرجاع"]),
    ("account_access", ["login", "password", "locked out"]),
    ("service_outage", ["outage", "is down", "not working"]),
    ("delivery_issue", ["delivery", "package", "courier"]),
    ("billing_question", ["invoice", "bill"]),
]


def _matches(text):
    t = text.lower()
    return [intent for intent, kws in _KEYWORDS if any(k in t for k in kws)]


def _fields(text):
    ref = re.search(r"\b[A-Z]{2,4}-\d{4,}\b", text)
    amt = re.search(r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:SAR|SR)\b", text)
    return {
        "reference_id": ref.group(0) if ref else None,
        "amount": amt.group(0) if amt else None,
    }


class MockBackend:
    name = "mock"

    def complete(self, model, messages, params):
        user = messages[-1]["content"]
        if "[[RATE_LIMIT]]" in user:
            raise BackendError("mock rate limit")
        small = model == settings.MODELS["small"]
        found = _matches(user)
        urgent = "urgent" in user.lower() or "asap" in user.lower()

        if small:
            if "[[BAD_JSON]]" in user:
                text = "I think this is a payment issue."  # malformed on purpose
                return text, self._usage(user, text, small), 0.4
            ambiguous = len(found) != 1
            intent = found[0] if not ambiguous else "general_inquiry"
            urgency = "high" if urgent else "normal"  # the weak model misses implicit urgency
            conf = 0.55 if ambiguous else 0.92
        else:
            intent = found[0] if found else "general_inquiry"
            urgency = "high" if (urgent or intent in ("unauthorized_transaction", "service_outage")) else "normal"
            conf = 0.93

        text = json.dumps({"intent": intent, "urgency": urgency, "fields": _fields(user), "confidence": conf})
        return text, self._usage(user, text, small), (0.4 if small else 1.1)

    @staticmethod
    def _usage(prompt, completion, small):
        p = len(prompt) // 4 + 150
        c = len(completion) // 4 + (30 if small else 60)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
                "completion_tokens_details": {"reasoning_tokens": 30 if small else 60}}
