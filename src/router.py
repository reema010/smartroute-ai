"""Heuristic pre-router: decides which model a ticket STARTS on. Uses only the ticket text."""
import re

from config import settings

_ARABIC = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[A-Za-z]")


def assess_risk(raw_text):
    low = raw_text.lower()
    return "high" if any(k in low for k in settings.HIGH_RISK_KEYWORDS) else "standard"


def route(ticket, raw_text, risk):
    low = raw_text.lower()
    n_issue_types = sum(any(k in low for k in kws) for kws in settings.INTENT_KEYWORDS.values())
    features = {
        "long_text": len(raw_text) > settings.LONG_TICKET_CHARS,
        "long_thread": len(ticket["messages"]) > settings.LONG_THREAD_MESSAGES,
        "mixed_script": bool(_ARABIC.search(raw_text) and _LATIN.search(raw_text)),
        "multi_issue": n_issue_types >= 2,
        "high_risk": risk == "high",
    }
    score = sum(features.values())
    start = "large" if score >= settings.ROUTER_LARGE_SCORE else "small"
    return {"start": start, "score": score, "features": features}
