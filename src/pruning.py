"""Deterministic, evidence-preserving context pruning (no LLM call, so no extra compute).

Rules: strip quoted/boilerplate lines and drop duplicate messages. Threads shorter than
LONG_THREAD_MESSAGES are never pruned. In long threads always keep the opening message (it states the
problem) and the latest message, and keep the messages in between only if they carry evidence
(id-like digits, risk or intent keywords). Dev finding: dropping the opening message lost the whole story.
"""
import re

from config import settings

_BOILERPLATE = re.compile(r"^(thanks|thank you|regards|kind regards|best regards|sent from my|--)", re.I)
_MAX_MSG_CHARS = 1000


def _clean(message):
    lines = [ln.strip() for ln in str(message).splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(">") and not _BOILERPLATE.match(ln)]
    text = " ".join(lines)
    if len(text) > _MAX_MSG_CHARS:  # keep head and tail, where ids and requests usually sit
        text = text[: _MAX_MSG_CHARS // 2] + " ... " + text[-_MAX_MSG_CHARS // 2 :]
    return text


def _is_evidence(message):
    low = message.lower()
    if re.search(r"\d{4,}", message):
        return True
    if any(k in low for k in settings.HIGH_RISK_KEYWORDS):
        return True
    return any(k in low for kws in settings.INTENT_KEYWORDS.values() for k in kws)


def _meta_line(ticket):
    meta = ticket.get("metadata") or {}
    return "Metadata: " + ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else ""


def _render(messages, ticket):
    body = "\n".join(f"Message {i + 1}: {m}" for i, m in enumerate(messages))
    meta = _meta_line(ticket)
    return body + ("\n" + meta if meta else "")


def render_raw(ticket):
    """The unpruned prompt body used by the baselines."""
    return _render([str(m) for m in ticket["messages"]], ticket)


def prune(ticket):
    raw_msgs = [str(m) for m in ticket["messages"]]
    cleaned = [_clean(m) for m in raw_msgs]
    last = len(cleaned) - 1
    short = len(cleaned) < settings.LONG_THREAD_MESSAGES
    kept, seen = [], set()
    for i, msg in enumerate(cleaned):
        norm = re.sub(r"\W+", " ", msg.lower()).strip()
        if not msg or (norm in seen and i != last):
            continue
        seen.add(norm)
        if short or i in (0, last) or _is_evidence(msg):
            kept.append(msg)
    text = _render(kept, ticket)
    raw = render_raw(ticket)
    return text, {
        "raw_chars": len(raw), "pruned_chars": len(text),
        "msgs_in": len(raw_msgs), "msgs_kept": len(kept),
    }
