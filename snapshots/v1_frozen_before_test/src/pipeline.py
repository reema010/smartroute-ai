"""End-to-end flow. This module only sees ticket text; scoring lives in evaluation/.

  ticket -> [prune] -> [pre-route] -> start model -> parse+validate
        -> (small model AND issues AND gate on) -> escalate ONCE to large -> parse+validate
        -> route_status (detailed) + final_status (validated | unverified | needs_review | api_error)
Every model call is recorded and summed into the per-ticket totals.

final_status meaning:
  validated    all deterministic checks passed (this is NOT a claim that the answer is correct)
  unverified   no verification was run (baselines), or only a soft signal remains (low confidence)
  needs_review a hard check still fails after the final model (bad JSON/label, ungrounded value, urgency conflict)
  api_error    the model call failed; this is an infrastructure failure, not a prediction
"""
from dataclasses import dataclass

from src.models import BackendError
from src.prompts import build_messages
from src.pruning import prune, render_raw
from src.router import assess_risk, route
from src.verification import SOFT_ISSUES, parse_output, validate


@dataclass(frozen=True)
class SystemConfig:
    name: str
    start: str        # "small" | "large" | "router"
    pruning: bool
    gate: bool        # escalate on validation/uncertainty issues
    risk_checks: bool  # stricter thresholds and urgency check for high-risk tickets


SYSTEMS = {
    "always_small": SystemConfig("always_small", "small", False, False, False),
    "always_large": SystemConfig("always_large", "large", False, False, False),
    "smartroute": SystemConfig("smartroute", "router", True, True, True),
    # optional fourth comparison, run with: --systems always_small always_large smartroute smartroute_no_pruning
    "smartroute_no_pruning": SystemConfig("smartroute_no_pruning", "router", False, True, True),
}
DEFAULT_SYSTEMS = ["always_small", "always_large", "smartroute"]


def decide_final_status(gate, error, issues):
    if error:
        return "api_error"
    if not gate:
        return "unverified"
    if not issues:
        return "validated"
    hard = [i for i in issues if i not in SOFT_ISSUES]
    return "needs_review" if hard else "unverified"


def decide_route_status(gate, error, start, escalated, issues):
    if error:
        return "api_error"
    if not gate:
        return "unchecked"
    if issues:
        return "unresolved_after_escalation" if escalated else "unresolved_large_direct"
    if escalated:
        return "escalated_verified"
    return "verified_large_direct" if start == "large" else "verified_small"


class Pipeline:
    def __init__(self, client):
        self.client = client

    def _run_tier(self, tier, text, raw, risk, cfg, calls):
        record = self.client.call(tier, build_messages(text))  # may raise BackendError
        calls.append(record)
        parsed, issue = parse_output(record.text)
        issues = [issue] if issue else validate(parsed, raw, risk, cfg.risk_checks)
        return parsed, issues

    def process(self, ticket, system):
        cfg = SYSTEMS[system]
        raw = render_raw(ticket)
        risk = assess_risk(raw)
        if cfg.pruning:
            text, prune_stats = prune(ticket)
        else:
            text, prune_stats = raw, {"raw_chars": len(raw), "pruned_chars": len(raw),
                                      "msgs_in": len(ticket["messages"]), "msgs_kept": len(ticket["messages"])}
        route_info = route(ticket, raw, risk) if cfg.start == "router" else {}
        start = route_info.get("start", cfg.start)

        calls, error, escalated = [], None, False
        parsed, issues = None, []
        try:
            parsed, issues = self._run_tier(start, text, raw, risk, cfg, calls)
            if cfg.gate and start == "small" and issues:
                escalated = True  # at most once; a large-model start never escalates
                parsed, issues = self._run_tier("large", text, raw, risk, cfg, calls)
        except BackendError as exc:
            error = str(exc)

        return {
            "id": ticket["id"], "system": system, "prediction": parsed,
            "final_status": decide_final_status(cfg.gate, error, issues),
            "route_status": decide_route_status(cfg.gate, error, start, escalated, issues),
            "issues": issues, "start_tier": start, "escalated": escalated, "risk": risk,
            "route": route_info, "prune": prune_stats, "error": error,
            "calls": [c.to_dict() for c in calls],
            "totals": {
                "n_calls": len(calls),
                "prompt_tokens": sum(c.prompt_tokens for c in calls),
                "completion_tokens": sum(c.completion_tokens for c in calls),
                "reasoning_tokens": sum(c.reasoning_tokens for c in calls),
                "cost_usd": sum(c.cost_usd for c in calls),
                "latency_s": sum(c.latency_s for c in calls),
                "cached_calls": sum(c.cached for c in calls),
            },
        }
