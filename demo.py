"""One-ticket live demo: runs the same ticket through the baselines and SmartRoute and prints a side-by-side view.

    python demo.py "my payment failed, order ORD-48213, 250 SAR"
    python demo.py "hi i paid 35 for esim pack money gone frm my acc but no pack txn TX88231"
    python demo.py "first message || second message || third message"   # ' || ' separates messages of one thread
    python demo.py --examples                                           # three built-in example tickets
    python demo.py --examples --systems always_small always_large smartroute smartroute_v2
    python demo.py --examples --mock                                    # offline plumbing check (no real model)

Live runs call Groq (needs GROQ_API_KEY in .env). Nothing is cached, so every run is a fresh, timed call.
This script never reads labels; it is for showing the system, not for evaluation.
"""
import argparse
import sys

from config import settings
from evaluation.run_eval import make_backend
from src.models import LLMClient
from src.pipeline import DEFAULT_SYSTEMS, SYSTEMS, Pipeline

EXAMPLES = [
    "Hello, my postpaid invoice INV-2026-0918 is 289.50 SAR but my plan is 200. Can you explain the extra charges?",
    "hi i paid 35 for the 10gb esim pack like 2 hrs ago money gone frm my acc but no pack nothing || txn TX88231 pls fix asap",
    "Hi, I was charged 120 SAR twice for order ORD-55012. Please refund the second charge.",
]


def show(ticket, systems, pipe):
    print("=" * 100)
    print("TICKET:", " / ".join(ticket["messages"]))
    large_cost = None
    rows = []
    for name in systems:
        r = pipe.process(ticket, name)
        rows.append((name, r))
        if name == "always_large":
            large_cost = r["totals"]["cost_usd"]
    for name, r in rows:
        p = r["prediction"] or {}
        t = r["totals"]
        route = "-" if name.startswith("always") else f"start={r['start_tier']} escalated={r['escalated']}"
        print(f"\n[{name}]")
        print(f"  intent={p.get('intent')}  urgency={p.get('urgency')}  fields={p.get('fields')}  confidence={p.get('confidence')}")
        print(f"  status={r['final_status']} ({r['route_status']})  route: {route}" + (f"  issues={r['issues']}" if r["issues"] else ""))
        saving = "" if not large_cost else f"  ({100 * (1 - t['cost_usd'] / large_cost):.0f}% cheaper than always_large)"
        print(f"  calls={t['n_calls']}  tokens={t['prompt_tokens'] + t['completion_tokens']}  cost=${t['cost_usd']:.6f}{saving}  latency={t['latency_s']:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticket", nargs="?", help="ticket text; use ' || ' between messages")
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--systems", nargs="*", default=DEFAULT_SYSTEMS, choices=list(SYSTEMS))
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    if not a.ticket and not a.examples:
        ap.error("give a ticket text or --examples")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    texts = EXAMPLES if a.examples else [a.ticket]
    pipe = Pipeline(LLMClient(make_backend("mock" if a.mock else "groq"), settings.CACHE_DIR / "demo_unused.jsonl", use_cache=False))
    for i, text in enumerate(texts, 1):
        show({"id": f"demo{i}", "messages": [m.strip() for m in text.split(" || ")], "metadata": {}}, a.systems, pipe)
    print("\nNote: 'validated' means the deterministic checks passed, not that the answer is guaranteed correct.")


if __name__ == "__main__":
    main()
