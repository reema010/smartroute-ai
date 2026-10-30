"""Run systems over a split and save raw pipeline outputs (no ground truth is loaded here).

  python -m evaluation.run_eval --split dev  --backend groq
  python -m evaluation.run_eval --split sample --backend mock
"""
import argparse
import json
import sys

from config import settings
from evaluation import analyze
from src.models import LLMClient
from src.pipeline import DEFAULT_SYSTEMS, SYSTEMS, Pipeline


def load_tickets(split):
    path = settings.DATA_DIR / f"{split}_tickets.jsonl"
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def make_backend(name):
    if name == "mock":
        from src.mock_backend import MockBackend
        return MockBackend()
    from src.models import GroqBackend
    return GroqBackend()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="sample | dev | test | val")
    ap.add_argument("--backend", default="groq", choices=["groq", "mock"])
    ap.add_argument("--systems", nargs="*", default=DEFAULT_SYSTEMS, choices=list(SYSTEMS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    try:  # Windows consoles: allow Arabic in progress output
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    tickets = load_tickets(args.split)[: args.limit]
    out_dir = settings.RESULTS_DIR / args.backend / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    client = LLMClient(make_backend(args.backend), settings.CACHE_DIR / f"{args.backend}_calls.jsonl",
                       use_cache=not args.no_cache)
    pipe = Pipeline(client)

    for system in args.systems:
        rows = []
        for i, ticket in enumerate(tickets, 1):
            row = pipe.process(ticket, system)
            rows.append(row)
            print(f"[{system}] {i}/{len(tickets)} {ticket['id']} -> {row['final_status']} ({row['route_status']})", flush=True)
        with open(out_dir / f"{system}.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    analyze.run(args.backend, args.split, args.systems)


if __name__ == "__main__":
    main()
