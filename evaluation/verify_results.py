"""Integrity checks on saved results (no ground truth is read here).

    python -m evaluation.verify_results --backend groq --splits test val dev

For every split and system it checks:
  * the system evaluated exactly the same ticket ids as the ticket file, each exactly once
  * no api_error rows; no JSON-format-failure calls (those have unknown token usage)
  * how many calls needed retries (attempts > 1): failed attempts report no usage, so they are disclosed
  * completion_tokens >= reasoning_tokens on every call (our evidence that reasoning tokens are included, not added twice)
  * per-ticket totals equal the sum of their calls, and each call's cost equals tokens x price in config/settings.py
It exits non-zero if a hard check fails.
"""
import argparse
import json
import sys
from collections import Counter

from config import settings


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="groq")
    ap.add_argument("--splits", nargs="+", default=["test", "val", "dev"])
    a = ap.parse_args()
    hard_fail = False
    price = settings.PRICES_PER_M
    model_tier = {v: k for k, v in settings.MODELS.items()}
    for split in a.splits:
        tpath = settings.DATA_DIR / f"{split}_tickets.jsonl"
        if not tpath.exists():
            continue
        expected = [t["id"] for t in load(tpath)]
        res_dir = settings.RESULTS_DIR / a.backend / split
        print(f"== {split}: {len(expected)} tickets in {tpath.name}")
        for path in sorted(res_dir.glob("*.jsonl")):
            rows = load(path)
            ids = [r["id"] for r in rows]
            dup = [i for i, n in Counter(ids).items() if n > 1]
            missing, extra = sorted(set(expected) - set(ids)), sorted(set(ids) - set(expected))
            api = sum(r["final_status"] == "api_error" for r in rows)
            calls = [c for r in rows for c in r["calls"]]
            retried = sum(c.get("attempts", 1) > 1 for c in calls)
            fmt = sum(bool((c.get("usage_raw") or {}).get("_json_validate_failed")) for c in calls)
            cost_bad = 0
            reasoning_over = sum(c["reasoning_tokens"] > c["completion_tokens"] for c in calls)
            for r in rows:
                if abs(sum(c["cost_usd"] for c in r["calls"]) - r["totals"]["cost_usd"]) > 1e-9:
                    cost_bad += 1
                for c in r["calls"]:
                    p = price[model_tier[c["model"]]]
                    exp = (c["prompt_tokens"] * p["in"] + c["completion_tokens"] * p["out"]) / 1e6
                    if abs(exp - c["cost_usd"]) > 1e-9:
                        cost_bad += 1
            ok = not (dup or missing or extra or cost_bad or reasoning_over) and len(rows) == len(expected)
            hard_fail |= not ok
            print(f"  {path.stem:22} rows={len(rows)} {'OK' if ok else 'PROBLEM'} | missing={missing} extra={extra} duplicates={dup} "
                  f"| api_errors={api} format_failures={fmt} retried_calls={retried} | cost_mismatches={cost_bad} reasoning_gt_completion={reasoning_over} | calls={len(calls)}")
    print("VERIFICATION FAILED" if hard_fail else "VERIFICATION PASSED (read the api_errors / format_failures / retried_calls columns)")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
