"""Print ticket text, gold label and every system's prediction for chosen ids (dev inspection only).

    python -m evaluation.show_failures --split dev --backend groq --ids d019 d028 d030
"""
import argparse
import json
import sys

from config import settings


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--backend", default="groq")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--systems", nargs="+", default=["always_small", "always_large", "smartroute"])
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    tickets = {t["id"]: t for t in load(settings.DATA_DIR / f"{a.split}_tickets.jsonl")}
    gold = {g["id"]: g for g in load(settings.DATA_DIR / f"{a.split}_labels.jsonl")}
    runs = {s: {r["id"]: r for r in load(settings.RESULTS_DIR / a.backend / a.split / f"{s}.jsonl")} for s in a.systems}
    for i in a.ids:
        g = gold[i]
        print("=" * 78)
        print(f"{i}  tags={g.get('tags')}")
        print("TICKET:", " / ".join(tickets[i]["messages"])[:700])
        print(f"GOLD  : {g['intent']} | {g['urgency']} | {g['fields']}")
        for s in a.systems:
            r = runs[s][i]
            p = r["prediction"] or {}
            ok = "OK " if (p.get("intent") == g["intent"] and p.get("urgency") == g["urgency"]) else "BAD"
            extra = ""
            if s == "smartroute":
                extra = f" | start={r['start_tier']} esc={r['escalated']} issues={r['issues']} prune={r['prune']['msgs_kept']}/{r['prune']['msgs_in']} msgs, {r['prune']['pruned_chars']}/{r['prune']['raw_chars']} chars"
            print(f"{s:12}: {ok} {p.get('intent')} | {p.get('urgency')} | {p.get('fields')} | conf={p.get('confidence')} | {r['final_status']}{extra}")


if __name__ == "__main__":
    main()
