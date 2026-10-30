"""Score saved outputs against ground truth and write metrics.csv, per_ticket.csv, by_tag.csv, report.html.

This is the ONLY place ground truth is read.
API errors are infrastructure failures: they are counted separately (`api_errors`) and we report
success both with them counted as failures (`task_success`) and without them (`success_excl_api_errors`).
A final result should have zero API errors: rerun (the cache makes this cheap) until it does.
"""
import csv
import html
import json
import math
import re
import statistics

from config import settings


def _load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _numbers(value):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value).replace(",", ""))]


def _field_eq(name, pred, gold):
    if pred in (None, "") and gold in (None, ""):
        return True
    if pred in (None, "") or gold in (None, ""):
        return False
    if name == "amount":
        return _numbers(pred) == _numbers(gold)
    norm = lambda v: re.sub(r"[\s\-_]", "", str(v).lower())
    return norm(pred) == norm(gold)


def score_one(pred, gold):
    if not isinstance(pred, dict):
        return {"intent_ok": False, "urgency_ok": False, "fields_ok": False, "success": False}
    fields = pred.get("fields") if isinstance(pred.get("fields"), dict) else {}
    intent_ok = pred.get("intent") == gold["intent"]
    urgency_ok = pred.get("urgency") == gold["urgency"]
    fields_ok = all(_field_eq(k, fields.get(k), (gold.get("fields") or {}).get(k)) for k in settings.FIELD_NAMES)
    return {"intent_ok": intent_ok, "urgency_ok": urgency_ok, "fields_ok": fields_ok,
            "success": intent_ok and urgency_ok and fields_ok}


def _p95(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _rate(count, total):
    return count / total if total else None


def run(backend, split, systems):
    res_dir = settings.RESULTS_DIR / backend / split
    gold = {g["id"]: g for g in _load_jsonl(settings.DATA_DIR / f"{split}_labels.jsonl")}
    per_ticket, metrics, by_tag = [], [], {}

    for system in systems:
        rows = _load_jsonl(res_dir / f"{system}.jsonl")
        scored = []
        for r in rows:
            g = gold[r["id"]]
            s = score_one(r["prediction"], g)
            t = r["totals"]
            rec = {
                "id": r["id"], "system": system, "final_status": r["final_status"],
                "route_status": r["route_status"], "escalated": r["escalated"],
                "start_tier": r["start_tier"], "tags": "|".join(g.get("tags", [])),
                "gold_urgency": g["urgency"], "pred_urgency": (r["prediction"] or {}).get("urgency"),
                **s, "format_failures": sum(1 for c in r["calls"] if (c.get("usage_raw") or {}).get("_json_validate_failed")),
                "n_calls": t["n_calls"], "cached_calls": t.get("cached_calls", 0),
                "tokens": t["prompt_tokens"] + t["completion_tokens"],
                "reasoning_tokens": t["reasoning_tokens"], "cost_usd": t["cost_usd"], "latency_s": t["latency_s"],
            }
            scored.append(rec)
            per_ticket.append(rec)
            for tag in g.get("tags", []):
                by_tag.setdefault((system, tag), []).append(s["success"])

        n = len(scored)
        ok = [x for x in scored if x["final_status"] != "api_error"]
        wins = sum(x["success"] for x in scored)
        validated = [x for x in scored if x["final_status"] == "validated"]
        high = [x for x in scored if x["gold_urgency"] == "high"]
        total_calls = sum(x["n_calls"] for x in scored)
        total_cost = sum(x["cost_usd"] for x in scored)
        lat = [x["latency_s"] for x in ok]
        metrics.append({
            "system": system, "n": n,
            "api_errors": n - len(ok),
            "format_failure_calls": sum(x["format_failures"] for x in scored),  # model returned invalid JSON; usage unknown, cost counted as 0
            "task_success": wins / n,
            "success_excl_api_errors": _rate(sum(x["success"] for x in ok), len(ok)),
            "intent_acc": sum(x["intent_ok"] for x in scored) / n,
            "urgency_acc": sum(x["urgency_ok"] for x in scored) / n,
            "fields_acc": sum(x["fields_ok"] for x in scored) / n,
            "high_urgency_recall": _rate(sum(x["pred_urgency"] == "high" for x in high), len(high)),
            "validated_rate": sum(x["final_status"] == "validated" for x in scored) / n,
            "unverified_rate": sum(x["final_status"] == "unverified" for x in scored) / n,
            "needs_review_rate": sum(x["final_status"] == "needs_review" for x in scored) / n,
            "validated_precision": _rate(sum(x["success"] for x in validated), len(validated)),
            "escalation_rate": sum(x["escalated"] for x in scored) / n,
            "avg_calls": statistics.mean(x["n_calls"] for x in scored),
            "avg_tokens": statistics.mean(x["tokens"] for x in ok) if ok else None,
            "avg_reasoning_tokens": statistics.mean(x["reasoning_tokens"] for x in ok) if ok else None,
            "total_cost_usd": total_cost,
            "cost_per_success_usd": (total_cost / wins) if wins else None,
            "cost_savings_vs_large_pct": None,
            "median_latency_s": statistics.median(lat) if lat else None,
            "p95_latency_s": _p95(lat),
            "cached_call_share": _rate(sum(x["cached_calls"] for x in scored), total_calls),
        })

    large = next((m for m in metrics if m["system"] == "always_large"), None)
    if large and large["total_cost_usd"]:
        for m in metrics:
            m["cost_savings_vs_large_pct"] = 100 * (1 - m["total_cost_usd"] / large["total_cost_usd"])

    _write_csv(res_dir / "metrics.csv", metrics)
    _write_csv(res_dir / "per_ticket.csv", per_ticket)
    tag_rows = [{"system": s, "tag": t, "n": len(v), "task_success": sum(v) / len(v)} for (s, t), v in sorted(by_tag.items())]
    _write_csv(res_dir / "by_tag.csv", tag_rows)
    _write_html(res_dir / "report.html", backend, split, metrics, tag_rows)
    print(f"Wrote {res_dir / 'metrics.csv'} and {res_dir / 'report.html'}")
    return metrics


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fmt(v):
    if v is None:
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _table(rows):
    if not rows:
        return ""
    head = "".join(f"<th>{html.escape(k)}</th>" for k in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{html.escape(_fmt(v))}</td>" for v in r.values()) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _bars(rows, key, title, pct=False):
    vals = [(r["system"], r[key]) for r in rows if r[key] is not None]
    top = max((v for _, v in vals), default=1) or 1
    out = [f"<h3>{html.escape(title)}</h3>"]
    for name, v in vals:
        w = int(100 * v / top)
        label = f"{v * 100:.1f}%" if pct else f"{v:.5f}"
        out.append(f'<div class="row"><span>{html.escape(name)}</span><div class="bar" style="width:{w}%"></div><b>{label}</b></div>')
    return "".join(out)


def _write_html(path, backend, split, metrics, tag_rows):
    warn = "<p class='warn'>MOCK BACKEND: plumbing test only. These numbers are not model results.</p>" if backend == "mock" else ""
    note = ("<p>final_status: <b>validated</b> = deterministic checks passed (not a guarantee of correctness); "
            "<b>unverified</b> = no verification run or only low confidence remains; <b>needs_review</b> = a hard check failed; "
            "<b>api_error</b> = infrastructure failure, reported separately.</p>")
    page = f"""<!doctype html><meta charset="utf-8"><title>SmartRoute report</title>
<style>body{{font-family:sans-serif;margin:24px;max-width:1200px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}
.row{{display:flex;align-items:center;gap:8px;margin:3px 0}}.row span{{width:190px}}.bar{{height:14px;background:#1D9E75}}.warn{{color:#a32d2d;font-weight:bold}}</style>
<h1>SmartRoute AI - {html.escape(split)} ({html.escape(backend)})</h1>{warn}{note}
{_bars(metrics, "task_success", "Task success (intent + urgency + fields all correct)", pct=True)}
{_bars(metrics, "cost_per_success_usd", "Cost per successful ticket (USD)")}
{_bars(metrics, "median_latency_s", "Median latency (s), original live measurements")}
<h2>All metrics</h2>{_table(metrics)}
<h2>Success by ticket tag</h2>{_table(tag_rows)}"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
