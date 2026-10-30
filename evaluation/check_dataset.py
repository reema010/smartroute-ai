"""Validate dataset files BEFORE running anything. Catches schema, label and leakage mistakes early.

    python -m evaluation.check_dataset --split dev
    python -m evaluation.check_dataset --split all

Exit code 1 if there are errors. Warnings are things worth a human look (e.g. a probable missing label).
"""
import argparse
import json
import re
import sys
from collections import Counter

from config import settings
from src.pruning import render_raw
from src.verification import _grounded

KNOWN_TAGS = {"easy", "noisy", "arabic_mixed", "long_thread", "ambiguous", "high_risk"}
EXPECTED_SIZE = {"dev": 40, "test": 80}
TICKET_KEYS = {"id", "messages", "metadata"}
LABEL_KEYS_IN_TICKET = {"intent", "urgency", "fields", "tags", "label", "labels", "gold"}
ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")
ARABIC_INDIC_DIGITS = re.compile(r"[٠-٩۰-۹]")
ID_LIKE = re.compile(r"\b[A-Z]{2,5}-\d{3,}\b")
AMOUNT_LIKE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:SAR|SR|ريال)", re.I)


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _template(ticket):
    text = " ".join(map(str, ticket["messages"])).lower()
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text)).strip()


def check(split, data_dir=None):
    data_dir = data_dir or settings.DATA_DIR
    errors, warnings = [], []
    tickets = _load(data_dir / f"{split}_tickets.jsonl")
    labels = _load(data_dir / f"{split}_labels.jsonl")

    t_ids = [t.get("id") for t in tickets]
    l_ids = [l.get("id") for l in labels]
    for name, ids in (("tickets", t_ids), ("labels", l_ids)):
        dup = [i for i, c in Counter(ids).items() if c > 1]
        if dup:
            errors.append(f"duplicate ids in {name}: {dup}")
    if set(t_ids) != set(l_ids):
        errors.append(f"ids differ: only in tickets {sorted(set(t_ids) - set(l_ids))}, only in labels {sorted(set(l_ids) - set(t_ids))}")
    if split in EXPECTED_SIZE and len(tickets) != EXPECTED_SIZE[split]:
        warnings.append(f"{split} has {len(tickets)} tickets, plan is {EXPECTED_SIZE[split]}")

    by_id = {t.get("id"): t for t in tickets}
    for t in tickets:
        tid = t.get("id")
        if not isinstance(t.get("messages"), list) or not t["messages"] or not all(isinstance(m, str) and m.strip() for m in t["messages"]):
            errors.append(f"{tid}: messages must be a non-empty list of non-empty strings")
        leaked = LABEL_KEYS_IN_TICKET & set(t)
        if leaked:
            errors.append(f"{tid}: label-like keys inside the ticket file {sorted(leaked)} (leakage)")
        extra = set(t) - TICKET_KEYS - LABEL_KEYS_IN_TICKET
        if extra:
            warnings.append(f"{tid}: unexpected ticket keys {sorted(extra)}")
        if isinstance(t.get("metadata"), dict) and LABEL_KEYS_IN_TICKET & set(t["metadata"]):
            errors.append(f"{tid}: label-like keys inside metadata (leakage)")

    for g in labels:
        gid = g.get("id")
        if g.get("intent") not in settings.INTENTS:
            errors.append(f"{gid}: intent {g.get('intent')!r} not in {settings.INTENTS}")
        if g.get("urgency") not in settings.URGENCY_LEVELS:
            errors.append(f"{gid}: urgency {g.get('urgency')!r} not in {settings.URGENCY_LEVELS}")
        fields = g.get("fields")
        if not isinstance(fields, dict) or set(fields) != set(settings.FIELD_NAMES):
            errors.append(f"{gid}: fields must have exactly the keys {settings.FIELD_NAMES}, got {fields if not isinstance(fields, dict) else sorted(fields)}")
            fields = fields if isinstance(fields, dict) else {}
        tags = g.get("tags", [])
        if not isinstance(tags, list) or set(tags) - KNOWN_TAGS:
            errors.append(f"{gid}: tags must be a list drawn from {sorted(KNOWN_TAGS)}, got {tags!r}")
            tags = []
        extra_label_keys = set(g) - {"id", "intent", "urgency", "fields", "tags"}
        if extra_label_keys:
            warnings.append(f"{gid}: unused label keys {sorted(extra_label_keys)}")

        ticket = by_id.get(gid)
        if ticket is None or not isinstance(ticket.get("messages"), list):
            continue
        raw = render_raw(ticket)
        for name in settings.FIELD_NAMES:
            value = fields.get(name)
            if value is not None and not isinstance(value, str):
                errors.append(f"{gid}: field {name} must be a string or null")
            elif value not in (None, "") and not _grounded(name, value, raw):
                errors.append(f"{gid}: field {name}={value!r} does not appear in the ticket text")
        if fields.get("reference_id") in (None, "") and ID_LIKE.search(raw):
            warnings.append(f"{gid}: ticket has an id-like token {ID_LIKE.search(raw).group(0)!r} but reference_id is null")
        if fields.get("amount") in (None, "") and AMOUNT_LIKE.search(raw):
            warnings.append(f"{gid}: ticket has an amount-like token {AMOUNT_LIKE.search(raw).group(0)!r} but amount is null")
        if ARABIC_INDIC_DIGITS.search(raw):
            warnings.append(f"{gid}: Arabic-Indic digits present; grounding and scoring compare Western digits only")
        if "arabic_mixed" in tags and not (ARABIC.search(raw) and LATIN.search(raw)):
            warnings.append(f"{gid}: tagged arabic_mixed but the text is not mixed-script")
        if "long_thread" in tags and len(ticket["messages"]) < 5:
            warnings.append(f"{gid}: tagged long_thread but has {len(ticket['messages'])} messages")
        if "high_risk" in tags and g.get("urgency") != "high":
            warnings.append(f"{gid}: tagged high_risk but urgency is {g.get('urgency')!r}")
        if g.get("intent") in ("unauthorized_transaction", "service_outage") and g.get("urgency") != "high":
            warnings.append(f"{gid}: {g.get('intent')} with urgency {g.get('urgency')!r} (rubric usually says high)")

    summary = {
        "n": len(tickets),
        "intent": dict(Counter(g.get("intent") for g in labels)),
        "urgency": dict(Counter(g.get("urgency") for g in labels)),
        "tags": dict(Counter(tag for g in labels for tag in (g.get("tags") or []))),
    }
    return errors, warnings, summary


def check_cross_split(data_dir=None):
    """Splits must not share tickets or templates (same sentence, different numbers): dev/test, and val vs both."""
    data_dir = data_dir or settings.DATA_DIR
    errors, warnings = [], []
    for earlier, later in (("dev", "test"), ("dev", "val"), ("test", "val")):
        a_path, b_path = data_dir / f"{earlier}_tickets.jsonl", data_dir / f"{later}_tickets.jsonl"
        if not (a_path.exists() and b_path.exists()):
            continue
        a, b = _load(a_path), _load(b_path)
        a_exact = {" ".join(map(str, t["messages"])).strip().lower() for t in a}
        a_tpl = {_template(t) for t in a}
        for t in b:
            exact = " ".join(map(str, t["messages"])).strip().lower()
            if exact in a_exact:
                errors.append(f"{t['id']}: identical to a {earlier} ticket")
            elif _template(t) in a_tpl:
                warnings.append(f"{t['id']}: same template as a {earlier} ticket with different numbers")
    return errors, warnings


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="sample | dev | test | val | all")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    splits = ["dev", "test"] if args.split == "all" else [args.split]
    if args.split == "all" and (settings.DATA_DIR / "val_tickets.jsonl").exists():
        splits.append("val")
    failed = False
    for split in splits:
        errors, warnings, summary = check(split)
        print(f"== {split}: {summary['n']} tickets | intents {summary['intent']} | urgency {summary['urgency']} | tags {summary['tags']}")
        for w in warnings:
            print("  WARN ", w)
        for e in errors:
            print("  ERROR", e)
        failed |= bool(errors)
    if args.split == "all":
        errors, warnings = check_cross_split()
        for w in warnings:
            print("  WARN ", w)
        for e in errors:
            print("  ERROR", e)
        failed |= bool(errors)
    print("DATASET CHECK FAILED" if failed else "DATASET CHECK PASSED (review any warnings)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
