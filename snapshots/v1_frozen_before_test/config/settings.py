"""Central configuration. Change values here, not inside the pipeline code.

Anything marked TUNE is set on the DEV split only, then frozen before the TEST run.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"

# ---------------------------------------------------------------- models ---
MODELS = {"small": "openai/gpt-oss-20b", "large": "openai/gpt-oss-120b"}

# USD per 1M tokens. Sources disagree (Groq's launch post vs. later price pages),
# so VERIFY at https://console.groq.com/docs/pricing and edit before reporting costs.
PRICES_PER_M = {
    "small": {"in": 0.075, "out": 0.30},
    "large": {"in": 0.15, "out": 0.60},
}

# Same effort on both models = controlled comparison. Optimise later (e.g. small="low", large="medium").
REASONING_EFFORT = {"small": "low", "large": "low"}
# "json_schema" = strict structured output (same schema for both models). "json_object" = plain JSON mode:
# on the real smoke test it made gpt-oss-20b fail with HTTP 400 json_validate_failed on a mixed Arabic/English ticket.
RESPONSE_FORMAT = "json_schema"
TEMPERATURE = 0.0
MAX_COMPLETION_TOKENS = 1000  # reasoning models need headroom; the smoke test shows real usage
MAX_RETRIES = 5

# Client-side throttle (sliding 60 s window). Defaults follow the published free-tier table
# (30 req/min, 8,000 tokens/min per GPT-OSS model): CHECK your own limits in the Groq console
# and edit. Use None to disable a limit.
RATE_LIMITS = {"rpm": 30, "tpm": 8000}

# ----------------------------------------------------------- task schema ---
INTENTS = [
    "payment_failed",
    "refund_request",
    "account_access",
    "unauthorized_transaction",
    "service_outage",
    "delivery_issue",
    "billing_question",
    "general_inquiry",
]
URGENCY_LEVELS = ["low", "normal", "high"]
FIELD_NAMES = ["reference_id", "amount"]  # order/transaction id and amount (kept to two for labelling speed)

# ------------------------------------------------- gate / router (TUNE) ---
CONF_THRESHOLD = {"standard": 0.70, "high": 0.85}  # escalate if confidence below this
ROUTER_LARGE_SCORE = 2  # start on the large model when complexity score >= this
LONG_TICKET_CHARS = 800
LONG_THREAD_MESSAGES = 4

# Lower-case substring match. Deliberately small and readable.
HIGH_RISK_KEYWORDS = [
    "unauthorized", "unauthorised", "fraud", "hacked", "stolen", "compromised",
    "charged twice", "double charged", "deducted", "outage", "completely down",
    # "خصم" alone is ambiguous (deduction vs discount), so only unambiguous deduction phrases are listed
    "احتيال", "اختراق", "مسروق", "غير مصرح", "انخصم", "تم خصم", "انقطاع",
]

# Used ONLY by the pre-router to spot tickets that mix several issue types.
INTENT_KEYWORDS = {
    "payment_failed": ["payment failed", "payment declined", "card declined", "transaction failed", "فشل الدفع", "فشلت"],
    "refund_request": ["refund", "money back", "استرجاع", "استرداد"],
    "account_access": ["login", "log in", "password", "locked out", "sign in", "تسجيل الدخول", "كلمة المرور"],
    "unauthorized_transaction": ["unauthorized", "unauthorised", "didn't make", "fraud", "غير مصرح", "احتيال"],
    "service_outage": ["outage", "is down", "not working", "no signal", "انقطاع", "لا يعمل"],
    "delivery_issue": ["delivery", "package", "courier", "shipment", "التوصيل", "الشحنة"],
    "billing_question": ["invoice", "bill", "فاتورة"],
}
