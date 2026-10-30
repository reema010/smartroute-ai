"""One prompt shared by every system and every model, so comparisons are fair."""
from config import settings

_FIELDS_SCHEMA = ", ".join(f'"{name}": <string or null>' for name in settings.FIELD_NAMES)

SYSTEM_PROMPT = f"""You are a customer-support ticket triage engine.
Read the ticket and reply with ONE JSON object and nothing else.

Schema:
{{"intent": <one of {settings.INTENTS}>,
 "urgency": <one of {settings.URGENCY_LEVELS}>,
 "fields": {{{_FIELDS_SCHEMA}}},
 "confidence": <number from 0 to 1: how sure you are about intent and urgency>}}

Intent definitions:
- payment_failed: a payment, top-up or purchase did not go through, or money was taken but the service, balance or bundle was not delivered.
- unauthorized_transaction: a charge or money movement the customer says they did not make or approve, including fraud, phishing and unexplained missing funds they did not spend.
- account_access: cannot log in, locked out, password or OTP problems, or an account/SIM takeover where the main problem is losing access.
- refund_request: the customer asks for money back for something they bought or cancelled.
- service_outage: mobile, internet or app service is down or badly degraded.
- delivery_issue: a physical order, SIM or device is late, lost or wrong.
- billing_question: questions about an invoice, plan charges or fees on an existing bill.
- general_inquiry: questions about plans, features, policies or switching, before any problem exists.
Choose the single main problem the customer is asking us to solve.

Urgency rubric:
- high: money was actually taken or lost (charged but the service/refund did not arrive, duplicate or wrong charge, money deducted with no result), suspected unauthorized activity or account compromise, or a service that is fully down for the customer.
- normal: a real problem that needs a fix but no money has been taken or lost, and it is not a total outage. Examples: a payment that was declined or failed with no deduction, a refund asked for a purchase the customer chose to cancel, a delayed delivery, a login problem with no sign of compromise.
- low: questions, information requests, feedback.
A failed or declined payment is "normal" unless the ticket says money was deducted.

Field rules: copy values exactly as written in the ticket, from any message (customer or agent). Use null only when the ticket really contains no such value. Never invent values.
- reference_id: the identifier of the thing the ticket is about. Any order, transaction, invoice, account, case, subscription, authorization or reference number or code counts. If several appear, choose the one tied to the main problem.
- amount: the main money amount involved (charged, missing, refunded, billed or disputed). A number followed by SAR, riyal or a currency word is an amount, and so is a bare number the customer pays, is charged or wants back. Copy it as written, with the currency if one is written.
Tickets may mix Arabic and English."""


def build_messages(ticket_text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ticket_text},
    ]


# Strict structured-output schema, built from the same settings as the prompt and the validator.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": settings.INTENTS},
        "urgency": {"type": "string", "enum": settings.URGENCY_LEVELS},
        "fields": {
            "type": "object",
            "properties": {n: {"type": ["string", "null"]} for n in settings.FIELD_NAMES},
            "required": list(settings.FIELD_NAMES),
            "additionalProperties": False,
        },
        "confidence": {"type": "number"},
    },
    "required": ["intent", "urgency", "fields", "confidence"],
    "additionalProperties": False,
}


def build_response_format(kind):
    if kind == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": "ticket_triage", "strict": True, "schema": RESPONSE_SCHEMA}}
    return {"type": "json_object"}
