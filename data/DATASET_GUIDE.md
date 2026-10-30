# Dataset guide (dev = 40 tickets, test = 80 tickets)

Files (same ids in each pair; runtime code reads ONLY the tickets file):

| File | Content |
|---|---|
| `dev_tickets.jsonl`, `test_tickets.jsonl` | `{"id","messages":[...],"metadata":{...}}` |
| `dev_labels.jsonl`, `test_labels.jsonl` | `{"id","intent","urgency","fields":{"reference_id","amount"},"tags":[...]}` |

Run `python -m evaluation.check_dataset --split dev` (or `all`) after every edit; it must say PASSED.

Rules
1. Labels are written when the ticket is created, so they are correct by construction.
2. Generate with a different model family than the ones being evaluated (Claude/ChatGPT in a chat, not GPT-OSS).
3. Generate dev and test in SEPARATE chats with different scenario lists and wording, so test contains templates and values the dev set never showed. Never copy a dev ticket and change its numbers.
4. Hand-check about 20%. Fix any label you disagree with BEFORE running anything.
5. Never edit test tickets or labels after the first dev run. Tune only on dev.
6. Values in `fields` must appear literally in the ticket text (ids like `ORD-48213`, amounts like `250 SAR`). Use `null` when absent.
7. Use Western digits (0-9) in ids and amounts.

Intents: payment_failed, refund_request, account_access, unauthorized_transaction, service_outage, delivery_issue, billing_question, general_inquiry
Urgency rubric (same meaning the models get): high = money was actually taken or lost (charged but no result, duplicate/wrong charge, deducted with nothing delivered), suspected unauthorized activity or account compromise, or a service fully down; normal = real problem where NO money was taken and it is not a total outage (a declined/failed payment with no deduction is normal; so is a delayed delivery or a plain refund request); low = questions, information requests, feedback.

Tag mix per split (a ticket can have several tags): easy ~45%, noisy (typos, fragments) ~15%, long_thread (5+ messages with chit-chat) ~15%, ambiguous (two issue types) ~10%, high_risk ~20%.
Tickets are in English. `arabic_mixed` is OPTIONAL: skip it, or add about 5 mixed Arabic/English tickets to the TEST split only, as a robustness extra.
Test split: make the high_risk share larger than in dev (e.g. 30% vs 15%) to test distribution shift.

## Prompt to generate a batch (paste into Claude or ChatGPT; for the test split, ask for different scenarios)

> Generate 20 realistic customer-support tickets for a Saudi telecom/payments company as JSONL, plus a matching labels JSONL. Ticket format: {"id","messages":[1-6 customer/agent messages as strings],"metadata":{"channel":"app|web|phone"}}. Label format: {"id","intent","urgency","fields":{"reference_id","amount"},"tags":[...]}. Intents: payment_failed, refund_request, account_access, unauthorized_transaction, service_outage, delivery_issue, billing_question, general_inquiry. Urgency: high = money actually taken or lost (charged but no result, duplicate/wrong charge), suspected unauthorized activity/account compromise, or service fully down; normal = real problem with NO money taken and no total outage (a failed or declined payment with no deduction is normal; delayed delivery and plain refund requests are normal); low = questions/feedback. reference_id and amount values must appear verbatim in the ticket text (Western digits); use null if absent. Write all tickets in English. Include this mix: 9 easy, 3 noisy (typos), 3 long threads with irrelevant chit-chat and repeated messages, 2 ambiguous with two competing issues (label the main one), 4 high-risk. Use ids {prefix}001... Return two code blocks only.
