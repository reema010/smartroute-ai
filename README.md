# SmartRoute AI

**Challenge 2: Deliver AI with Less Compute.** Customer-support ticket triage (intent, urgency, extracted fields) that
uses a small model by default and spends the larger model only where it is needed. Built on Groq with
`openai/gpt-oss-20b` (small) and `openai/gpt-oss-120b` (large). Every call's tokens, reasoning tokens, latency,
attempts and cost are recorded, so the cost/quality trade-off is measured, not assumed.

## Result in one paragraph
On the **official frozen 80-ticket held-out test**, SmartRoute v1 achieved **81.25% complete-ticket success** versus
**78.75%** for Always-Small and **90.0%** for Always-Large, while reducing estimated inference cost by **38.7%**
relative to Always-Large. It therefore offers an intermediate quality–cost trade-off: its **estimated cost per correct ticket is lower than Always-Large but higher than Always-Small**, while accuracy remains below Always-Large. An additional 40-ticket targeted stress set and a post-test v2 iteration
expose the central limitation: confidently wrong urgency predictions can pass deterministic validation. v2 improves
high-urgency recall on that stress set from **86.7% to 93.3%**, but reduces cost savings from **48.9% to 36.1%**.

## Results

Task success = intent AND urgency AND every field correct. Costs use Groq's published prices
($0.075 / $0.30 per 1M input / output tokens for the 20B model, $0.15 / $0.60 for the 120B model).

**Primary result: frozen held-out test (80 tickets; SmartRoute v1 and baselines)**

| System | Task success | Cost saving vs large | Cost per successful ticket |
|---|---:|---:|---:|
| always_small | 78.75% | 53.35% | $0.000117 |
| **smartroute (v1, frozen)** | **81.25%** | **38.70%** | **$0.000149** |
| always_large | 90.00% | 0% | $0.000219 |

**Supplementary pooled analysis: 80-ticket test + 40-ticket targeted validation = 120 tickets.**
The pooled population is **not a representative independent test sample**: validation was deliberately constructed
after inspecting v1's test failures. None of the v1/baseline settings were tuned on the validation rows, but the
validation *design* was informed by observed weaknesses. Use the official 80-ticket test for the primary claim.

| System | Task success | Cost saving vs large | Cost per successful ticket |
|---|---|---|---|
| always_small | 77.5% | 54% | $0.000118 |
| **smartroute (v1, frozen)** | **80.0%** | **42%** | **$0.000142** |
| always_large | 89.2% | 0% | $0.000220 |

**Per-split detail**

| Split | System | Task success | Urgency acc. | High-urgency recall | Cost saving | Escalation |
|---|---|---|---|---|---|---|
| test (80) | always_small | 78.75% | 85.0% | 86.1% | 53% | 0% |
| test (80) | always_large | 90.0% | 92.5% | 100% | 0% | n/a |
| test (80) | smartroute v1 | 81.25% | 86.25% | 91.7% | 39% | 7.5% |
| val (40) | always_small | 75.0% | 85.0% | 86.7% | 54% | 0% |
| val (40) | always_large | 87.5% | 92.5% | 93.3% | 0% | n/a |
| val (40) | smartroute v1 | 77.5% | 85.0% | 86.7% | 49% | 2.5% |
| val (40) | smartroute v2 | 80.0% | 87.5% | 93.3% | 36% | 15% |

v2 on the original test set (**exploratory only**: v2 was designed after inspecting v1's test failures, so this is not independent confirmation):
82.5% task success, 97.2% high-urgency recall, 24% cost saving, 22.5% escalation.
The dev set (40 tickets, tuned on three times, so optimistic): small 95%, large 97.5%, v1 97.5%.

**Robustness by ticket type (test set; tags overlap; small n, read as indications only)**

| Tag | n | always_small | smartroute v1 | always_large |
|---|---|---|---|---|
| easy | 20 | 80% | 75% | 90% |
| noisy (typos, slang) | 16 | 81% | 81% | 100% |
| long_thread (chit-chat, repeats) | 12 | 83% | 92% | 100% |
| ambiguous (two issues) | 16 | 69% | 75% | 75% |
| high_risk | 15 | 87% | 93% | 93% |
| arabic_mixed | 5 | 100% | 100% | 80% |

## Honest findings
1. **Routing buys back only part of the quality gap.** On the test split, Always-Small recalled 86.1% of high-urgency cases versus 91.7% for v1 and 100% for Always-Large. Misses include everyday descriptions of money being taken without delivery.
2. **Self-reported confidence is not informative here.** In the failures we inspected, wrong small-model answers came with confidence >= 0.95, so confidence
   thresholds rarely trigger escalation. Deterministic checks catch broken output, not wrong judgment:
   `validated` only means the output passed the checks. About 80% of validated answers were correct.
3. **v2 trades cost for urgent-ticket recall.** The urgency audit sends money tickets that the small model rates low/normal to the
   large model. It gained about one ticket of accuracy per set and 6 points of recall, and cost 13 to 15 points of savings.
   Use it only if missing an urgent ticket is expensive.
4. **Some errors may reflect label ambiguity as well as model error.** The large model scored 87.5–90% on the two splits. Disagreements cluster on
   low vs normal urgency for billing questions, billing vs general inquiry, and duplicate charges (refund request vs unauthorized
   transaction, which our prompt does not define). These cases were not independently adjudicated, so ambiguity cannot be quantified as the cause of the accuracy ceiling. We report them without retuning on the test.

## How it works
```
ticket -> pre-route (using raw ticket) -> prune (SmartRoute only) -> start model -> parse + validate -> (small model AND issue) -> escalate once to large -> status
```
- **Pruning** (no LLM call): strips quoted and boilerplate lines, drops duplicate messages, never prunes short threads, and in
  long threads always keeps the opening message, the last message and any message with evidence (ids, risk or intent words).
- **Pre-router**: a transparent heuristic over the ticket text (length, long thread, mixed Arabic/English, several issue types,
  risk keywords). Tickets with enough risk features start directly on the large model.
- **Validation** (deterministic): valid JSON, required keys, valid labels, confidence threshold (0.70 normal, 0.85 for
  high-risk), every extracted value must literally appear in the ticket (hallucination check), and urgency must not
  conflict with risk keywords. A large-model start never escalates again; escalation happens at most once.
- **Statuses**: `validated` (checks passed, not a correctness guarantee), `unverified` (no check run, or only low confidence
  remains), `needs_review` (a hard check failed after the final model), `api_error` (infrastructure failure, reported
  separately and never counted as a model answer).
- **v2 urgency audit** (`smartroute_v2`): if the small model returns a money intent (payment failed, refund, unauthorized
  transaction) with an amount but urgency low/normal, the large model settles the urgency. No keyword list is involved.

## Evaluation protocol (what protects the numbers)
- All systems run on identical ticket IDs, with the same **system instructions**, output schema, temperature (0) and reasoning effort (low). Always-Small and Always-Large receive raw context; SmartRoute v1/v2 receive pruned context by design. Thus the comparison measures the combined strategy, not routing alone.
- Dev tickets (40) were used for tuning. The 80-ticket test set was generated after the configuration was frozen
  (a copy of `src/` and `config/` is in `snapshots/v1_frozen_before_test/`; later changes such as the v2 urgency audit and the validation tooling are not in it) and run once. A separate 40-ticket validation set was built later to judge v2.
- Labels are read only by `evaluation/analyze.py`. `src/` never reads labels, and a test greps the runtime code for label words.
- `evaluation/verify_results.py` confirms that every system covered the same ticket ids exactly once, with 0 API errors,
  0 format failures and 0 retried calls on the final runs, and that every cost equals tokens x price.
- Iteration history is kept in `results/groq/dev_run1`, `dev_run2` and `test_v1_frozen`.

## Limitations
- **All tickets are synthetic**, written by an LLM to a stated rubric, not real customer data. Labels follow the same rubric the
  prompt uses, and only a part of them was hand-checked.
- **Small samples**: one ticket is 1.25 percentage points on the test set and 2.5 on val. Differences of one or two tickets are unstable and should not be treated as definitive evidence of improvement.
  The Arabic/English cut has 5 tickets and proves nothing.
- The validation set deliberately contains the failure type v2 targets, so it is a stress test, not a representative sample.
- `validated` does not mean correct: the checker verifies presence in the text, not that the right amount or id was chosen.
- The median recorded inference latency is around 0.5 s per ticket on Groq. Small differences are not established as meaningful; client-side rate-limit waiting is excluded. Cache replays preserve original measured API latency, not live rerun latency.
- Both models use the same reasoning effort on purpose. Tuning effort per model was not explored.
- Cost assumes full input price on every call (no cached-input discount) and ignores any rate-limit waiting. **Published API prices are only a proxy for computational expense**: this project did not measure GPU time, hardware utilization, electricity use, or energy consumption.
- Pruning keeps only the head and tail of a single message longer than 1,000 characters (no ticket here is that long).

## Next Steps
The following improvements were identified through evaluation but were not implemented within the hackathon timeframe:

1. **Refine intent definitions:** Define how duplicate-charge tickets should be distinguished from refund requests and unauthorized transactions, then evaluate the updated rubric on a fresh split.
2. **Improve ground-truth quality:** Obtain independent human adjudication for disputed labels to separate taxonomy ambiguity from model error.
3. **Expand evaluation:** Test on more representative, appropriately anonymized real support tickets and larger samples, including mixed Arabic-English and high-risk cases. Larger samples can improve precision, but do not automatically make small differences statistically significant.
4. **Improve uncertainty estimation:** Explore better-calibrated routing signals instead of relying primarily on model-reported confidence.
5. **Optimize reasoning effort:** Evaluate model-specific reasoning settings rather than one shared setting, with a new controlled benchmark.
6. **Measure compute directly:** Where infrastructure access permits, track accelerator time, utilization and energy alongside API price estimates.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then put your own GROQ_API_KEY in .env (never commit it)
python -m pytest -q         # 34 offline tests with a mock backend
python smoke_test.py        # real Groq call check: both models, tokens, rate-limit headers
```

## Reproduce
```powershell
python -m evaluation.check_dataset --split all
python -m evaluation.run_eval --split test --backend groq --systems always_small always_large smartroute
python -m evaluation.run_eval --split val  --backend groq --systems always_small always_large smartroute smartroute_v2
python -m evaluation.verify_results --backend groq --splits test val dev
```
Outputs go to `results/<backend>/<split>/` (`metrics.csv`, `per_ticket.csv`, `by_tag.csv`, `report.html`). The frozen v1 test snapshot is in `results/groq/test_v1_frozen/`; rerunning the test with the current code is **not** a replacement for that frozen record. The evaluator regenerates analysis files and writes per-system prediction files, so back up any results before a new live run.
Calls are cached in `cache/groq_calls.jsonl` with their original latency, token usage and time, and reruns replay them
(a Groq key is still required to start the client). `--no-cache` forces new calls.

## Layout
```
config/settings.py        single source of truth: models, prices, labels, thresholds, keywords
src/                      pipeline, prompts, pruning, router, verification, models (Groq client, cache, rate limiter)
evaluation/               run_eval, analyze (only reader of labels), check_dataset, verify_results, show_failures
data/                     dev / test / val tickets and labels (synthetic), DATASET_GUIDE.md
results/groq/             saved outputs and metrics for every run
snapshots/                code and config exactly as frozen before the test run (v1)
tests/                    34 offline tests
smoke_test.py             real Groq smoke test
```

## Cost and reproducibility notes
Rates were checked against Groq's model documentation:
[GPT-OSS 20B](https://console.groq.com/docs/model/openai/gpt-oss-20b) and
[GPT-OSS 120B](https://console.groq.com/docs/model/openai/gpt-oss-120b).
Prices are assumptions for estimated inference expense, not observed charges.
Reasoning-token usage is recorded separately when available. Estimated costs use reported completion-token totals without adding reasoning tokens a second time; this is an accounting assumption and does not independently establish the provider's billing policy. In the final-run verification, reasoning tokens never exceeded completion tokens in any recorded call. Client-side wait time
and any unreported usage from failed attempts are not included. The saved final-run verification recorded no
API errors, format failures, retried calls or internal cost mismatches.

## Security
The API key belongs in `.env`, which is git-ignored. Do not include it in public archives, issues, slides or logs.
