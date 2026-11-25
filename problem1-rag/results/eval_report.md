# RAG evaluation report

Generated 2026-11-25T07:38:55+00:00 — 27 questions (24 answerable, 3 unanswerable), 12 documents / 28 vectors.

Config: k=5, chunk 256/32 tokens, embeddings BAAI/bge-small-en-v1.5 (384-d), generator groq/openai/gpt-oss-120b, judge groq/qwen/qwen3.8-27b, min_score=0.5.

## 1. Retrieval (answerable questions)

| Metric | Value |
|---|---|
| Hit rate@5 | 1.000 |
| Recall@5 (evidence spans) | 0.979 |
| MRR | 0.938 |
| nDCG@5 | 0.922 |
| Precision@5 | 0.217 |
| Context precision (rank-aware AP) | 0.896 |
| Precision of chunks passed to the LLM (score ≥ min_score) | 0.233 |
| Unanswerable rejected by similarity threshold | 0.000 |

## 2. Answers

| Metric | Value |
|---|---|
| Faithfulness (supported claims / claims, judge) | 1.000 |
| Answers with ≥1 unsupported claim | 0.000 |
| Answer relevance (1–5, judge) | 5.00 |
| Correctness vs gold (1–5, judge) | 5.00 |
| Exact match | 0.000 |
| Gold answer contained in response | 0.875 |
| Token F1 | 0.307 |
| Citation accuracy (cites a relevant chunk) | 1.000 |
| False abstention on answerable | 0.000 |
| Correct abstention on unanswerable | 1.000 |

## 3. Latency

| Stage | p50 ms | p95 ms |
|---|---|---|
| Vector search (SQL, k=5, 28 vectors) | 3.72 | 4.74 |
| Query embedding (CPU) | 31.11 | 39.66 |
| Retrieval total (embed + search) | 34.8 | 43.25 |
| LLM generation | 693.4 | 908.6 |
| End-to-end (LLM-answered queries) | 755.5 | 1187.1 |

Judge: 24/24 answered questions judged, 0 judge errors, 0 calls, 0 tokens.

## Per-question answers

| id | type | gold | answer | F1 | faithful | correct |
|---|---|---|---|---|---|---|
| q01 | single | 31 minutes | The KX‑200 can fly for about **31 minutes** when carrying its maximum 1.5 kg payload[1]. | 0.25 | 1.00 | 5 |
| q02 | single | $375 | The nightly hotel cap in high‑cost cities—including London—is $375 per night, excluding taxes[1]. | 0.15 | 1.00 | 5 |
| q03 | single | 25 | The Team plan permits up to 25 concurrent open streams. [1] | 0.20 | 1.00 | 5 |
| q04 | single | 429 | When a client exceeds its rate limit, the Kestrel API returns HTTP 429 Too Many Requests [2]. | 0.13 | 1.00 | 5 |
| q05 | single | 16 weeks | Non-birth parents, including adoptive and foster parents, receive 16 weeks of fully paid parental leave [1]. | 0.24 | 1.00 | 5 |
| q06 | single | 12 months or 200 charge cycles, whichever comes first | The KB‑22 smart battery is covered by a limited warranty for 12 months or 200 charge cycles, whichever occurs first[2]. | 0.59 | 1.00 | 5 |
| q07 | single | two | Code changes require at least two approving reviews, with one review coming from a code owner of the files changed [1]. | 0.11 | 1.00 | 5 |
| q08 | single | $0.67 per mile | The policy reimburses personal‑car mileage at a rate of **$0.67 per mile** for business trips[1]. | 0.38 | 1.00 | 5 |
| q09 | single | SEV-2 | The incident is classified as **SEV‑2** when telemetry ingestion lag exceeds 10 minutes[1]. | 0.15 | 1.00 | 5 |
| q10 | single | 72 hours | The required soak test runs for 72 hours. [1] | 0.44 | 1.00 | 5 |
| q11 | single | $18.4 million | Kestrel Labs generated $18.4 million in revenue for Q3 2026[1]. | 0.33 | 1.00 | 5 |
| q12 | single | $1,490 per drone per year | Kestrel Care+ costs $1,490 per drone per year[1]. | 0.77 | 1.00 | 5 |
| q13 | single | 13 months | Telemetry and flight logs are retained for 13 months for customers on the Team plan[1]. | 0.25 | 1.00 | 5 |
| q14 | single | 14 characters | Passwords must be at least 14 characters long [1]. | 0.40 | 1.00 | 5 |
| q15 | single | 16 | A Vela gateway can collect telemetry from up to 16 KX-series drones simultaneously[1]. | 0.15 | 1.00 | 5 |
| q16 | multi_doc | No. Vela gateways buffer up to 6 hours of telemetry and replay it automatically when ingestion recovers. | No customer telemetry data will be lost; gateways buffer up to 6 hours and an outage lasting less than that (e.g., 5 hours) allows the buffered data to be replayed automatically once ingestion recovers[2]. | 0.48 | 1.00 | 5 |
| q17 | multi_doc | $18,100 | The KX‑200 base kit is priced at $14,900 and the Vela gateway costs $3,200, giving a combined list price of $18,100[4][1]. | 0.11 | 1.00 | 5 |
| q18 | single | 3 months | A sabbatical must be requested with at least three months’ notice. [1] | 0.00 | 1.00 | 5 |
| q19 | single | within 1 hour of discovery | Employees must report a suspected security incident within 1 hour of discovery [1]. | 0.62 | 1.00 | 5 |
| q20 | single | 12 m/s | The KX‑200 is rated to resist winds up to **12 m/s**[1]. | 0.33 | 1.00 | 5 |
| q21 | single | $95 per day | The daily meal per diem for international travel is $95 per day[1][2]. | 0.43 | 1.00 | 5 |
| q22 | single | the release captain, the QA lead and the head of flight safety | A firmware release must be signed off by three people: the release captain, the QA lead, and the head of flight safety (currently Tomás Okafor)[1]. | 0.60 | 1.00 | 5 |
| q23 | single | 0.5% | The rollout is automatically paused when the crash‑report rate in any stage exceeds 0.5 % of the devices in that stage within 24 hours[1]. | 0.10 | 1.00 | 5 |
| q24 | single | $2,000 | Each engineer receives an annual learning budget of **$2,000** for courses, books, and conferences[1]. | 0.14 | 1.00 | 5 |
| q25 | unanswerable | — | I couldn't find information about this in the available documents. | n/a | n/a | — |
| q26 | unanswerable | — | I couldn't find information about this in the available documents. | n/a | n/a | — |
| q27 | unanswerable | — | I couldn't find information about this in the available documents. | n/a | n/a | — |
