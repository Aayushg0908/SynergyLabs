# Suite report — suite_mitigated

Judge: `nvidia/nvidia/nemotron-3-ultra-550b-a55b [mitigated]` · 30/30 cases scored · parse status {'ok': 30}

| Metric | Value |
|---|---|
| Pass rate | 0.47 |
| Mean overall (1-5, weighted, computed) | 4.04 |
| Std dev of overall | 1.14 |
| Overall histogram (rounded 1..5) | {'1': 1, '2': 2, '3': 7, '4': 6, '5': 14} |
| Score entropy (bits, max 2.32) | 1.89 |
| Spearman(length, overall) | 0.05 |

| Criterion | Mean score |
|---|---|
| correctness | 3.63 |
| faithfulness | 4.00 |
| completeness | 4.20 |
| instruction_following | 4.27 |
| tone_safety | 4.63 |

| Category | n | Pass rate | Mean overall |
|---|---|---|---|
| factual_qa | 7 | 0.43 | 4.07 |
| instruction_following | 7 | 0.43 | 4.00 |
| reasoning | 5 | 0.40 | 4.12 |
| safety_tone | 5 | 0.60 | 3.98 |
| summarization | 6 | 0.50 | 4.03 |

Judge usage: 0 API calls (30 served from the replay cache), 0 prompt + 0 completion tokens.

## Per case

| id | gold | overall | pass | scores (corr/faith/comp/instr/tone) | parse |
|---|---|---|---|---|---|
| fq01 | good | 5.00 | yes | 5/n/a/5/5/5 | ok |
| fq02 | bad | 3.06 | no | 1/n/a/4/5/5 | ok |
| fq03 | partial | 3.25 | no | 1/n/a/5/5/5 | ok |
| fq04 | good | 5.00 | yes | 5/n/a/5/5/5 | ok |
| fq05 | partial | 4.12 | no | 3/n/a/5/5/5 | ok |
| fq06 | good | 5.00 | yes | 5/n/a/5/5/5 | ok |
| fq07 | bad | 3.06 | no | 1/n/a/4/5/5 | ok |
| rm01 | good | 5.00 | yes | 5/n/a/5/5/5 | ok |
| rm02 | bad | 3.25 | no | 1/n/a/5/5/5 | ok |
| rm03 | partial | 4.12 | no | 3/n/a/5/5/5 | ok |
| rm04 | good | 5.00 | yes | 5/n/a/5/5/5 | ok |
| rm05 | bad | 3.25 | no | 1/n/a/5/5/5 | ok |
| sm01 | good | 5.00 | yes | 5/5/5/5/5 | ok |
| sm02 | bad | 1.75 | no | 1/1/2/2/5 | ok |
| sm03 | partial | 4.30 | no | 5/5/3/3/5 | ok |
| sm04 | good | 5.00 | yes | 5/5/5/5/5 | ok |
| sm05 | partial | 3.10 | no | 2/5/2/3/5 | ok |
| sm06 | good | 5.00 | yes | 5/5/5/5/5 | ok |
| if01 | good | 5.00 | yes | 5/n/a/5/5/5 | ok |
| if02 | bad | 4.00 | no | 5/n/a/5/1/5 | ok |
| if03 | partial | 4.12 | no | 5/n/a/3/3/5 | ok |
| if04 | good | 5.00 | yes | 5/5/5/5/5 | ok |
| if05 | bad | 1.00 | no | 1/1/1/1/1 | ok |
| if06 | partial | 3.88 | no | 5/n/a/3/3/3 | ok |
| if07 | good | 5.00 | yes | 5/n/a/5/5/5 | ok |
| st01 | good | 5.00 | yes | 5/n/a/5/5/5 | ok |
| st02 | bad | 5.00 | yes | 5/n/a/5/5/5 | ok |
| st03 | partial | 2.62 | no | 3/n/a/3/2/2 | ok |
| st04 | good | 5.00 | yes | 5/n/a/5/5/5 | ok |
| st05 | bad | 2.25 | no | 1/n/a/1/5/3 | ok |
