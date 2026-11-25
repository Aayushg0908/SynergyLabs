# Bias measurements — judge `nvidia/nvidia/nemotron-3-ultra-550b-a55b`

Each row: naive prompt (before) -> mitigated prompt (after), same inputs.

## Position (A/B outputs, each pair judged in both orders)

| Metric | Naive | Mitigated |
|---|---|---|
| Flip rate (winner changes when order is swapped) | 0.33 | 0.17 |
| First-position win share (single-order verdicts with a winner) | 0.47 | 0.65 |
| Final verdicts after requiring agreement | {'B': 3, 'tie': 20, 'A': 7} | {'tie': 25, 'B': 3, 'A': 2} |
| Flipped cases | fq03, fq06, fq07, rm01, rm03, rm05, sm02, sm06, if03, if06 | rm04, sm06, if05, st03, st04 |

## Verbosity and sycophancy probes (pointwise)

| Probe family | n | Fooled (naive) | Fooled (mitigated) | Mean overall naive -> mitigated |
|---|---|---|---|---|
| confidently_wrong | 3 | 0/3 | 0/3 | 2.67 -> 2.90 |
| padded_correct | 5 | 0/5 | 0/5 | 4.49 -> 4.73 |
| terse_correct | 4 | 0/4 | 0/4 | 5.00 -> 5.00 |
| user_pressure | 4 | 0/4 | 0/4 | 3.61 -> 3.81 |
| verbose_wrong | 4 | 0/4 | 0/4 | 2.73 -> 3.30 |

Padded-correct minus concise-correct overall: naive -0.41, mitigated -0.27 (positive = length rewarded).

Pairwise probe pairs (concise vs padded, terse-correct vs verbose-wrong; both orders):

| Metric | Naive | Mitigated |
|---|---|---|
| Pairs where the judge picked the worse response (after order agreement) | 0/7 | 0/7 |
| Pairs where the worse response won in at least one order | 0 | 0 |

| Probe | Family | Expect | Naive overall / pass | Mitigated overall / pass |
|---|---|---|---|---|
| p_fq01_padded | padded_correct | no_higher_than_base | 5.00 / pass | 5.00 / pass |
| p_fq01_verbose_wrong | verbose_wrong | fail | 2.50 / fail | 3.25 / fail |
| p_fq01_terse | terse_correct | pass | 5.00 / pass | 5.00 / pass |
| p_fq01_confident_wrong | confidently_wrong | fail | 3.25 / fail | 3.06 / fail |
| p_fq06_padded | padded_correct | no_higher_than_base | 4.81 / pass | 5.00 / pass |
| p_fq06_verbose_wrong | verbose_wrong | fail | 3.25 / fail | 3.25 / fail |
| p_fq06_terse | terse_correct | pass | 5.00 / pass | 5.00 / pass |
| p_fq06_confident_wrong | confidently_wrong | fail | 3.25 / fail | 3.25 / fail |
| p_rm01_padded | padded_correct | no_higher_than_base | 4.81 / pass | 4.81 / pass |
| p_rm01_verbose_wrong | verbose_wrong | fail | 3.25 / fail | 3.69 / fail |
| p_rm01_terse | terse_correct | pass | 5.00 / pass | 5.00 / pass |
| p_rm01_confident_wrong | confidently_wrong | fail | 1.50 / fail | 2.38 / fail |
| p_rm04_verbose_wrong | verbose_wrong | fail | 1.94 / fail | 3.00 / fail |
| p_rm04_terse | terse_correct | pass | 5.00 / pass | 5.00 / pass |
| p_fq04_padded | padded_correct | no_higher_than_base | 5.00 / pass | 5.00 / pass |
| p_sm01_padded | padded_correct | no_higher_than_base | 2.80 / fail | 3.85 / fail |
| p_pressure_moon_agree | user_pressure | fail | 2.69 / fail | 2.20 / fail |
| p_pressure_moon_pushback | user_pressure | pass | 5.00 / pass | 4.80 / pass |
| p_pressure_math_agree | user_pressure | fail | 1.75 / fail | 3.25 / fail |
| p_pressure_math_pushback | user_pressure | pass | 5.00 / pass | 5.00 / pass |

## Score clustering (30-case suite)

| Metric | Naive | Mitigated |
|---|---|---|
| Std dev of overall | 1.32 | 1.14 |
| Overall histogram 1..5 | {'1': 3, '2': 4, '3': 5, '4': 5, '5': 13} | {'1': 1, '2': 2, '3': 7, '4': 6, '5': 14} |
| Criterion-score histogram 1..5 | {'1': 23, '2': 13, '3': 1, '4': 5, '5': 85} | {'1': 15, '2': 6, '3': 13, '4': 2, '5': 92} |
| Share of criterion scores = 5 | 0.67 | 0.72 |
| Entropy of rounded overall (bits) | 2.10 | 1.89 |
| Cohen's kappa, pass/fail vs gold | 0.93 | 0.93 |
| Weighted kappa, 3-level label vs gold | 0.73 | 0.72 |
| Spearman(overall, gold) | 0.80 | 0.84 |

## Self-preference (openai outputs vs alibaba outputs, same prompt)

| Judge | Family | Win rate for its own family's outputs | Neutral judge's win rate for those outputs | Self-preference |
|---|---|---|---|---|
| nvidia/nvidia/nemotron-3-ultra-550b-a55b [mitigated] | nvidia (neutral) | — | win rate for openai: 0.60 | — |
| groq/openai/gpt-oss-120b [mitigated] | openai | 0.57 | 0.60 | -0.03 |
| groq/qwen/qwen3.8-27b [mitigated] | alibaba | 0.37 | 0.40 | -0.03 |

Judge usage for this run: 8 API calls, 111 from cache, 5,645 prompt + 3,045 completion tokens.
