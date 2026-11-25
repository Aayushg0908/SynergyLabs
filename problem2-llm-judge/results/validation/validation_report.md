# Judge validation

Primary judge `nvidia/nvidia/nemotron-3-ultra-550b-a55b [mitigated]` (mitigated prompt), 30-case suite with author-assigned gold labels.

## 1. Agreement with gold labels

| Judge | Pass/fail agreement | Cohen's kappa | 3-level agreement | Weighted kappa | Spearman | False pass | False fail |
|---|---|---|---|---|---|---|---|
| primary | 0.97 | 0.93 | 0.80 | 0.72 | 0.84 | st02 | - |
| secondary (groq/qwen/qwen3.8-27b [mitigated]) | 0.93 | 0.86 | 0.60 | 0.69 | 0.91 | - | sm06, if04 |
| ensemble (mean of both) | 1.00 | 1.00 | 0.77 | 0.72 | 0.91 | - | - |

Confusion matrix, primary judge (rows = gold, columns = judge):

| gold \ judge | bad | partial | good |
|---|---|---|---|
| bad | 3 | 5 | 1 |
| partial | 0 | 8 | 0 |
| good | 0 | 0 | 13 |

## 2. Test-retest consistency

| Temperature | Re-runs | Pass/fail flip rate | Label flip rate | Mean abs Δ overall | Max abs Δ | Kappa vs gold per re-run | Flipped cases |
|---|---|---|---|---|---|---|---|
| 0.0 | 2 | 0.00 | 0.00 | 0.10 | 1.31 | 0.93, 0.93 | - |
| 0.7 | 2 | 0.00 | 0.02 | 0.12 | 1.31 | 0.93, 0.93 | - |

## 3. Cross-family agreement

Primary vs secondary: pass/fail kappa 0.80, Pearson of overall 0.91.

## 4. Adversarial probes (was the judge fooled?)

| Probe family | Naive prompt | Mitigated prompt |
|---|---|---|
| confidently_wrong | fooled 0/3 | fooled 0/3 |
| padded_correct | fooled 0/5 | fooled 0/5 |
| terse_correct | fooled 0/4 | fooled 0/4 |
| user_pressure | fooled 0/4 | fooled 0/4 |
| verbose_wrong | fooled 0/4 | fooled 0/4 |
| pairwise probe pairs | fooled 0/7 | fooled 0/7 |

Usage: 145 judge API calls, 30 cached, 189,657 prompt + 176,332 completion tokens.
