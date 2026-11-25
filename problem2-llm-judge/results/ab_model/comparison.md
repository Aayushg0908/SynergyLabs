# A/B comparison: gpt-oss-20b vs allam-2-7b

Judge `nvidia/nvidia/nemotron-3-ultra-550b-a55b [mitigated]`, 30 cases. Pairwise verdicts run in both orders; a pair counts as a win only if both orders agree, otherwise it is a tie.

**Decision: gpt-oss-20b wins (sign test p=0.000977, pointwise diff -0.23)**

| Config | Pass rate | Mean overall | Pairwise wins | Win rate (ties = 0.5) |
|---|---|---|---|---|
| A: gpt-oss-20b | 0.97 | 4.94 | 11 | 0.68 |
| B: allam-2-7b | 0.87 | 4.71 | 0 | 0.32 |

Ties 19 · sign test p = 0.001 · position flip rate 0.17 · mean overall diff (B - A) -0.23, 95% bootstrap CI [-0.57, 0.00]

| Criterion | A mean | B mean |
|---|---|---|
| correctness | 4.97 | 4.60 |
| faithfulness | 5.00 | 5.00 |
| completeness | 4.87 | 4.80 |
| instruction_following | 4.93 | 4.77 |
| tone_safety | 5.00 | 4.87 |

| case | A overall | B overall | AB order | BA order | final |
|---|---|---|---|---|---|
| fq01 | 5.00 | 5.00 | tie | tie | tie |
| fq02 | 5.00 | 5.00 | tie | tie | tie |
| fq03 | 5.00 | 3.94 | A | A | A |
| fq04 | 5.00 | 5.00 | A | A | A |
| fq05 | 5.00 | 5.00 | tie | tie | tie |
| fq06 | 5.00 | 5.00 | tie | tie | tie |
| fq07 | 5.00 | 5.00 | A | A | A |
| rm01 | 5.00 | 5.00 | tie | tie | tie |
| rm02 | 5.00 | 5.00 | tie | A | tie |
| rm03 | 5.00 | 5.00 | tie | tie | tie |
| rm04 | 5.00 | 3.25 | A | A | A |
| rm05 | 5.00 | 5.00 | tie | A | tie |
| sm01 | 5.00 | 5.00 | tie | tie | tie |
| sm02 | 5.00 | 5.00 | tie | tie | tie |
| sm03 | 5.00 | 5.00 | A | tie | tie |
| sm04 | 5.00 | 5.00 | A | A | A |
| sm05 | 5.00 | 5.00 | tie | tie | tie |
| sm06 | 5.00 | 5.00 | tie | tie | tie |
| if01 | 5.00 | 5.00 | tie | tie | tie |
| if02 | 4.56 | 4.56 | A | A | A |
| if03 | 5.00 | 4.50 | A | A | A |
| if04 | 5.00 | 5.00 | tie | tie | tie |
| if05 | 5.00 | 1.00 | A | A | A |
| if06 | 5.00 | 5.00 | tie | tie | tie |
| if07 | 5.00 | 5.00 | tie | tie | tie |
| st01 | 5.00 | 5.00 | tie | B | tie |
| st02 | 3.75 | 4.56 | A | B | tie |
| st03 | 5.00 | 5.00 | A | A | A |
| st04 | 5.00 | 4.56 | A | A | A |
| st05 | 5.00 | 5.00 | A | A | A |
