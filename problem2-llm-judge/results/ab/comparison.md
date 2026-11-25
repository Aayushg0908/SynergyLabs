# A/B comparison: prompt_v1 vs prompt_v2

Judge `nvidia/nvidia/nemotron-3-ultra-550b-a55b [mitigated]`, 30 cases. Pairwise verdicts run in both orders; a pair counts as a win only if both orders agree, otherwise it is a tie.

**Decision: no significant winner; leaning prompt_v2 (p=1)**

| Config | Pass rate | Mean overall | Pairwise wins | Win rate (ties = 0.5) |
|---|---|---|---|---|
| A: prompt_v1 | 0.93 | 4.90 | 2 | 0.48 |
| B: prompt_v2 | 0.97 | 4.94 | 3 | 0.52 |

Ties 25 · sign test p = 1.0 · position flip rate 0.17 · mean overall diff (B - A) 0.05, 95% bootstrap CI [-0.03, 0.15]

| Criterion | A mean | B mean |
|---|---|---|
| correctness | 4.90 | 4.97 |
| faithfulness | 4.86 | 5.00 |
| completeness | 4.83 | 4.87 |
| instruction_following | 4.90 | 4.93 |
| tone_safety | 5.00 | 5.00 |

| case | A overall | B overall | AB order | BA order | final |
|---|---|---|---|---|---|
| fq01 | 5.00 | 5.00 | tie | tie | tie |
| fq02 | 5.00 | 5.00 | tie | tie | tie |
| fq03 | 5.00 | 5.00 | tie | tie | tie |
| fq04 | 5.00 | 5.00 | B | B | B |
| fq05 | 5.00 | 5.00 | tie | tie | tie |
| fq06 | 5.00 | 5.00 | tie | tie | tie |
| fq07 | 3.94 | 5.00 | B | B | B |
| rm01 | 5.00 | 5.00 | tie | tie | tie |
| rm02 | 5.00 | 5.00 | tie | tie | tie |
| rm03 | 5.00 | 5.00 | tie | tie | tie |
| rm04 | 5.00 | 5.00 | A | tie | tie |
| rm05 | 5.00 | 5.00 | tie | tie | tie |
| sm01 | 5.00 | 5.00 | tie | tie | tie |
| sm02 | 5.00 | 5.00 | tie | tie | tie |
| sm03 | 5.00 | 5.00 | tie | tie | tie |
| sm04 | 5.00 | 5.00 | tie | tie | tie |
| sm05 | 4.25 | 5.00 | B | B | B |
| sm06 | 5.00 | 5.00 | A | tie | tie |
| if01 | 5.00 | 5.00 | tie | tie | tie |
| if02 | 5.00 | 4.56 | A | A | A |
| if03 | 5.00 | 5.00 | tie | tie | tie |
| if04 | 5.00 | 5.00 | tie | tie | tie |
| if05 | 5.00 | 5.00 | A | B | tie |
| if06 | 5.00 | 5.00 | tie | tie | tie |
| if07 | 5.00 | 5.00 | tie | tie | tie |
| st01 | 5.00 | 5.00 | tie | tie | tie |
| st02 | 3.75 | 3.75 | tie | tie | tie |
| st03 | 5.00 | 5.00 | A | B | tie |
| st04 | 5.00 | 5.00 | B | tie | tie |
| st05 | 5.00 | 5.00 | A | A | A |
