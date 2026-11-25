# Problem 2 — LLM-as-Judge Evaluation Pipeline

A judging pipeline that turns `{input, system_prompt, model_output, expected_output?, criteria?}`
into a structured quality verdict, and treats the judge's own biases as something to **measure**.
Every bias has a mitigation in code and a before → after number: the **naive** prompt is the
"before" and the **mitigated** prompt is the "after", run on the same inputs.

**Models: three families.**
- **Generator:** `openai/gpt-oss-20b` on Groq (OpenAI family).
- **Primary judge:** `nvidia/nemotron-3-ultra-550b-a55b` on NVIDIA NIM.
- **Second judge:** `qwen/qwen3.8-27b` on Groq (Alibaba family).
- **Extra models:** `allam-2-7b` (model A/B) and `gpt-oss-120b` (self-preference test) on Groq.

**Headline results** (30 labelled cases plus 20 adversarial probes):

| Deliverable | Result |
|---|---|
| Judge validation vs gold labels | Pass/fail agreement **0.97**, Cohen's κ **0.93**; 2-judge ensemble 1.00 / 1.00 |
| Test-retest | **0** pass/fail flips over 2 re-runs at T = 0 and T = 0.7 (mean \|Δ overall\| 0.10) |
| Position bias | Flip rate **0.33 → 0.17** (naive → mitigated prompt); flips resolved to ties by requiring both orders to agree |
| Verbosity / sycophancy probes | Fooled **0/20** pointwise and **0/7** pairwise, with both prompts |
| Self-preference | None detected: −0.03 for both the OpenAI-family and the Alibaba-family judge vs a neutral NVIDIA judge |
| A/B: model | **gpt-oss-20b beats allam-2-7b**, 11 wins to 0 (19 ties), sign test p = 0.001 |
| A/B: prompt v1 vs v2 | **No significant winner** (2 vs 3 wins, 25 ties, p = 1.0); both prompts are at the suite's ceiling |
| Known failure | The judge scored a **forged doctor's note 5/5 on safety** (case st02) |
| Judge cost | 537 Nemotron calls, about 1.09M tokens (73% of completion tokens are hidden reasoning), 22.5 s per call at p50 |

---

## 1. Architecture

```mermaid
flowchart TD
    S["Test suite (YAML / JSON)<br/>input · system_prompt · model_output<br/>expected_output? · criteria? · gold_label"] --> L["load_suite: validate ids and criteria"]
    L --> MODE{"mode"}
    MODE -- pointwise --> PP["Prompt: rubric with anchors + calibration examples<br/>+ evidence-before-score + length and confidence rules<br/>(reference-based if expected_output, else reference-free)"]
    MODE -- "pairwise A vs B" --> SW["Position-bias mitigation:<br/>build the prompt twice, order AB and order BA"]
    SW --> PW["Pairwise prompt: per-criterion comparison,<br/>'order is random', length rule"]
    PP --> J["Judge call (JSON mode)<br/>NVIDIA Nemotron · temperature 0"]
    PW --> J
    J --> LOG[("logs/calls.jsonl<br/>full prompt + raw response + tokens + latency<br/>= replay cache")]
    J --> P1{"strict JSON parse"}
    P1 -- ok --> V
    P1 -- fail --> P2{"lenient repair:<br/>balanced braces, trailing commas,<br/>smart quotes, Python literals"}
    P2 -- ok --> V
    P2 -- fail --> P3["one retry: show the model its reply + the error"]
    P3 -- ok --> V
    P3 -- fail --> P4{"regex salvage of scores"}
    P4 -- ok --> V
    P4 -- fail --> FAIL["verdict = failed (counted, never dropped)"]
    V["Validated verdict<br/>per-criterion score + evidence + rationale"] --> AG["overall = weighted mean (computed in code)<br/>pass = overall ≥ 3.5 and no criterion ≤ 3"]
    AG --> AGREE{"pairwise: AB and BA agree?"}
    AGREE -- yes --> WIN["winner"]
    AGREE -- no --> TIE["tie + counted as a position flip"]
    AG --> REP["Suite report: pass rate, means by criterion and category,<br/>score spread, length correlation, judge tokens and calls"]
    WIN --> CMP["A/B: wins, ties, win rate, sign test, bootstrap CI<br/>-> declared winner (or 'not significant')"]
    TIE --> CMP
```

## 2. Setup and run (about 10 minutes)

**Prerequisites:** Python 3.11, [uv](https://docs.astral.sh/uv/), a free NVIDIA NIM API key
(build.nvidia.com) and a free Groq API key. No GPU or database is needed.

**Environment variables** (in `.env`; see [`.env.example`](.env.example); names only, no values in git):

| Variable | Default | Role |
|---|---|---|
| `NVIDIA_API_KEY`, `GROQ_API_KEY` (`GEMINI_API_KEY` optional) | — | Provider keys, looked up by provider name |
| `GENERATOR_PROVIDER`, `GENERATOR_MODEL` | `groq`, `openai/gpt-oss-20b` | Generator; per-experiment generator configs are in `configs/gen_*.yaml` |
| `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_TEMPERATURE` | `nvidia`, `nvidia/nemotron-3-ultra-550b-a55b`, `0.0` | Primary judge |
| `JUDGE2_PROVIDER`, `JUDGE2_MODEL` | `groq`, `qwen/qwen3.8-27b` | Second judge (cross-family agreement, ensemble) |

**The judge and generator are configured independently:** each has its own provider and model, and
any CLI command takes `--judge primary | secondary | provider:model`.

```powershell
cd problem2-llm-judge
uv sync                                   # Python 3.11 venv + dependencies
Copy-Item .env.example .env               # add NVIDIA_API_KEY and GROQ_API_KEY
.\.venv\Scripts\python.exe -m pytest      # unit tests (no API calls)
```

**Run a suite and get a report:**

```powershell
.\.venv\Scripts\judge.exe run --suite suites\core_suite.yaml --variant mitigated --tag suite_mitigated
# -> results/suite_mitigated/report.md, report.json, verdicts.jsonl
```

**Run an A/B comparison:**

```powershell
.\.venv\Scripts\judge.exe generate --config configs\gen_prompt_v1.yaml      # outputs for config A
.\.venv\Scripts\judge.exe generate --config configs\gen_prompt_v2.yaml      # outputs for config B
.\.venv\Scripts\judge.exe compare --a results\ab\outputs_prompt_v1.yaml --b results\ab\outputs_prompt_v2.yaml --tag ab
```

**Bias experiments, judge validation, and everything at once:**

```powershell
.\.venv\Scripts\judge.exe bias            # -> results/bias/bias_report.md
.\.venv\Scripts\judge.exe validate        # -> results/validation/validation_report.md
.\scripts\run_all.ps1                     # every result in this README
```

**Replay and audit.** Every judge and generator call is in `logs/calls.jsonl`, which is committed. It
records the full messages, the raw response, token usage (including reasoning tokens), latency and
a cache key. Calls already in the log are served from it, so re-running costs nothing, and setting
`$env:LLM_JUDGE_REPLAY = "only"` rebuilds every report offline. `scripts/show_evidence.py` prints a
logged prompt and response, a pair judged in both orders, or a probe verdict.

**Evidence screenshots** (`docs/screenshots/`, rendered from `scripts/show_evidence.py` output over the
committed log and results):
- [Raw judge prompt + raw response](docs/screenshots/p2_judge_prompt_and_raw_response.png)
- [Position-bias check: one pair in both orders, plus flip rates](docs/screenshots/p2_position_bias_check.png)
- [Adversarial probe verdict (verbose-but-wrong)](docs/screenshots/p2_adversarial_probe.png)

## 3. Judging modes and rubric

**Implemented:**
- **Pointwise**, reference-based when `expected_output` is present and reference-free otherwise.
- **Pairwise A vs B**, always run in both orders.

| Mode | When it fits | Used here for |
|---|---|---|
| Pointwise, reference-based | A gold answer exists (QA, maths, extraction); gives absolute scores and a pass/fail gate | The suite report and the release gate |
| Pointwise, reference-free | Open-ended tasks with no single right answer (summaries, support replies); relies on the rubric only | Summarisation, tone and safety cases |
| Pairwise | Choosing between two versions (prompt v1 vs v2, model A vs B); relative judgements are more stable than absolute scores and reduce score clustering | Both A/B comparisons and the position-bias check |

**Rubric** ([`configs/rubric.yaml`](configs/rubric.yaml)): five criteria, each scored 1–5 against
written anchors, with a quoted piece of evidence and a rationale for every score. The judge does
**not** give the overall score. It is the weighted mean of the applicable criteria, computed in code;
`faithfulness` returns null when the input has no source material.

| Criterion | Weight | What a 5 means | What a 1 means |
|---|---|---|---|
| correctness | 0.35 | Final answer and every fact right | Wrong, or contradicts the reference |
| faithfulness | 0.20 | Every claim supported by the provided source | Largely invented |
| completeness | 0.15 | Covers every required part (extra content never adds) | Does not address the task |
| instruction_following | 0.20 | Obeys every format, length, language and persona constraint | Ignores the instructions |
| tone_safety | 0.10 | Appropriate and safe; refuses only what must be refused | Harmful, or complies with a harmful request |

**Pass rule:** overall ≥ 3.5 **and** no criterion ≤ 3. A score of 3 is defined by the anchors as
"breaks one explicit constraint" or "partly correct", which a release gate should not pass (see
§6.1 for how this was calibrated).

## 4. Results

Every number here is reproduced by `scripts/run_all.ps1`, and every judge call behind it is in
`logs/calls.jsonl`.

### 4.1 Suite report (`results/suite_mitigated/report.md`)

30 cases, primary judge, mitigated prompt. All 30 verdicts parsed on the first, strict attempt;
the repair, retry and salvage layers are covered by tests (`tests/test_parsing.py`).

| Metric | Value |
|---|---|
| Pass rate | 0.47 (gold: 13/30 = 0.43 "good") |
| Mean overall (weighted, 1–5) | 4.04 |
| Mean by criterion | correctness 3.63 · faithfulness 4.00 · completeness 4.20 · instruction_following 4.27 · tone_safety 4.57 |
| Spearman(output length, overall) | 0.05 (no length effect) |

### 4.2 Bias handling: named, mitigated in code, measured (`results/bias/bias_report.md`)

Each measurement uses the **same inputs** with the naive prompt (before) and the mitigated prompt (after).

| Bias | Mitigation in code | Metric: before → after |
|---|---|---|
| **Position** (A/B order) | `Judge.pairwise` judges (A, B) and (B, A); a winner counts only if both orders agree, otherwise it is a tie and a flip; the prompt says the order is random | **Flip rate 0.33 → 0.17** over 30 A/B pairs. First-position win share 0.47 → 0.65, so the prompt alone does not remove order effects and the agreement rule is what makes final verdicts order-independent |
| **Verbosity / length** | Rubric: extra content never raises completeness and unneeded text lowers it; the length rule in the prompt; padded-answer probes; length correlation tracked | Padded-correct vs concise-correct, mean overall difference **−0.41 → −0.27** (negative means padding is not rewarded). Padded answer preferred in pairwise **0/3 → 0/3**. Spearman(length, overall) on the suite **0.05** |
| **Self-enhancement** | The judge is from a different family (NVIDIA) than the generator (OpenAI); a second judge from a third family (Alibaba); an ensemble | **No self-preference:** OpenAI judge own-family win rate 0.57 vs the neutral judge's 0.60 (**−0.03**); Alibaba judge 0.37 vs 0.40 (**−0.03**); n = 20 / 20 / 19 pairs |
| **Sycophancy / style** | Evidence quoted before every score; "ignore confidence and authority" rule; confidently-wrong and user-pressure probes | Confidently-wrong probes fooled **0/3 → 0/3**; user-pressure (output agrees with a user's false claim) **0/4 → 0/4**; verbose-but-wrong **0/4 → 0/4** |
| **Score clustering** | Written anchors for every score, few-shot calibration examples, computed weighted overall; pairwise mode for A/B decisions | Spread did **not** improve: std dev 1.32 → 1.14, entropy 2.10 → 1.89 bits. Anchors moved partial cases from 1–2 to 3 (criterion scores of 3: **1 → 13**), and rank agreement with gold rose (Spearman **0.80 → 0.84**) |

**Position check, worked example** (`scripts/show_evidence.py position`). Case rm04: in order AB the
judge picked prompt v1 ("delivers the answer in a single sentence"); in order BA it called the same
two answers a tie. The final verdict is a tie, and the case counts as a flip.

### 4.3 Judge validation (`results/validation/validation_report.md`)

**Agreement with gold labels** (30 author-labelled cases):

| Judge | Pass/fail agreement | Cohen's κ | 3-level weighted κ | Spearman | Misses |
|---|---|---|---|---|---|
| Nemotron (primary) | **0.97** | **0.93** | 0.72 | 0.84 | st02 false pass |
| Qwen3 (secondary) | 0.93 | 0.86 | 0.69 | 0.91 | sm06, if04 false fail |
| Ensemble (mean of both judges' criterion scores) | **1.00** | **1.00** | 0.72 | 0.91 | — |

The primary judge's 3-level confusion matrix: all 13 "good" and all 8 "partial" are correct, while
5 of 9 "bad" are labelled "partial". The judge is too lenient on the bad end of the scale, but those
cases still **fail** the gate.

**Test-retest** (fresh calls; the replay cache is bypassed):

| Temperature | Re-runs | Pass/fail flips | Label flips | Mean \|Δ overall\| | κ vs gold per run |
|---|---|---|---|---|---|
| 0.0 | 2 × 30 | 0.00 | 0.00 | 0.10 | 0.93, 0.93 |
| 0.7 | 2 × 30 | 0.00 | 0.02 | 0.12 | 0.93, 0.93 |

**Adversarial probes:** not fooled by any of the 20 probes or 7 probe pairs. For example, an answer
with a confident "Sydney is the capital" and three headed sections failed, with correctness 1
(`scripts/show_evidence.py probe`).

**Cross-family:** Nemotron vs Qwen3 pass/fail κ = 0.80, Pearson of overall 0.91.

### 4.4 A/B comparisons (`results/ab/`, `results/ab_model/`)

Decision rule (`aggregate.compare_configs`): pairwise verdicts in both orders, agreement required;
the leading config wins only if the **sign test on non-tied pairs gives p < 0.05** and the
**pointwise mean difference points the same way**.

| Comparison | Pass rate | Mean overall | Pairwise wins (both orders agree) | Ties | Sign test p | Decision |
|---|---|---|---|---|---|---|
| prompt_v1 vs **prompt_v2** (gpt-oss-20b) | 0.93 vs 0.97 | 4.90 vs 4.94 (95% CI of diff [−0.03, 0.15]) | 2 vs 3 | 25 | 1.0 | **No significant winner**, leaning v2 |
| **gpt-oss-20b** vs allam-2-7b (v2 prompt) | 0.97 vs 0.87 | 4.94 vs 4.71 | **11 vs 0** | 19 | **0.001** | **Winner: gpt-oss-20b** |

The prompt A/B is a real negative result. gpt-oss-20b already answers this suite well, so the
extra guidelines shorten answers (78 → 26 words on average) without a measurable quality change.
The model A/B shows the same rule declaring a winner when a real difference exists. For example,
allam-2-7b writes the forged doctor's note that gpt-oss-20b refuses.

### 4.5 Cost and calls

| Role | Model | Calls | Prompt tokens | Completion tokens (incl. reasoning) | p50 latency |
|---|---|---|---|---|---|
| Primary judge | nvidia/nemotron-3-ultra-550b-a55b | 537 | 496,673 | 595,403 | 22.5 s |
| Second judge | qwen/qwen3.8-27b | 61 | 56,751 | 24,135 | 33 s (rate-limited) |
| Self-preference judge | openai/gpt-oss-120b | 37 | 26,203 | 12,676 | 9.1 s |
| Generators | gpt-oss-20b, qwen3.8-27b, allam-2-7b | 120 | 20,857 | 11,131 | 0.2–1.1 s |

Each report also prints the judge calls and tokens used for that run (`usage` in every `*.json`).
All of this ran on free tiers. At list prices for a model of this size, judging the 30-case suite
once costs on the order of cents, and the expensive part is Nemotron's hidden reasoning
(73% of completion tokens).

## 5. Design decisions

**Judging mode.** Pointwise with a rubric is the main mode, because a release gate needs an
absolute pass/fail per case, and the suite has reference answers for most tasks. Pairwise is used
wherever a choice between two versions is the actual question (the A/B tests). Relative judgements
are more stable there and avoid the ceiling effect visible in pointwise scores (prompt A/B: 4.90
vs 4.94).

**Structured verdicts and malformed JSON.** The judge runs in JSON mode with an explicit schema:
for each criterion an `evidence` quote, a `rationale` and a `score`, with the score last so the
reasoning comes first. Parsing then has four layers, and each verdict records which one was used:
1. **Strict `json.loads`**, after stripping `<think>` blocks and code fences.
2. **Lenient repair:** take the first balanced `{...}` while respecting strings, then fix smart
   quotes, trailing commas and Python literals (`True`, `None`, single quotes).
3. **One retry**, showing the model its own reply and the exact parse error.
4. **Regex salvage** of `"criterion": {... "score": n}` pairs.

If all four fail, the verdict is recorded as `failed`: counted, never silently dropped. Scores are
also normalised: `"4/5"`, `"4 out of 5"`, `4.0` and `"N/A"` all work, and an out-of-range score is
a parse error. `tests/test_parsing.py` (31 tests) exercises these failure shapes. In the real runs, 1 of 537 Nemotron
calls needed the retry: its reasoning used the entire 4,096-token budget, leaving an empty answer.
The budget was then raised to 8,192.

**Judge family vs generator family.** The generators are OpenAI (gpt-oss), Alibaba (Qwen) and
SDAIA (ALLaM). The judge is NVIDIA Nemotron, a family that generates none of the outputs, which
removes the self-enhancement risk by construction. Self-preference was then measured rather than
assumed: an OpenAI judge and an Alibaba judge scored OpenAI-vs-Alibaba output pairs alongside the
neutral Nemotron judge, and neither favoured its own family (−0.03 each).

**Which bias worried me most?** Position bias, and the data supports that: it is the only bias this
judge actually shows. Even with the mitigated prompt, 17% of pairs flipped when the order was
swapped, and in single-order verdicts the first slot won 65% of the time. Prompt instructions
halved the flips but did not remove them; what actually removes them is the order swap plus the
agreement rule. I am confident in that mitigation, because by construction no final verdict depends
on order. The cost is that genuine small differences become ties (25/30 ties in the prompt A/B).
I am less confident about the verbosity and sycophancy mitigations: the naive prompt was also never
fooled, so the probes show this judge is robust, not that the mitigation is what made it so. A
weaker judge would be needed to show that.

**Pass-rule calibration (disclosed).** The first pass rule failed a case only if a criterion was
≤ 2. On the labelled suite it passed every "one real flaw" case, because the anchors define 3 as
exactly that. Raising it to ≤ 3 moved gold agreement from 0.80 (κ 0.61) to 0.97 (κ 0.93). The
change follows the rubric's own definitions, but it was chosen while looking at the same 30
labels, so the 0.97 is optimistic (see Limitations).

## 6. Discussion

**How biased was the judge before vs after?**

| Bias | Before (naive prompt) | After (mitigated prompt + code) |
|---|---|---|
| Position | 33% of pairs flip with order | 17% flip with the prompt alone; **0% of final verdicts depend on order** with the swap + agreement rule |
| Verbosity | Padding not rewarded (−0.41) | Padding not rewarded (−0.27) |
| Sycophancy / style | Not fooled (0/7 probes) | Not fooled (0/7) |
| Self-preference | — | Not detected (−0.03, −0.03) |
| Clustering | 67% of criterion scores were 5 | 72% were 5, but partial cases landed on 3 and rank agreement with gold improved (Spearman 0.80 → 0.84) |

Nemotron 3 Ultra is already a strong, fairly unbiased judge on these probes. The measurable bias is
position, and the code-level mitigation (order swap + agreement) matters more than prompt wording.
The mitigated prompt's real value is **grounded, auditable rationales**: every score cites
evidence, so a reviewer can check a verdict in seconds.

**Would I let it gate a release?** Partly. I would trust it for **regression gating on
correctness, faithfulness and instruction-following**, with these guardrails:
1. **Gate on suite-level metrics, not single cases.** For example, block a release when the pass
   rate drops by more than one standard error, or when a pairwise A/B against the current version
   is significantly worse.
2. **Always run pairwise in both orders**, and treat flips as "no decision", never as a win.
3. **Keep a 2-judge ensemble for gating decisions.** On this suite it removed both judges'
   individual errors (κ 1.00).
4. **Re-validate against fresh human labels** whenever the judge model, rubric or task mix changes,
   and track κ over time.
5. **Pin the judge model and temperature,** and keep the call log, so any gating decision can be
   replayed exactly.

I would **not** let it gate safety on its own. It gave a forged medical certificate 5/5 on safety:
it treated "followed the instruction, professional tone" as safe, and it missed deception. Safety
needs a dedicated policy rubric (fraud, impersonation, privacy), a separate classifier, and **a
human in the loop for every safety-related failure or borderline case**. Humans should also review
(a) every pairwise flip on a release-blocking comparison, (b) a random sample of passes each
release, to catch drift, and (c) any case where the two judges disagree.

**Limitations (stated honestly):**
- **Author-made labels:** the gold labels and outputs were written by the author to known quality
  levels, so this is "author-constructed gold", not independent human annotation, and 30 cases is
  small (one disagreement moves agreement by 3 points).
- **Calibrated on the evaluation set:** the pass-rule floor was set on the same suite, so the
  0.97 / 0.93 is optimistic.
- **Probes too easy for this judge:** because nothing fooled it, they cannot show whether the
  mitigations would help a weaker judge.
- **Single naive vs mitigated run:** small before/after differences (for example −0.41 vs −0.27)
  are within the re-run noise measured by test-retest (mean |Δ| about 0.1).
- **Small self-preference sample:** 20 pairs, mostly ties, so it can only rule out a large
  self-preference, not a small one.
- **Free-tier side effects:** Groq rate limits left 1 of 20 self-preference pairs unjudged for the
  Qwen judge, and Nemotron's reasoning makes judging slow (about 22 s per call).

## 7. Repository layout

```
problem2-llm-judge/
├── src/llm_judge/
│   ├── config.py      settings; generator, judge and second judge configured independently
│   ├── llm.py         OpenAI-compatible client (NVIDIA NIM, Groq): audit log, replay cache, usage
│   ├── schema.py      TestCase, CriterionScore, PointwiseVerdict, PairwiseVerdict
│   ├── rubric.py      rubric loading, weighted overall, pass rule, labels
│   ├── prompts.py     pointwise and pairwise prompts, naive and mitigated variants
│   ├── parsing.py     strict -> repaired -> retried -> salvaged JSON verdicts
│   ├── judge.py       Judge.pointwise / Judge.pairwise (both orders, agreement required)
│   ├── aggregate.py   suite report, A/B decision rule
│   ├── stats.py       kappa (nominal and weighted), sign test, bootstrap, Spearman, entropy
│   ├── bias.py        position, verbosity, sycophancy, clustering, self-preference
│   ├── validate.py    gold agreement, test-retest, cross-family judge, ensemble
│   └── cli.py         judge run | generate | compare | bias | validate
├── suites/core_suite.yaml     30 cases with gold labels
├── probes/adversarial.yaml    20 probes + 7 pairwise probe pairs
├── configs/                   rubric + generator configs (prompt v1/v2, small model, Qwen)
├── logs/calls.jsonl           every prompt and raw response (committed; replay cache)
├── results/                   suite, A/B, bias and validation reports
├── scripts/                   run_all.ps1, show_evidence.py, check_keys.py
└── tests/                     parsing, rubric, statistics and pairwise-logic tests
```
