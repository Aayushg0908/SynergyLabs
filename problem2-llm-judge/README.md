# Problem 2 — LLM-as-Judge Evaluation Pipeline

Work in progress.

## Layout

| Path | Purpose |
|---|---|
| `src/llm_judge/` | Pipeline code: suite loading, prompt building, judge clients, verdict parsing, aggregation, bias checks |
| `suites/` | Test suites (JSON/YAML): `{input, system_prompt, model_output, expected_output?, criteria?}` |
| `configs/` | Judge / generator configs, incl. the two configs compared in the A/B run |
| `probes/` | Adversarial probes: padded, verbose-but-wrong, terse-but-correct, confidently-wrong |
| `labels/` | Human/gold labels used to validate the judge (agreement, Cohen's kappa) |
| `results/` | Suite reports, bias measurements, validation and A/B outputs |
| `logs/` | Every judge prompt + raw response (git-ignored, replayable) |
| `scripts/` | One-off utilities (key checks, data prep) |
| `tests/` | Unit tests |

## Setup

```powershell
cd problem2-llm-judge
uv venv --python 3.11 .venv
uv sync
Copy-Item .env.example .env   # then fill in the keys
```
