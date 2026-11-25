"""Write results: verdicts as JSONL, summaries as JSON + Markdown."""

import json
from pathlib import Path

from .config import get_settings


def results_dir(tag: str) -> Path:
    d = get_settings().results_dir / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda o: o.model_dump()
                               if hasattr(o, "model_dump") else str(o)), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.model_dump() if hasattr(r, "model_dump") else r, ensure_ascii=False) + "\n")


def f(x, nd=2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def suite_markdown(title: str, judge_name: str, rep: dict, usage: dict, cases, verdicts) -> str:
    lines = [f"# {title}", "", f"Judge: `{judge_name}` · {rep['n_scored']}/{rep['n_cases']} cases scored · "
             f"parse status {rep['parse_status']}", "",
             "| Metric | Value |", "|---|---|",
             f"| Pass rate | {f(rep['pass_rate'])} |",
             f"| Mean overall (1-5, weighted, computed) | {f(rep['mean_overall'])} |",
             f"| Std dev of overall | {f(rep['stdev_overall'])} |",
             f"| Overall histogram (rounded 1..5) | {rep['overall_histogram']} |",
             f"| Score entropy (bits, max 2.32) | {f(rep['score_entropy_bits'])} |",
             f"| Spearman(length, overall) | {f(rep['length_vs_overall_spearman'])} |", "",
             "| Criterion | Mean score |", "|---|---|"]
    lines += [f"| {k} | {f(v)} |" for k, v in rep["mean_by_criterion"].items()]
    lines += ["", "| Category | n | Pass rate | Mean overall |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {f(v['pass_rate'])} | {f(v['mean_overall'])} |" for k, v in rep["by_category"].items()]
    lines += ["", f"Judge usage: {usage.get('calls', 0)} API calls ({usage.get('cached_calls', 0)} served from the "
              f"replay cache), {usage.get('prompt_tokens', 0):,} prompt + {usage.get('completion_tokens', 0):,} completion tokens.",
              "", "## Per case", "", "| id | gold | overall | pass | scores (corr/faith/comp/instr/tone) | parse |",
              "|---|---|---|---|---|---|"]
    order = ["correctness", "faithfulness", "completeness", "instruction_following", "tone_safety"]
    for c, v in zip(cases, verdicts):
        sc = "/".join(f(v.criteria[k].score) if k in v.criteria else "-" for k in order)
        lines.append(f"| {c.id} | {c.gold_label or '-'} | {f(v.overall)} | {'yes' if v.passed else 'no'} | {sc} | {v.parse_status} |")
    return "\n".join(lines) + "\n"
