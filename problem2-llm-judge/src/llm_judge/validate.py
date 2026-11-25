"""Judge validation.

  1. Agreement with gold labels: pass/fail agreement + Cohen's kappa, 3-level weighted kappa,
     confusion matrix, Spearman(overall, gold).
  2. Test-retest: the suite re-judged with fresh calls (sample index > 0 bypasses the replay
     cache) at temperature 0 and 0.7; verdict flip rate and mean |delta overall| vs run 0.
  3. Cross-family agreement: the secondary judge (Qwen3) scores the same suite; agreement between
     judges and of a 2-judge ensemble (mean of criterion scores) with gold.
  4. Adversarial probes: summarised from results/bias (run `judge bias` first).
"""

import json
import sys
from collections import Counter

from . import stats
from .aggregate import judge_label
from .config import get_settings
from .judge import Judge, run_parallel
from .llm import Usage, usage_dict
from .report import f, results_dir, write_json
from .rubric import overall, passed
from .schema import CriterionScore, PointwiseVerdict
from .suite import load_suite

ORDER = ["bad", "partial", "good"]


def gold_agreement(cases, verdicts) -> dict:
    gold = [c.gold_label for c in cases]
    jl = [judge_label(v) for v in verdicts]
    jp = [v.passed for v in verdicts]
    gp = [g == "good" for g in gold]
    conf = Counter((g, j) for g, j in zip(gold, jl))
    return {
        "n": len(cases),
        "pass_fail_agreement": stats.mean([float(a == b) for a, b in zip(jp, gp)]),
        "pass_fail_kappa": stats.cohen_kappa(jp, gp),
        "label_agreement": stats.mean([float(a == b) for a, b in zip(jl, gold)]),
        "label_weighted_kappa": stats.cohen_kappa(jl, gold, "linear", ORDER),
        "spearman_overall_vs_gold": stats.spearman([v.overall for v in verdicts], [ORDER.index(g) for g in gold]),
        "confusion_gold_rows_judge_cols": {g: {j: conf.get((g, j), 0) for j in ORDER} for g in ORDER},
        "false_pass": [c.id for c, a, b in zip(cases, jp, gp) if a and not b],
        "false_fail": [c.id for c, a, b in zip(cases, jp, gp) if b and not a],
    }


def ensemble(v1: PointwiseVerdict, v2: PointwiseVerdict) -> PointwiseVerdict:
    names = set(v1.criteria) | set(v2.criteria)
    merged = {}
    for n in names:
        s = [x.criteria[n].score for x in (v1, v2) if n in x.criteria and x.criteria[n].score is not None]
        merged[n] = CriterionScore(score=round(sum(s) / len(s)) if s else None)
    ov = overall(merged)
    return PointwiseVerdict(case_id=v1.case_id, criteria=merged, overall=ov, passed=passed(merged, ov))


def cmd_validate(args) -> None:
    from .experiments import _judge_client
    cases = load_suite()
    usage = Usage()
    w = get_settings().max_concurrency
    out = {}

    judge = Judge(_judge_client("primary", usage), "mitigated")
    base = run_parallel(lambda c: judge.pointwise(c, tag="suite"), cases, w, "run 0 (cached)")
    out["judge"] = judge.name
    out["gold_agreement"] = gold_agreement(cases, base)

    retest = {}
    for temp in (0.0, 0.7):
        j = Judge(_judge_client("primary", usage, temperature=temp), "mitigated")
        runs = []
        for sample in range(1, args.repeats + 1):
            runs.append(run_parallel(lambda c, s=sample: j.pointwise(c, sample=s, tag=f"retest:t{temp}"), cases, w,
                                     f"retest T={temp} #{sample}"))
        flips = [float(r.passed != b.passed) for run in runs for r, b in zip(run, base)]
        lab_flips = [float(judge_label(r) != judge_label(b)) for run in runs for r, b in zip(run, base)]
        deltas = [abs(r.overall - b.overall) for run in runs for r, b in zip(run, base)
                  if r.overall is not None and b.overall is not None]
        retest[str(temp)] = {"repeats": args.repeats, "pass_fail_flip_rate": stats.mean(flips),
                             "label_flip_rate": stats.mean(lab_flips), "mean_abs_overall_delta": stats.mean(deltas),
                             "max_abs_overall_delta": max(deltas) if deltas else None,
                             "kappa_vs_gold_per_run": [stats.cohen_kappa([v.passed for v in run], [c.gold_label == "good" for c in cases]) for run in runs],
                             "flipped_cases": sorted({c.id for run in runs for c, r, b in zip(cases, run, base) if r.passed != b.passed})}
    out["test_retest"] = retest

    if not args.skip_secondary:
        j2 = Judge(_judge_client("secondary", usage), "mitigated")
        sec = run_parallel(lambda c: j2.pointwise(c, tag="secondary"), cases, 2, "secondary judge")
        ens = [ensemble(a, b) for a, b in zip(base, sec)]
        out["secondary_judge"] = j2.name
        out["secondary_gold_agreement"] = gold_agreement(cases, sec)
        out["ensemble_gold_agreement"] = gold_agreement(cases, ens)
        out["judge_vs_judge"] = {
            "pass_fail_kappa": stats.cohen_kappa([a.passed for a in base], [b.passed for b in sec]),
            "overall_pearson": stats.pearson([a.overall for a in base], [b.overall for b in sec]),
        }

    bias_path = get_settings().results_dir / "bias" / "bias_results.json"
    if bias_path.exists():
        b = json.loads(bias_path.read_text(encoding="utf-8"))
        if "probes" in b:
            out["adversarial"] = {v: b["probes"][v]["by_family"] for v in ("naive", "mitigated")}
            out["adversarial_pairs"] = {v: {"fooled": b["probes"][v]["pairwise_fooled"], "n": len(b["probes"][v]["pairs"])}
                                        for v in ("naive", "mitigated")}

    out["usage"] = usage_dict(usage)
    d = results_dir("validation")
    write_json(d / "validation.json", out)
    (d / "validation_report.md").write_text(validation_markdown(out), encoding="utf-8")
    print((d / "validation_report.md").read_text(encoding="utf-8"))


def _agree_rows(name: str, g: dict) -> list[str]:
    return [f"| {name} | {f(g['pass_fail_agreement'])} | {f(g['pass_fail_kappa'])} | {f(g['label_agreement'])} | "
            f"{f(g['label_weighted_kappa'])} | {f(g['spearman_overall_vs_gold'])} | {', '.join(g['false_pass']) or '-'} | "
            f"{', '.join(g['false_fail']) or '-'} |"]


def validation_markdown(o: dict) -> str:
    L = ["# Judge validation", "", f"Primary judge `{o['judge']}` (mitigated prompt), 30-case suite with author-assigned gold labels.", "",
         "## 1. Agreement with gold labels", "",
         "| Judge | Pass/fail agreement | Cohen's kappa | 3-level agreement | Weighted kappa | Spearman | False pass | False fail |",
         "|---|---|---|---|---|---|---|---|"]
    L += _agree_rows("primary", o["gold_agreement"])
    if "secondary_gold_agreement" in o:
        L += _agree_rows(f"secondary ({o['secondary_judge']})", o["secondary_gold_agreement"])
        L += _agree_rows("ensemble (mean of both)", o["ensemble_gold_agreement"])
    L += ["", "Confusion matrix, primary judge (rows = gold, columns = judge):", "", "| gold \\ judge | bad | partial | good |", "|---|---|---|---|"]
    for g, row in o["gold_agreement"]["confusion_gold_rows_judge_cols"].items():
        L.append(f"| {g} | {row['bad']} | {row['partial']} | {row['good']} |")
    L += ["", "## 2. Test-retest consistency", "", "| Temperature | Re-runs | Pass/fail flip rate | Label flip rate | Mean abs Δ overall | Max abs Δ | Kappa vs gold per re-run | Flipped cases |",
          "|---|---|---|---|---|---|---|---|"]
    for t, r in o["test_retest"].items():
        L.append(f"| {t} | {r['repeats']} | {f(r['pass_fail_flip_rate'])} | {f(r['label_flip_rate'])} | {f(r['mean_abs_overall_delta'])} | "
                 f"{f(r['max_abs_overall_delta'])} | {', '.join(f(k) for k in r['kappa_vs_gold_per_run'])} | {', '.join(r['flipped_cases']) or '-'} |")
    if "judge_vs_judge" in o:
        jj = o["judge_vs_judge"]
        L += ["", "## 3. Cross-family agreement", "", f"Primary vs secondary: pass/fail kappa {f(jj['pass_fail_kappa'])}, "
              f"Pearson of overall {f(jj['overall_pearson'])}."]
    if "adversarial" in o:
        L += ["", "## 4. Adversarial probes (was the judge fooled?)", "", "| Probe family | Naive prompt | Mitigated prompt |", "|---|---|---|"]
        for fam in o["adversarial"]["mitigated"]:
            a, b = o["adversarial"]["naive"][fam], o["adversarial"]["mitigated"][fam]
            L.append(f"| {fam} | fooled {a['fooled']}/{a['n']} | fooled {b['fooled']}/{b['n']} |")
        ap = o["adversarial_pairs"]
        L.append(f"| pairwise probe pairs | fooled {ap['naive']['fooled']}/{ap['naive']['n']} | fooled {ap['mitigated']['fooled']}/{ap['mitigated']['n']} |")
    u = o["usage"]
    L += ["", f"Usage: {u['calls']} judge API calls, {u['cached_calls']} cached, {u['prompt_tokens']:,} prompt + "
          f"{u['completion_tokens']:,} completion tokens."]
    return "\n".join(L) + "\n"
