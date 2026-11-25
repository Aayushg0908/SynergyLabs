"""Bias experiments: every bias is measured with the naive prompt (before) and the mitigated
prompt (after) on the same inputs, so each mitigation has a before -> after number.

  position         flip rate when the same pair is judged in both orders; first-slot win share
  verbosity        padded-correct vs concise-correct (pointwise delta, pairwise preference);
                   Spearman(length, score) on the A/B outputs
  sycophancy       pass rate on confidently-wrong and user-pressure probes (should be 0)
  score clustering spread of suite scores (std dev, entropy, histogram) and gold agreement
  self-preference  pairs of outputs from two model families judged by a neutral judge and by a
                   judge from each family; self-preference = own-family win rate minus neutral's
"""

import sys
from collections import Counter
from pathlib import Path

from . import stats
from .aggregate import judge_label
from .config import get_settings
from .judge import Judge, run_parallel
from .llm import Usage, usage_dict
from .report import f, results_dir, write_json
from .suite import load_probes, load_suite

VARIANTS = ("naive", "mitigated")


def _workers() -> int:
    return get_settings().max_concurrency


# ------------------------------------------------------------------ position
def position_bias(judges: dict[str, Judge], pairs: list[tuple]) -> dict:
    """pairs: (case, out_a, out_b). Returns flip rate and first-slot preference per variant."""
    out = {}
    for variant, judge in judges.items():
        res = run_parallel(lambda p: judge.pairwise(p[0], p[1], p[2], tag=f"bias:position:{variant}"),
                           pairs, _workers(), f"position {variant}")
        valid = [r for r in res if r["flip"] is not None]
        singles = [v for r in valid for v in (r["ab"], r["ba"])]
        decided = [v for v in singles if v.winner != "tie"]
        first_wins = sum(1 for v in decided if v.winner == v.order[0])
        out[variant] = {
            "n_pairs": len(valid),
            "flip_rate": stats.mean([float(r["flip"]) for r in valid]),
            "first_position_win_share": first_wins / len(decided) if decided else None,
            "single_order_ties": sum(1 for v in singles if v.winner == "tie"),
            "final_after_agreement": dict(Counter(r["final"] for r in valid)),
            "flipped_cases": [r["case_id"] for r in valid if r["flip"]],
        }
    return out


# ------------------------------------------------------------------ probe-based biases
def probe_bias(judges: dict[str, Judge], suite) -> dict:
    probes, probe_pairs = load_probes(suite=suite)
    base_ids = sorted({p["base"] for p in probes if p["base"]})
    by_id = {c.id: c for c in suite}
    out = {}
    for variant, judge in judges.items():
        base_v = {b: v for b, v in zip(base_ids, run_parallel(
            lambda b: judge.pointwise(by_id[b], tag=f"bias:base:{variant}"), base_ids, _workers(), f"base {variant}"))}
        probe_v = run_parallel(lambda p: judge.pointwise(p["case"], tag=f"bias:probe:{variant}"), probes, _workers(),
                               f"probes {variant}")
        rows = []
        for p, v in zip(probes, probe_v):
            base = base_v.get(p["base"])
            fooled = {"fail": v.passed, "pass": not v.passed,
                      "no_higher_than_base": base is not None and v.overall is not None and base.overall is not None
                      and v.overall > base.overall + 1e-9}[p["expect"]]
            rows.append({"id": p["id"], "family": p["family"], "expect": p["expect"], "overall": v.overall,
                         "passed": v.passed, "correctness": v.criteria.get("correctness").score if "correctness" in v.criteria else None,
                         "base_overall": base.overall if base else None, "fooled": fooled, "parse": v.parse_status})
        pair_res = run_parallel(lambda pp: judge.pairwise(pp["case"], pp["better_output"], pp["worse_output"],
                                                          tag=f"bias:probe_pairs:{variant}"),
                                probe_pairs, _workers(), f"probe pairs {variant}")
        pair_rows = []
        for pp, r in zip(probe_pairs, pair_res):
            single = [r["ab"].winner, r["ba"].winner]
            ok = {"better": r["final"] == "A", "better_or_tie": r["final"] in ("A", "tie")}[pp["expect"]]
            pair_rows.append({"id": pp["id"], "expect": pp["expect"], "final": r["final"], "single_orders": single,
                              "worse_won_any_order": "B" in single, "fooled": not ok})
        fam = {}
        for family in sorted({r["family"] for r in rows}):
            rs = [r for r in rows if r["family"] == family]
            fam[family] = {"n": len(rs), "fooled": sum(r["fooled"] for r in rs),
                           "fooled_rate": stats.mean([float(r["fooled"]) for r in rs]),
                           "mean_overall": stats.mean([r["overall"] for r in rs])}
        padded = [r for r in rows if r["family"] == "padded_correct"]
        out[variant] = {
            "by_family": fam, "probes": rows, "pairs": pair_rows,
            "padded_minus_base_mean": stats.mean([r["overall"] - r["base_overall"] for r in padded
                                                  if r["overall"] is not None and r["base_overall"] is not None]),
            "pairwise_fooled": sum(r["fooled"] for r in pair_rows),
            "pairwise_worse_won_in_some_order": sum(r["worse_won_any_order"] for r in pair_rows),
        }
    return out


# ------------------------------------------------------------------ score clustering / spread
def score_spread(judges: dict[str, Judge], suite) -> dict:
    out = {}
    order = ["bad", "partial", "good"]
    for variant, judge in judges.items():
        vs = run_parallel(lambda c: judge.pointwise(c, tag="suite" if variant == "mitigated" else f"suite_{variant}"),
                          suite, _workers(), f"suite {variant}")
        overalls = [v.overall for v in vs]
        hist = Counter(round(x) for x in overalls if x is not None)
        crit_scores = [s.score for v in vs for s in v.criteria.values() if s.score is not None]
        gold = [c.gold_label for c in suite]
        out[variant] = {
            "stdev_overall": stats.stdev(overalls),
            "overall_histogram": {str(k): hist.get(k, 0) for k in range(1, 6)},
            "criterion_score_histogram": {str(k): crit_scores.count(k) for k in range(1, 6)},
            "share_of_criterion_scores_equal_5": crit_scores.count(5) / len(crit_scores) if crit_scores else None,
            "entropy_bits": stats.entropy(hist),
            "gold_pass_kappa": stats.cohen_kappa([v.passed for v in vs], [g == "good" for g in gold]),
            "gold_label_weighted_kappa": stats.cohen_kappa([judge_label(v) for v in vs], gold, "linear", order),
            "spearman_overall_vs_gold": stats.spearman(overalls, [order.index(g) for g in gold]),
        }
    return out


# ------------------------------------------------------------------ self-preference
def self_preference(judges_by_family: dict[str, Judge], neutral: str, pairs: list[tuple], fam_a: str, fam_b: str) -> dict:
    """pairs: (case, output_from_fam_a, output_from_fam_b). Win rates are for fam_a's output."""
    out = {}
    for family, judge in judges_by_family.items():
        res = run_parallel(lambda p: judge.pairwise(p[0], p[1], p[2], tag=f"bias:selfpref:{family}"),
                           pairs, 2 if judge.client.provider == "groq" else _workers(), f"self-pref judge={family}")
        valid = [r for r in res if r["flip"] is not None]
        c = Counter(r["final"] for r in valid)
        n = len(valid)
        out[family] = {"judge": judge.name, "n": n, "a_wins": c.get("A", 0), "b_wins": c.get("B", 0), "ties": c.get("tie", 0),
                       "win_rate_a": (c.get("A", 0) + 0.5 * c.get("tie", 0)) / n if n else None,
                       "flip_rate": stats.mean([float(r["flip"]) for r in valid])}
    base = out[neutral]["win_rate_a"]
    for family, r in out.items():
        if family == neutral or base is None or r["win_rate_a"] is None:
            continue
        own = r["win_rate_a"] if family == fam_a else 1 - r["win_rate_a"]
        neutral_own = base if family == fam_a else 1 - base
        r["own_family_win_rate"] = own
        r["neutral_judge_win_rate_for_same_outputs"] = neutral_own
        r["self_preference"] = own - neutral_own
    return {"family_a": fam_a, "family_b": fam_b, "neutral": neutral, "by_judge": out}


# ------------------------------------------------------------------ command
def cmd_bias(args) -> None:
    from .experiments import _judge_client
    usage = Usage()
    suite = load_suite()
    primary = {v: Judge(_judge_client("primary", usage), v) for v in VARIANTS}
    results = {"judge": primary["mitigated"].client.name}
    parts = set(args.only or ["position", "probes", "spread", "selfpref"])

    if "spread" in parts:
        print("== score clustering", file=sys.stderr)
        results["score_spread"] = score_spread(primary, suite)
    if "probes" in parts:
        print("== verbosity / sycophancy probes", file=sys.stderr)
        results["probes"] = probe_bias(primary, suite)
    if "position" in parts:
        print("== position", file=sys.stderr)
        ab_dir = get_settings().results_dir / "ab"
        a, b = load_suite(ab_dir / "outputs_prompt_v1.yaml"), load_suite(ab_dir / "outputs_prompt_v2.yaml")
        pairs = [(ca, ca.model_output, cb.model_output) for ca, cb in zip(a, b)]
        results["position"] = position_bias(
            {v: Judge(primary[v].client, v) for v in VARIANTS}, pairs)
        # same pairs reused with the A/B run's tag would not be cached across tags; tags do not enter the cache key
        results["position_length"] = {
            "mean_words_a": stats.mean([len(x[1].split()) for x in pairs]),
            "mean_words_b": stats.mean([len(x[2].split()) for x in pairs])}
    if "selfpref" in parts:
        print("== self-preference", file=sys.stderr)
        ab_dir = get_settings().results_dir / "ab"
        gpt = load_suite(ab_dir / "outputs_prompt_v2.yaml")
        qwen_path = ab_dir / "outputs_qwen_v2.yaml"
        if not qwen_path.exists():
            raise SystemExit(f"missing {qwen_path}: run `judge generate --config configs/gen_qwen_v2.yaml` first")
        qwen = load_suite(qwen_path)
        pairs = [(cg, cg.model_output, cq.model_output) for cg, cq in zip(gpt, qwen)][: args.selfpref_n]
        judges = {"nvidia": Judge(_judge_client("primary", usage), "mitigated"),
                  "openai": Judge(_judge_client("groq:openai/gpt-oss-120b", usage), "mitigated"),
                  "alibaba": Judge(_judge_client("secondary", usage), "mitigated")}
        results["self_preference"] = self_preference(judges, "nvidia", pairs, "openai", "alibaba")

    results["usage"] = usage_dict(usage)
    out = results_dir("bias")
    path = out / "bias_results.json"
    if path.exists() and args.only:  # merge partial re-runs
        import json
        prev = json.loads(path.read_text(encoding="utf-8"))
        prev.update({k: v for k, v in results.items()})
        results = prev
    write_json(path, results)
    (out / "bias_report.md").write_text(bias_markdown(results), encoding="utf-8")
    print((out / "bias_report.md").read_text(encoding="utf-8"))


def bias_markdown(r: dict) -> str:
    L = [f"# Bias measurements — judge `{r.get('judge')}`", "",
         "Each row: naive prompt (before) -> mitigated prompt (after), same inputs.", ""]
    if "position" in r:
        p = r["position"]
        L += ["## Position (A/B outputs, each pair judged in both orders)", "",
              "| Metric | Naive | Mitigated |", "|---|---|---|",
              f"| Flip rate (winner changes when order is swapped) | {f(p['naive']['flip_rate'])} | {f(p['mitigated']['flip_rate'])} |",
              f"| First-position win share (single-order verdicts with a winner) | {f(p['naive']['first_position_win_share'])} | {f(p['mitigated']['first_position_win_share'])} |",
              f"| Final verdicts after requiring agreement | {p['naive']['final_after_agreement']} | {p['mitigated']['final_after_agreement']} |",
              f"| Flipped cases | {', '.join(p['naive']['flipped_cases']) or '-'} | {', '.join(p['mitigated']['flipped_cases']) or '-'} |", ""]
    if "probes" in r:
        pr = r["probes"]
        fams = sorted(set(pr["naive"]["by_family"]) | set(pr["mitigated"]["by_family"]))
        L += ["## Verbosity and sycophancy probes (pointwise)", "",
              "| Probe family | n | Fooled (naive) | Fooled (mitigated) | Mean overall naive -> mitigated |", "|---|---|---|---|---|"]
        for fam in fams:
            a, b = pr["naive"]["by_family"][fam], pr["mitigated"]["by_family"][fam]
            L.append(f"| {fam} | {a['n']} | {a['fooled']}/{a['n']} | {b['fooled']}/{b['n']} | {f(a['mean_overall'])} -> {f(b['mean_overall'])} |")
        L += ["", f"Padded-correct minus concise-correct overall: naive {f(pr['naive']['padded_minus_base_mean'])}, "
              f"mitigated {f(pr['mitigated']['padded_minus_base_mean'])} (positive = length rewarded).", "",
              "Pairwise probe pairs (concise vs padded, terse-correct vs verbose-wrong; both orders):", "",
              "| Metric | Naive | Mitigated |", "|---|---|---|",
              f"| Pairs where the judge picked the worse response (after order agreement) | {pr['naive']['pairwise_fooled']}/{len(pr['naive']['pairs'])} | {pr['mitigated']['pairwise_fooled']}/{len(pr['mitigated']['pairs'])} |",
              f"| Pairs where the worse response won in at least one order | {pr['naive']['pairwise_worse_won_in_some_order']} | {pr['mitigated']['pairwise_worse_won_in_some_order']} |", "",
              "| Probe | Family | Expect | Naive overall / pass | Mitigated overall / pass |", "|---|---|---|---|---|"]
        mit = {x["id"]: x for x in pr["mitigated"]["probes"]}
        for x in pr["naive"]["probes"]:
            m = mit[x["id"]]
            L.append(f"| {x['id']} | {x['family']} | {x['expect']} | {f(x['overall'])} / {'pass' if x['passed'] else 'fail'}"
                     f"{' (fooled)' if x['fooled'] else ''} | {f(m['overall'])} / {'pass' if m['passed'] else 'fail'}{' (fooled)' if m['fooled'] else ''} |")
        L.append("")
    if "score_spread" in r:
        s = r["score_spread"]
        L += ["## Score clustering (30-case suite)", "", "| Metric | Naive | Mitigated |", "|---|---|---|",
              f"| Std dev of overall | {f(s['naive']['stdev_overall'])} | {f(s['mitigated']['stdev_overall'])} |",
              f"| Overall histogram 1..5 | {s['naive']['overall_histogram']} | {s['mitigated']['overall_histogram']} |",
              f"| Criterion-score histogram 1..5 | {s['naive']['criterion_score_histogram']} | {s['mitigated']['criterion_score_histogram']} |",
              f"| Share of criterion scores = 5 | {f(s['naive']['share_of_criterion_scores_equal_5'])} | {f(s['mitigated']['share_of_criterion_scores_equal_5'])} |",
              f"| Entropy of rounded overall (bits) | {f(s['naive']['entropy_bits'])} | {f(s['mitigated']['entropy_bits'])} |",
              f"| Cohen's kappa, pass/fail vs gold | {f(s['naive']['gold_pass_kappa'])} | {f(s['mitigated']['gold_pass_kappa'])} |",
              f"| Weighted kappa, 3-level label vs gold | {f(s['naive']['gold_label_weighted_kappa'])} | {f(s['mitigated']['gold_label_weighted_kappa'])} |",
              f"| Spearman(overall, gold) | {f(s['naive']['spearman_overall_vs_gold'])} | {f(s['mitigated']['spearman_overall_vs_gold'])} |", ""]
    if "self_preference" in r:
        sp = r["self_preference"]
        L += [f"## Self-preference ({sp['family_a']} outputs vs {sp['family_b']} outputs, same prompt)", "",
              "| Judge | Family | Win rate for its own family's outputs | Neutral judge's win rate for those outputs | Self-preference |",
              "|---|---|---|---|---|"]
        for fam, x in sp["by_judge"].items():
            if fam == sp["neutral"]:
                L.append(f"| {x['judge']} | {fam} (neutral) | — | win rate for {sp['family_a']}: {f(x['win_rate_a'])} | — |")
            else:
                L.append(f"| {x['judge']} | {fam} | {f(x.get('own_family_win_rate'))} | {f(x.get('neutral_judge_win_rate_for_same_outputs'))} | {f(x.get('self_preference'))} |")
        L.append("")
    u = r.get("usage", {})
    L.append(f"Judge usage for this run: {u.get('calls', 0)} API calls, {u.get('cached_calls', 0)} from cache, "
             f"{u.get('prompt_tokens', 0):,} prompt + {u.get('completion_tokens', 0):,} completion tokens.")
    return "\n".join(L) + "\n"
