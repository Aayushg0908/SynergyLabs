"""Per-case verdicts -> suite report; two configs -> comparison with a declared winner."""

from collections import Counter

from . import stats
from .rubric import label, load_rubric
from .schema import PointwiseVerdict, TestCase


def judge_label(v: PointwiseVerdict) -> str | None:
    """3-level label from the computed overall; a failed case is never 'good'."""
    lab = label(v.overall)
    return "partial" if lab == "good" and not v.passed else lab


def suite_report(cases: list[TestCase], verdicts: list[PointwiseVerdict], outputs: dict[str, str] | None = None) -> dict:
    crit_names = list(load_rubric()["criteria"])
    ok = [v for v in verdicts if v.parse_status != "failed"]
    by_cat: dict[str, list] = {}
    for c, v in zip(cases, verdicts):
        by_cat.setdefault(c.category, []).append(v)
    lengths = [len((outputs or {}).get(c.id, c.model_output).split()) for c in cases]
    overalls = [v.overall for v in verdicts]
    hist = Counter(round(v.overall) for v in ok if v.overall is not None)
    return {
        "n_cases": len(cases),
        "n_scored": len(ok),
        "parse_status": dict(Counter(v.parse_status for v in verdicts)),
        "pass_rate": stats.mean([float(v.passed) for v in ok]),
        "mean_overall": stats.mean(overalls),
        "stdev_overall": stats.stdev(overalls),
        "mean_by_criterion": {n: stats.mean([v.criteria[n].score for v in ok if n in v.criteria]) for n in crit_names},
        "by_category": {cat: {"n": len(vs), "pass_rate": stats.mean([float(v.passed) for v in vs]),
                              "mean_overall": stats.mean([v.overall for v in vs])} for cat, vs in sorted(by_cat.items())},
        "overall_histogram": {str(k): hist.get(k, 0) for k in range(1, 6)},
        "score_entropy_bits": round(stats.entropy(hist), 3),
        "length_vs_overall_spearman": stats.spearman(lengths, overalls),
        "judge_claimed_vs_computed_overall_pearson": stats.pearson([v.judge_overall_claimed for v in verdicts], overalls),
    }


def compare_configs(name_a: str, name_b: str, cases: list[TestCase], point_a: list[PointwiseVerdict],
                    point_b: list[PointwiseVerdict], pairs: list[dict]) -> dict:
    """Combine pairwise (both orders, agreement required) and pointwise evidence into one decision.

    Decision rule: the config with more pairwise wins is the winner if the two-sided sign test on
    non-tied pairs gives p < 0.05 AND the pointwise mean difference points the same way; otherwise
    the leader is reported as 'leaning' and the result as not significant."""
    finals = Counter(p["final"] for p in pairs)
    wins_a, wins_b, ties = finals.get("A", 0), finals.get("B", 0), finals.get("tie", 0)
    n = len(pairs)
    p_value = stats.sign_test_p(wins_a, wins_b)
    flips = [p["flip"] for p in pairs if p["flip"] is not None]
    diffs = [(b.overall - a.overall) if a.overall is not None and b.overall is not None else None
             for a, b in zip(point_a, point_b)]
    mean_diff = stats.mean(diffs)
    ci = stats.bootstrap_ci(diffs)
    report_a, report_b = suite_report(cases, point_a), suite_report(cases, point_b)
    leader = name_b if wins_b > wins_a else name_a if wins_a > wins_b else None
    consistent = leader is None or (mean_diff is not None and ((leader == name_b) == (mean_diff > 0)))
    significant = p_value < 0.05 and consistent and leader is not None
    return {
        "config_a": name_a, "config_b": name_b, "n_pairs": n,
        "pairwise": {"wins_a": wins_a, "wins_b": wins_b, "ties": ties,
                     "win_rate_a": (wins_a + 0.5 * ties) / n if n else None,
                     "win_rate_b": (wins_b + 0.5 * ties) / n if n else None,
                     "sign_test_p": round(p_value, 4),
                     "position_flip_rate": stats.mean([float(f) for f in flips])},
        "pointwise": {"a": {k: report_a[k] for k in ("pass_rate", "mean_overall", "mean_by_criterion")},
                      "b": {k: report_b[k] for k in ("pass_rate", "mean_overall", "mean_by_criterion")},
                      "mean_overall_diff_b_minus_a": mean_diff, "bootstrap_95ci": ci},
        "winner": leader if significant else None,
        "leaning": leader,
        "decision": (f"{leader} wins (sign test p={p_value:.3g}, pointwise diff {mean_diff:+.2f})" if significant else
                     f"no significant winner; leaning {leader} (p={p_value:.3g})" if leader else "no difference"),
    }
