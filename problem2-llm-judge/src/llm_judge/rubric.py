"""Load the rubric and compute overall score, pass/fail and label from per-criterion scores."""

from functools import lru_cache
from pathlib import Path

import yaml

from .config import PROJECT_ROOT
from .schema import CriterionScore

RUBRIC_PATH = PROJECT_ROOT / "configs" / "rubric.yaml"


@lru_cache
def load_rubric(path: Path = RUBRIC_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def criteria_for(case_criteria: list[str] | None) -> list[str]:
    all_c = list(load_rubric()["criteria"])
    if not case_criteria:
        return all_c
    unknown = set(case_criteria) - set(all_c)
    if unknown:
        raise ValueError(f"unknown criteria {sorted(unknown)}")
    return [c for c in all_c if c in case_criteria]


def overall(scores: dict[str, CriterionScore]) -> float | None:
    """Weighted mean over applicable (non-null) criteria, re-normalising the weights."""
    crit = load_rubric()["criteria"]
    pairs = [(crit[name]["weight"], s.score) for name, s in scores.items() if s.score is not None and name in crit]
    total_w = sum(w for w, _ in pairs)
    return round(sum(w * s for w, s in pairs) / total_w, 3) if total_w else None


def passed(scores: dict[str, CriterionScore], overall_score: float | None) -> bool:
    rule = load_rubric()["pass_rule"]
    if overall_score is None:
        return False
    floor_ok = all(s.score is None or s.score > rule["fail_floor"] for s in scores.values())
    return overall_score >= rule["min_overall"] and floor_ok


def label(overall_score: float | None) -> str | None:
    if overall_score is None:
        return None
    bands = load_rubric()["label_bands"]
    if overall_score >= bands["good"]:
        return "good"
    return "partial" if overall_score >= bands["partial"] else "bad"


def render(names: list[str], with_anchors: bool = True) -> str:
    crit = load_rubric()["criteria"]
    out = []
    for n in names:
        c = crit[n]
        out.append(f"### {n}\n{c['definition']}")
        if with_anchors:
            out += [f"  {k} = {v}" for k, v in sorted(c["anchors"].items(), reverse=True)]
    return "\n".join(out)
