"""Verdict parsing must survive the ways real judges break JSON."""

import pytest

from llm_judge.parsing import (
    VerdictParseError, coerce_score, extract_json, pairwise_choice, pointwise_scores, salvage_scores,
)
from llm_judge.rubric import label, overall, passed
from llm_judge.schema import CriterionScore

C = ["correctness", "completeness"]


@pytest.mark.parametrize("raw,status", [
    ('{"a": 1}', "ok"),
    ('```json\n{"a": 1}\n```', "ok"),
    ('<think>let me think {not json}</think>{"a": 1}', "ok"),
    ('Sure! Here is my verdict:\n{"a": 1}\nHope that helps.', "repaired"),
    ('{"a": 1,}', "repaired"),                               # trailing comma
    ('{“a”: 1}', "repaired"),                       # smart quotes
    ("{'a': 1}", "repaired"),                                 # single quotes
    ('{"a": True, "b": None}', "repaired"),                   # Python literals
    ('{"note": "has a } brace and \\" quote", "a": 1}', "ok"),
])
def test_extract_json_variants(raw, status):
    obj, got = extract_json(raw)
    assert obj["a"] in (1, True) and got == status


@pytest.mark.parametrize("raw", ["", "no json at all", '{"a": [1, 2}', "{{{"])
def test_extract_json_failures(raw):
    with pytest.raises(VerdictParseError):
        extract_json(raw)


@pytest.mark.parametrize("v,expected", [(4, 4), ("4", 4), (4.0, 4), ("4/5", 4), ("4 out of 5", 4),
                                        ("N/A", None), (None, None), ("null", None), (3.6, 4)])
def test_coerce_score(v, expected):
    assert coerce_score(v) == expected


@pytest.mark.parametrize("v", [0, 6, "great", True])
def test_coerce_score_rejects(v):
    with pytest.raises(VerdictParseError):
        coerce_score(v)


def test_pointwise_scores_shapes():
    nested = {"criteria": {"correctness": {"score": 5, "evidence": "x"}, "completeness": {"score": "3/5"}}, "overall": 4}
    flat = {"correctness": 5, "completeness": 3}
    for obj in (nested, flat):
        scores, _ = pointwise_scores(obj, C)
        assert scores["correctness"].score == 5 and scores["completeness"].score == 3
    with pytest.raises(VerdictParseError, match="missing"):
        pointwise_scores({"criteria": {"correctness": {"score": 5}}}, C)


def test_salvage_from_broken_json():
    broken = '{"criteria": {"correctness": {"evidence": "says "51"", "score": 5}, "completeness": {"score": 2'
    s = salvage_scores(broken, C)
    assert s["correctness"].score == 5 and s["completeness"].score == 2


def test_pairwise_choice_normalisation():
    w, per, _ = pairwise_choice({"winner": "Response 2", "criteria": {"correctness": {"better": "1"}, "x": "tie"}},
                                ["correctness"])
    assert w == "2" and per == {"correctness": "1"}
    assert pairwise_choice({"winner": "TIE"})[0] == "tie"


def test_overall_is_weighted_and_skips_null():
    s = {"correctness": CriterionScore(score=5), "faithfulness": CriterionScore(score=None),
         "instruction_following": CriterionScore(score=2)}
    # weights 0.35 and 0.20 -> (0.35*5 + 0.2*2) / 0.55
    assert overall(s) == pytest.approx(round((0.35 * 5 + 0.2 * 2) / 0.55, 3))
    assert not passed(s, overall(s))                       # a criterion at the fail floor (2) fails the case
    good = {k: CriterionScore(score=4) for k in ("correctness", "completeness")}
    assert passed(good, overall(good)) and label(overall(good)) == "good"
    assert label(3.0) == "partial" and label(2.0) == "bad"
