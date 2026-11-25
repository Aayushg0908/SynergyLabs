"""Judge parsing and faithfulness computation, with a fake LLM client (no API calls)."""

import json

import pytest

from rag_app.evaluation.judge import AnswerJudge, parse_json_loose
from rag_app.llm import LLMResult


def test_parse_json_loose_variants():
    assert parse_json_loose('{"a": 1}') == {"a": 1}
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('Here is my verdict: {"a": {"b": 2}} Thanks!') == {"a": {"b": 2}}
    assert parse_json_loose('<think>score {"x": 0}?</think>\n{"a": 1}') == {"a": 1}
    with pytest.raises(json.JSONDecodeError):
        parse_json_loose("no json here")


class FakeClient:
    provider, model = "fake", "fake-1"

    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, system, user, json_mode=False):
        return LLMResult(self.responses.pop(0), 10, 5, 15, 1.0, "fake", "fake-1")


GOOD = json.dumps({
    "claims": [{"claim": "31 minutes with full payload", "supported": True, "passage": 1},
               {"claim": "it flies in snow", "supported": False, "passage": None}],
    "answer_relevance": 5, "relevance_reason": "direct", "correctness": 4, "correctness_reason": "ok",
})


def test_faithfulness_is_computed_from_claim_labels(tmp_path):
    judge = AnswerJudge(FakeClient([GOOD]), tmp_path / "judge.jsonl")
    v = judge.judge("q1", "q?", "a", "ref", ["passage"])
    assert v["faithfulness"] == 0.5
    assert v["unsupported_claims"] == ["it flies in snow"]
    assert (v["answer_relevance"], v["correctness"]) == (5, 4)
    assert len((tmp_path / "judge.jsonl").read_text().splitlines()) == 1  # prompt + raw response logged


def test_malformed_json_is_retried_once(tmp_path):
    judge = AnswerJudge(FakeClient(["{not json", GOOD]), tmp_path / "judge.jsonl")
    v = judge.judge("q1", "q?", "a", "ref", ["p"])
    assert v["judge_error"] is None and judge.calls == 2


def test_out_of_range_score_reports_error(tmp_path):
    bad = json.dumps({"claims": [], "answer_relevance": 9, "correctness": 3})
    judge = AnswerJudge(FakeClient([bad, bad]), tmp_path / "judge.jsonl")
    v = judge.judge("q1", "q?", "a", "ref", ["p"])
    assert v["judge_error"] and v["faithfulness"] is None
