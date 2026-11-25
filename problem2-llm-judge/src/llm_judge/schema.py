"""Data models: test cases in, structured verdicts out."""

from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["good", "partial", "bad"]


class TestCase(BaseModel):
    """One judging task: {input, system_prompt, model_output, expected_output?, criteria?}."""
    id: str
    category: str = "general"
    input: str
    system_prompt: str = ""
    model_output: str = ""
    expected_output: str | None = None
    criteria: list[str] | None = None       # subset of rubric criteria to apply; default all
    gold_label: Label | None = None          # human/author label used to validate the judge
    notes: str | None = None


class CriterionScore(BaseModel):
    score: int | None = Field(default=None, ge=1, le=5)   # None = not applicable
    evidence: str = ""
    rationale: str = ""


class PointwiseVerdict(BaseModel):
    case_id: str
    criteria: dict[str, CriterionScore]
    overall: float | None             # weighted mean of applicable criteria, computed by us
    passed: bool
    judge_overall_claimed: float | None = None   # what the judge itself said, kept for comparison
    parse_status: str = "ok"          # ok | repaired | retried | failed
    error: str | None = None
    call_ids: list[str] = []


class PairwiseVerdict(BaseModel):
    case_id: str
    winner: Literal["A", "B", "tie"]          # in terms of the ORIGINAL labels A and B
    order: Literal["AB", "BA"]                # which output was shown first
    criteria: dict[str, Literal["A", "B", "tie"]] = {}
    rationale: str = ""
    parse_status: str = "ok"
    error: str | None = None
    call_ids: list[str] = []
