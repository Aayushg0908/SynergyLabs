"""LLM-as-judge for answer quality.

The judge (Gemini, a different model family from the Groq/OpenAI-family generator, to
avoid self-preference) does claim-level grading:

* faithfulness = supported_claims / total_claims, computed here from the judge's
  per-claim labels rather than taken from a single number the judge asserts;
* answer relevance and correctness-vs-reference on anchored 1-5 scales.

Every prompt and raw response is appended to logs/judge_calls.jsonl for audit.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ..llm import LLMClient

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of answers produced by a retrieval-augmented question-answering "
    "system. Judge only against the material provided to you. Respond with JSON only."
)

JUDGE_TEMPLATE = """Evaluate the ANSWER to the QUESTION.

## Context passages the system was given
{context}

## Reference answer (human-written; may be shorter than the system answer)
{reference}

## Question
{question}

## Answer to evaluate
{answer}

Do three things.

1. FAITHFULNESS. Split the answer into atomic factual claims (ignore citation markers like [1]).
   For each claim decide whether it is directly supported by the context passages. A claim is
   supported only if a passage states it, or it follows by simple arithmetic from numbers the
   passages state. Background knowledge does NOT count as support.

2. ANSWER RELEVANCE, 1-5: does the answer address the question that was asked?
   5 = directly and completely answers it
   4 = answers it, with a minor gap or some unnecessary material
   3 = partially answers it
   2 = mostly off-target
   1 = does not address the question

3. CORRECTNESS vs the reference answer, 1-5:
   5 = contains the reference's key facts and nothing contradicts it
   4 = key fact present, minor omission or imprecision
   3 = partially correct
   2 = mostly wrong
   1 = wrong, or contradicts the reference

Return JSON in exactly this shape:
{{"claims": [{{"claim": "<text>", "supported": true, "passage": 1}}],
  "answer_relevance": 5, "relevance_reason": "<one sentence>",
  "correctness": 5, "correctness_reason": "<one sentence>"}}"""


def parse_json_loose(text: str) -> dict:
    """Parse a JSON object from a model response, tolerating code fences and surrounding prose."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start:end + 1])


class AnswerJudge:
    def __init__(self, client: LLMClient, log_path: Path):
        self.client = client
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.tokens = 0

    def judge(self, qid: str, question: str, answer: str, reference: str, passages: list[str]) -> dict:
        context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1)) or "(none)"
        prompt = JUDGE_TEMPLATE.format(context=context, reference=reference, question=question, answer=answer)

        verdict, raw, error = None, "", None
        for attempt in range(2):  # one retry if the JSON is malformed
            try:
                result = self.client.complete(JUDGE_SYSTEM, prompt, json_mode=True)
            except Exception as exc:  # API failure after retries: record it, don't abort the whole eval
                error = f"api_error: {type(exc).__name__}: {str(exc)[:300]}"
                self._log(qid, attempt, prompt, error, 0)
                break
            self.calls += 1
            self.tokens += result.total_tokens
            raw = result.text
            self._log(qid, attempt, prompt, raw, result.total_tokens)
            try:
                verdict = parse_json_loose(raw)
                self._validate(verdict)
                error = None
                break
            except (ValueError, KeyError, TypeError) as exc:
                error = f"{type(exc).__name__}: {exc}"

        if verdict is None or error:
            return {"judge_error": error, "faithfulness": None, "answer_relevance": None, "correctness": None}

        claims = verdict["claims"]
        supported = sum(1 for c in claims if c.get("supported") is True)
        return {
            "faithfulness": supported / len(claims) if claims else None,
            "n_claims": len(claims),
            "n_supported": supported,
            "unsupported_claims": [c["claim"] for c in claims if c.get("supported") is not True],
            "answer_relevance": int(verdict["answer_relevance"]),
            "relevance_reason": verdict.get("relevance_reason", ""),
            "correctness": int(verdict["correctness"]),
            "correctness_reason": verdict.get("correctness_reason", ""),
            "judge_error": None,
        }

    @staticmethod
    def _validate(v: dict) -> None:
        if not isinstance(v.get("claims"), list):
            raise ValueError("claims must be a list")
        for key in ("answer_relevance", "correctness"):
            if not 1 <= int(v[key]) <= 5:
                raise ValueError(f"{key} out of range")

    def _log(self, qid: str, attempt: int, prompt: str, raw: str, tokens: int) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "qid": qid, "attempt": attempt,
               "judge": f"{self.client.provider}/{self.client.model}", "system": JUDGE_SYSTEM,
               "prompt": prompt, "raw_response": raw, "total_tokens": tokens}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
