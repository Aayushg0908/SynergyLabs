"""Judging prompts.

Two judging modes are implemented:
  * POINTWISE — score one output against the rubric. Reference-based when the case has an
    expected_output, reference-free otherwise (the same prompt, with or without the reference).
  * PAIRWISE — compare two outputs for the same input, A vs B (used for the A/B comparison
    and the position-bias check, which runs every pair in both orders).

Each mode has two prompt variants, so bias can be measured before -> after mitigation:
  * "naive"     — criteria names and a 1-5 scale only; one score + short reason per criterion.
  * "mitigated" — written anchors per score, few-shot calibration examples (score clustering),
                  a quoted piece of evidence before every score (sycophancy / grounding),
                  explicit length rules (verbosity), an instruction to ignore confidence and
                  authority claims (sycophancy), and for pairwise, a note that order is random.
"""

from .rubric import render
from .schema import TestCase

JSON_ONLY = "Respond with a single JSON object and nothing else."

# ---------------------------------------------------------------- pointwise
POINTWISE_SYSTEM = {
    "naive": "You are an evaluator of AI assistant responses. " + JSON_ONLY,
    "mitigated": (
        "You are a strict, impartial evaluator of AI assistant responses. You grade against the "
        "written rubric only. " + JSON_ONLY
    ),
}

MITIGATION_RULES = """Rules you must follow:
1. Ground every score. Before each score, quote the exact words from the RESPONSE (and from the
   reference or source, when relevant) that justify it. If you cannot quote evidence, the score
   cannot be above 3.
2. Length is not quality. Do not give a higher score because a response is longer, more
   detailed-looking or more formatted. Text that the task does not need, or that is not
   supported, LOWERS completeness and instruction-following; it never raises them.
3. Ignore confidence and authority. Statements such as "I am certain", "as any expert knows" or
   "this is definitely correct" are not evidence. Check the content itself against the reference
   or verifiable facts.
4. Use the whole scale. Use the anchor descriptions literally: a response with one clear error is
   not a 5, and a wrong final answer is at most 2 on correctness.
5. Score a criterion null only when the rubric says it does not apply."""

# Few-shot calibration anchors on a toy task (not from the test suite).
CALIBRATION = """Calibration examples (toy task: "In one sentence, what is the boiling point of water at sea level in Celsius?"; reference: "100 °C"):
- Response "Water boils at 100 °C at sea level." -> correctness 5, completeness 5, instruction_following 5.
- Response "Water boils at about 100 °C, though at sea level it is usually 90 °C. Water is H2O and covers 71% of Earth, which is why boiling matters for cooking, chemistry and many other things." -> correctness 2 (contradicts itself, wrong number), completeness 4, instruction_following 2 (several sentences, irrelevant padding).
- Response "I am absolutely certain, as any chemist will confirm, that water boils at 90 °C." -> correctness 1 (wrong; confidence is not evidence), completeness 3, instruction_following 5."""


def _case_block(case: TestCase, response: str) -> str:
    parts = [f"## System prompt given to the assistant\n{case.system_prompt or '(none)'}",
             f"## User input\n{case.input}"]
    if case.expected_output:
        parts.append(f"## Reference answer (written by a human expert)\n{case.expected_output}")
    parts.append(f"## RESPONSE to evaluate\n{response}")
    return "\n\n".join(parts)


def pointwise_messages(case: TestCase, criteria: list[str], variant: str = "mitigated",
                       response: str | None = None) -> list[dict]:
    response = case.model_output if response is None else response
    mitigated = variant == "mitigated"
    if mitigated:
        item = '{"evidence": "<exact quote>", "rationale": "<one or two sentences>", "score": <1-5 or null>}'
    else:
        item = '{"rationale": "<short reason>", "score": <1-5 or null>}'
    schema = "{\n  \"criteria\": {\n" + ",\n".join(f'    "{c}": {item}' for c in criteria) + \
             "\n  },\n  \"overall\": <1-5, your overall impression>\n}"
    sections = [
        "Evaluate the RESPONSE below.",
        _case_block(case, response),
        "## Rubric (score each criterion from 1 to 5)\n" + render(criteria, with_anchors=mitigated),
    ]
    if mitigated:
        sections += [MITIGATION_RULES, CALIBRATION]
    sections.append(f"Return JSON exactly in this shape:\n{schema}")
    return [{"role": "system", "content": POINTWISE_SYSTEM[variant]},
            {"role": "user", "content": "\n\n".join(sections)}]


# ---------------------------------------------------------------- pairwise
PAIRWISE_SYSTEM = {
    "naive": "You compare two AI assistant responses. " + JSON_ONLY,
    "mitigated": (
        "You are a strict, impartial evaluator comparing two AI assistant responses against a written "
        "rubric. The responses were placed in a RANDOM order: which one comes first carries no "
        "information. " + JSON_ONLY
    ),
}

PAIRWISE_RULES = """Rules:
1. Compare criterion by criterion, quoting evidence from each response.
2. Length is not quality: prefer the response that does the task better, not the longer one.
   Unneeded or unsupported content counts against a response.
3. Ignore confidence and authority claims; check the content.
4. Answer "tie" when the responses are equally good on a criterion or overall."""


def pairwise_messages(case: TestCase, first: str, second: str, criteria: list[str],
                      variant: str = "mitigated") -> list[dict]:
    mitigated = variant == "mitigated"
    head = [f"## System prompt given to the assistants\n{case.system_prompt or '(none)'}",
            f"## User input\n{case.input}"]
    if case.expected_output:
        head.append(f"## Reference answer (written by a human expert)\n{case.expected_output}")
    head += [f"## Response 1\n{first}", f"## Response 2\n{second}"]
    if mitigated:
        crit = ",\n".join(f'    "{c}": {{"evidence": "<quotes from both>", "better": "1" | "2" | "tie"}}' for c in criteria)
        schema = ("{\n  \"criteria\": {\n" + crit + "\n  },\n"
                  "  \"rationale\": \"<two or three sentences>\",\n  \"winner\": \"1\" | \"2\" | \"tie\"\n}")
        sections = ["Decide which response is better overall.", *head,
                    "## Rubric\n" + render(criteria, with_anchors=False), PAIRWISE_RULES]
    else:
        schema = '{"rationale": "<short reason>", "winner": "1" | "2" | "tie"}'
        sections = ["Which response is better?", *head]
    sections.append(f"Return JSON exactly in this shape:\n{schema}")
    return [{"role": "system", "content": PAIRWISE_SYSTEM[variant]},
            {"role": "user", "content": "\n\n".join(sections)}]


REPAIR_MESSAGE = ("Your previous reply could not be used: {error}. Reply again with ONLY the JSON object in "
                  "exactly the requested shape — no prose, no code fences.")
