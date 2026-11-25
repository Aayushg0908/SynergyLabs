"""Robust extraction of structured verdicts from judge output.

Layers, in order (each result records which one succeeded):
  1. strict json.loads on the reply with reasoning tags and code fences stripped    -> "ok"
  2. lenient repair: pick the first balanced {...} object, fix smart quotes,
     trailing commas, Python literals and single-quoted strings                     -> "repaired"
  3. (in judge.py) one retry asking the model to re-emit valid JSON                 -> "retried"
  4. regex salvage of `"criterion": ... "score": n` pairs                          -> "salvaged"
Anything else is "failed" and is counted, never silently dropped.
"""

import ast
import json
import re

from .schema import CriterionScore


class VerdictParseError(ValueError):
    pass


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _balanced_object(text: str) -> str | None:
    """Return the first top-level {...} block, respecting strings and escapes."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _repair(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)                         # trailing commas
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return s


def extract_json(text: str) -> tuple[dict, str]:
    """Return (object, status) where status is 'ok' or 'repaired'; raise VerdictParseError otherwise."""
    if not text or not text.strip():
        raise VerdictParseError("empty response")
    cleaned = _FENCE.sub("", _THINK.sub("", text)).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj, "ok"
    except json.JSONDecodeError:
        pass
    block = _balanced_object(cleaned)
    if block is None:
        raise VerdictParseError("no JSON object found")
    for candidate in (block, _repair(block)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, "repaired"
        except json.JSONDecodeError:
            continue
    try:  # single-quoted dicts etc.: Python literal syntax
        obj = ast.literal_eval(_repair(block).replace("true", "True").replace("false", "False").replace("null", "None"))
        if isinstance(obj, dict):
            return obj, "repaired"
    except (ValueError, SyntaxError):
        pass
    raise VerdictParseError("JSON object could not be parsed or repaired")


def coerce_score(value) -> int | None:
    """'4', 4.0, '4/5', '4 out of 5', 'N/A', None -> int in 1..5 or None. Out of range -> error."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise VerdictParseError(f"invalid score {value!r}")
    if isinstance(value, (int, float)):
        n = value
    else:
        s = str(value).strip().lower()
        if s in {"", "n/a", "na", "null", "none", "not applicable"}:
            return None
        m = re.match(r"^(\d+(?:\.\d+)?)", s)
        if not m:
            raise VerdictParseError(f"invalid score {value!r}")
        n = float(m.group(1))
    n = int(round(n))
    if not 1 <= n <= 5:
        raise VerdictParseError(f"score {value!r} outside 1-5")
    return n


def pointwise_scores(obj: dict, criteria: list[str]) -> tuple[dict[str, CriterionScore], float | None]:
    """Validate a pointwise verdict. Accepts {"criteria": {...}} or the criteria at top level,
    and each criterion as an object or a bare score."""
    crit = obj.get("criteria", obj)
    if not isinstance(crit, dict):
        raise VerdictParseError("criteria is not an object")
    out = {}
    for name in criteria:
        if name not in crit:
            raise VerdictParseError(f"missing criterion {name!r}")
        v = crit[name]
        if isinstance(v, dict):
            out[name] = CriterionScore(score=coerce_score(v.get("score")), evidence=str(v.get("evidence", "") or ""),
                                       rationale=str(v.get("rationale", "") or ""))
        else:
            out[name] = CriterionScore(score=coerce_score(v))
    if all(s.score is None for s in out.values()):
        raise VerdictParseError("no criterion was scored")
    claimed = obj.get("overall")
    try:
        claimed = float(claimed) if claimed is not None else None
    except (TypeError, ValueError):
        claimed = None
    return out, claimed


_SALVAGE = re.compile(r'"?(?P<name>[a-z_]+)"?\s*:\s*\{[^{}]*?"?score"?\s*:\s*"?(?P<score>\d|null)', re.S)


def salvage_scores(text: str, criteria: list[str]) -> dict[str, CriterionScore] | None:
    found = {m.group("name"): m.group("score") for m in _SALVAGE.finditer(text or "")}
    if not all(c in found for c in criteria):
        return None
    return {c: CriterionScore(score=None if found[c] == "null" else coerce_score(found[c]), rationale="(salvaged)")
            for c in criteria}


def pairwise_choice(obj: dict, criteria: list[str] | None = None) -> tuple[str, dict[str, str], str]:
    """Return (winner '1'|'2'|'tie', per-criterion choices, rationale)."""
    def norm(v) -> str:
        s = str(v).strip().lower().replace("response", "").strip().strip('"')
        if s in {"1", "a", "first"}:
            return "1"
        if s in {"2", "b", "second"}:
            return "2"
        if s in {"tie", "equal", "draw", "both", "none"}:
            return "tie"
        raise VerdictParseError(f"invalid choice {v!r}")

    if "winner" not in obj:
        raise VerdictParseError("missing winner")
    per = {}
    for name, v in (obj.get("criteria") or {}).items():
        if criteria and name not in criteria:
            continue
        try:
            per[name] = norm(v.get("better") if isinstance(v, dict) else v)
        except VerdictParseError:
            continue
    return norm(obj["winner"]), per, str(obj.get("rationale", ""))
