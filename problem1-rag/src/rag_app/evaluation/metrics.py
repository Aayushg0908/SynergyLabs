"""Pure metric functions (no I/O) so they can be unit-tested against hand-computed values.

Retrieval metrics use binary relevance. `rels` is the list of relevance flags of the
retrieved chunks in rank order (rank 1 first); `n_relevant` is the number of relevant
chunks that exist in the whole index for the question (needed for nDCG's ideal ranking).
"""

import math
import re
import string
import unicodedata
from collections import Counter

# ----------------------------------------------------------------- text normalisation
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def normalize_span(text: str) -> str:
    """For evidence matching: unicode-normalise, drop markdown emphasis, collapse whitespace, lower-case."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES)
    text = text.replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_span(chunk_text: str, span: str) -> bool:
    return normalize_span(span) in normalize_span(chunk_text)


# ----------------------------------------------------------------- retrieval metrics
def hit_rate(rels: list[bool]) -> float:
    return 1.0 if any(rels) else 0.0


def reciprocal_rank(rels: list[bool]) -> float:
    for i, r in enumerate(rels, 1):
        if r:
            return 1.0 / i
    return 0.0


def dcg(rels: list[bool]) -> float:
    return sum((1.0 if r else 0.0) / math.log2(i + 1) for i, r in enumerate(rels, 1))


def ndcg_at_k(rels: list[bool], n_relevant: int) -> float:
    if n_relevant == 0:
        return 0.0
    ideal = dcg([True] * min(len(rels), n_relevant))
    return dcg(rels) / ideal if ideal else 0.0


def precision_at_k(rels: list[bool]) -> float:
    return sum(rels) / len(rels) if rels else 0.0


def average_precision(rels: list[bool], n_relevant: int) -> float:
    """Rank-aware context precision (as in RAGAS): mean of precision@i over the ranks i
    that hold a relevant chunk, normalised by the number of relevant chunks retrievable in k."""
    denom = min(len(rels), n_relevant)
    if denom == 0:
        return 0.0
    hits, total = 0, 0.0
    for i, r in enumerate(rels, 1):
        if r:
            hits += 1
            total += hits / i
    return total / denom


def evidence_recall(spans_found: list[bool]) -> float:
    """Recall@k over evidence spans: share of the question's required facts present in the top-k."""
    return sum(spans_found) / len(spans_found) if spans_found else 0.0


# ----------------------------------------------------------------- answer metrics (SQuAD-style)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_DASHES).lower()
    text = re.sub(r"\[\d+\]", " ", text)  # drop citation markers
    text = text.replace(",", "")          # 1,490 == 1490
    text = re.sub(r"(\d)\s+%", r"\1%", text)  # "0.5 %" == "0.5%"
    text = "".join(ch for ch in text if ch not in set(string.punctuation) - {"$", "%", "."} )
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)  # keep decimal points only
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def contains_gold(prediction: str, gold: str) -> float:
    """Lenient EM: the normalised gold answer appears verbatim inside the normalised prediction."""
    return float(normalize_answer(gold) in normalize_answer(prediction))


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(pred), overlap / len(ref)
    return 2 * p * r / (p + r)


# ----------------------------------------------------------------- aggregation
def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (same as numpy's default)."""
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")
