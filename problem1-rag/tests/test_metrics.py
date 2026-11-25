"""Metric correctness against hand-computed values."""

import math

import numpy as np
import pytest

from rag_app.evaluation.metrics import (
    average_precision, contains_gold, contains_span, exact_match, hit_rate, ndcg_at_k, percentile,
    precision_at_k, reciprocal_rank, token_f1,
)

T, F = True, False


def test_hit_rate_and_mrr():
    assert hit_rate([F, F, T, F, F]) == 1.0
    assert hit_rate([F] * 5) == 0.0
    assert reciprocal_rank([F, F, T, F, F]) == pytest.approx(1 / 3)
    assert reciprocal_rank([T, T, F]) == 1.0
    assert reciprocal_rank([F, F]) == 0.0


def test_ndcg_hand_computed():
    # one relevant chunk at rank 2, one relevant chunk exists -> 1/log2(3)
    assert ndcg_at_k([F, T, F, F, F], 1) == pytest.approx(1 / math.log2(3))
    # perfect ranking
    assert ndcg_at_k([T, T, F, F, F], 2) == pytest.approx(1.0)
    # relevant at ranks 1 and 3, two relevant exist: (1 + 1/2) / (1 + 1/log2(3))
    assert ndcg_at_k([T, F, T, F, F], 2) == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    # more relevant chunks than k: ideal is capped at k
    assert ndcg_at_k([T, T], 5) == pytest.approx(1.0)
    assert ndcg_at_k([F, F], 0) == 0.0


def test_precision_and_average_precision():
    assert precision_at_k([T, F, T, F, F]) == pytest.approx(0.4)
    # AP: relevant at ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert average_precision([T, F, T, F, F], 2) == pytest.approx((1 + 2 / 3) / 2)
    # one relevant at rank 2 -> (1/2) / 1
    assert average_precision([F, T, F], 1) == pytest.approx(0.5)


def test_percentile_matches_numpy():
    xs = [12.0, 3.5, 7.1, 9.9, 1.2, 30.0, 4.4]
    for q in (50, 95, 99):
        assert percentile(xs, q) == pytest.approx(np.percentile(xs, q))


def test_answer_metrics():
    assert exact_match("$1,490 per drone per year.", "$1,490 per drone per year") == 1.0
    assert contains_gold("Care+ costs $1,490 per drone per year [1].", "$1,490 per drone per year") == 1.0
    assert contains_gold("It is 0.5% [2].", "0.5%") == 1.0
    assert contains_gold("exceeds 0.5 % of devices", "0.5%") == 1.0
    assert contains_gold("SEV‑2 [1]", "SEV-2") == 1.0
    # articles are dropped: pred has 5 tokens, gold 9, overlap 5 -> P=1, R=5/9, F1=10/14
    assert token_f1("the release captain and QA lead", "the release captain, the QA lead and the head of flight safety") \
        == pytest.approx(10 / 14)


def test_span_matching_ignores_markdown_and_whitespace():
    assert contains_span("Lag above 10 minutes, or more\nthan 25% of gateways disconnected: **SEV-2**.",
                         "Lag above 10 minutes, or more than 25% of gateways disconnected: SEV-2")
