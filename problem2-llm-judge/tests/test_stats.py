"""Statistics and pairwise-aggregation logic against hand-computed values."""

import pytest

from llm_judge import stats
from llm_judge.judge import Judge
from llm_judge.llm import ChatResult
from llm_judge.schema import TestCase


def test_cohen_kappa_hand_computed():
    # 10 items: 4 both yes, 4 both no, 2 disagree -> po=0.8; pa(yes)=0.5, pb(yes)=0.5 -> pe=0.5 -> kappa=0.6
    a = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    b = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    assert stats.cohen_kappa(a, b) == pytest.approx(0.6)
    assert stats.cohen_kappa(a, a) == pytest.approx(1.0)


def test_weighted_kappa_penalises_far_disagreement_more():
    gold = ["bad", "partial", "good", "good"]
    near = ["partial", "partial", "good", "good"]
    far = ["good", "partial", "good", "good"]
    order = ["bad", "partial", "good"]
    assert stats.cohen_kappa(near, gold, "linear", order) > stats.cohen_kappa(far, gold, "linear", order)


def test_sign_test():
    assert stats.sign_test_p(10, 0) == pytest.approx(2 / 1024)
    assert stats.sign_test_p(5, 5) == 1.0
    assert stats.sign_test_p(0, 0) == 1.0


def test_spearman_and_entropy():
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert stats.entropy({"a": 1, "b": 1}) == pytest.approx(1.0)


class ScriptedClient:
    """Returns pairwise winners from a script keyed by which output is shown first."""
    provider, model, name = "fake", "fake", "fake/fake"

    def __init__(self, prefer_first: bool):
        self.prefer_first = prefer_first

    def chat(self, messages, json_mode=False, tag="", sample=0, meta=None):
        user = messages[-1]["content"]
        first_is_good = user.index("GOOD") < user.index("BAD")
        win = "1" if (self.prefer_first or first_is_good) else "2"
        return ChatResult(f'{{"winner": "{win}", "rationale": "x"}}', 1, 1, 1.0, "fake", "fake", False, "id")


CASE = TestCase(id="c1", input="q", model_output="")


def test_pairwise_maps_back_to_original_labels_and_detects_flips():
    consistent = Judge(ScriptedClient(prefer_first=False), "naive").pairwise(CASE, "GOOD answer", "BAD answer")
    assert consistent["ab"].winner == "A" and consistent["ba"].winner == "A"
    assert consistent["final"] == "A" and consistent["flip"] is False

    biased = Judge(ScriptedClient(prefer_first=True), "naive").pairwise(CASE, "GOOD answer", "BAD answer")
    assert biased["ab"].winner == "A" and biased["ba"].winner == "B"   # always picks whatever is shown first
    assert biased["flip"] is True and biased["final"] == "tie"         # disagreement -> tie
