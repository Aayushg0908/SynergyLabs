"""The judge: pointwise scoring and pairwise comparison with order swapping."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable

from . import prompts
from .llm import ChatClient
from .parsing import VerdictParseError, extract_json, pairwise_choice, pointwise_scores, salvage_scores
from .rubric import criteria_for, overall, passed
from .schema import CriterionScore, PairwiseVerdict, PointwiseVerdict, TestCase


class Judge:
    def __init__(self, client: ChatClient, variant: str = "mitigated"):
        if variant not in ("naive", "mitigated"):
            raise ValueError("variant must be 'naive' or 'mitigated'")
        self.client, self.variant = client, variant

    @property
    def name(self) -> str:
        return f"{self.client.name} [{self.variant}]"

    # ------------------------------------------------------------ pointwise
    def pointwise(self, case: TestCase, response: str | None = None, sample: int = 0,
                  tag: str = "pointwise") -> PointwiseVerdict:
        criteria = criteria_for(case.criteria)
        messages = prompts.pointwise_messages(case, criteria, self.variant, response)
        meta = {"case_id": case.id, "mode": "pointwise", "variant": self.variant}
        call_ids, status, error = [], "ok", None
        try:
            r = self.client.chat(messages, json_mode=True, tag=tag, sample=sample, meta=meta)
        except RuntimeError as exc:
            return self._failed(case.id, f"api: {exc}", call_ids)
        call_ids.append(r.call_id)
        try:
            obj, status = extract_json(r.text)
            scores, claimed = pointwise_scores(obj, criteria)
        except VerdictParseError as first_error:
            # Retry once, showing the model its own reply and the parse error.
            retry_msgs = messages + [{"role": "assistant", "content": r.text},
                                     {"role": "user", "content": prompts.REPAIR_MESSAGE.format(error=first_error)}]
            try:
                r2 = self.client.chat(retry_msgs, json_mode=True, tag=tag + ":repair", sample=sample, meta=meta)
                call_ids.append(r2.call_id)
                obj, _ = extract_json(r2.text)
                scores, claimed = pointwise_scores(obj, criteria)
                status = "retried"
            except (VerdictParseError, RuntimeError) as second_error:
                salvaged = salvage_scores(r.text, criteria)
                if not salvaged:
                    return self._failed(case.id, f"parse: {first_error}; retry: {second_error}", call_ids)
                scores, claimed, status = salvaged, None, "salvaged"
        ov = overall(scores)
        return PointwiseVerdict(case_id=case.id, criteria=scores, overall=ov, passed=passed(scores, ov),
                                judge_overall_claimed=claimed, parse_status=status, error=error, call_ids=call_ids)

    @staticmethod
    def _failed(case_id: str, error: str, call_ids: list[str]) -> PointwiseVerdict:
        return PointwiseVerdict(case_id=case_id, criteria={}, overall=None, passed=False,
                                parse_status="failed", error=error, call_ids=call_ids)

    # ------------------------------------------------------------ pairwise
    def pairwise_once(self, case: TestCase, out_a: str, out_b: str, order: str, sample: int = 0,
                      tag: str = "pairwise") -> PairwiseVerdict:
        """Compare A and B, presenting them in `order` ('AB' = A shown first). The returned
        winner is always expressed in the ORIGINAL labels A / B."""
        criteria = criteria_for(case.criteria)
        first, second = (out_a, out_b) if order == "AB" else (out_b, out_a)
        messages = prompts.pairwise_messages(case, first, second, criteria, self.variant)
        meta = {"case_id": case.id, "mode": "pairwise", "order": order, "variant": self.variant}
        to_label = {"1": order[0], "2": order[1], "tie": "tie"}
        call_ids, status = [], "ok"
        try:
            r = self.client.chat(messages, json_mode=True, tag=tag, sample=sample, meta=meta)
            call_ids.append(r.call_id)
            try:
                obj, status = extract_json(r.text)
                win, per, why = pairwise_choice(obj, criteria)
            except VerdictParseError as err:
                retry_msgs = messages + [{"role": "assistant", "content": r.text},
                                         {"role": "user", "content": prompts.REPAIR_MESSAGE.format(error=err)}]
                r2 = self.client.chat(retry_msgs, json_mode=True, tag=tag + ":repair", sample=sample, meta=meta)
                call_ids.append(r2.call_id)
                obj, _ = extract_json(r2.text)
                win, per, why = pairwise_choice(obj, criteria)
                status = "retried"
        except (VerdictParseError, RuntimeError) as exc:
            return PairwiseVerdict(case_id=case.id, winner="tie", order=order, parse_status="failed",
                                   error=str(exc), call_ids=call_ids)
        return PairwiseVerdict(case_id=case.id, winner=to_label[win], order=order,
                               criteria={k: to_label[v] for k, v in per.items()}, rationale=why,
                               parse_status=status, call_ids=call_ids)

    def pairwise(self, case: TestCase, out_a: str, out_b: str, both_orders: bool = True,
                 tag: str = "pairwise") -> dict:
        """Position-bias mitigation: judge (A,B) and (B,A). The final verdict requires the two
        orders to agree; if they disagree the pair is a tie and counted as a position flip."""
        ab = self.pairwise_once(case, out_a, out_b, "AB", tag=tag)
        if not both_orders:
            return {"case_id": case.id, "ab": ab, "ba": None, "final": ab.winner, "flip": None}
        ba = self.pairwise_once(case, out_a, out_b, "BA", tag=tag)
        valid = ab.parse_status != "failed" and ba.parse_status != "failed"
        flip = valid and ab.winner != ba.winner
        final = ab.winner if valid and not flip else "tie"
        return {"case_id": case.id, "ab": ab, "ba": ba, "final": final, "flip": flip if valid else None}


def run_parallel(fn: Callable, items: Iterable, workers: int = 4, label: str = "") -> list:
    """Run fn over items concurrently, preserving input order, with a progress counter."""
    items = list(items)
    out = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, it): i for i, it in enumerate(items)}
        for done, fut in enumerate(as_completed(futures), 1):
            out[futures[fut]] = fut.result()
            print(f"\r  {label} {done}/{len(items)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return out


def score_of(v: PointwiseVerdict, criterion: str) -> int | None:
    s: CriterionScore | None = v.criteria.get(criterion)
    return s.score if s else None
