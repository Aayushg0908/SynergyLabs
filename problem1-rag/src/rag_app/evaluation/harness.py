"""Three-layer evaluation: retrieval (IR metrics), answers (gold + LLM judge), latency.

Run:  rag eval            (full)
      rag eval --no-judge (skip Gemini)
      rag eval --retrieval-only
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..config import PROJECT_ROOT
from ..llm import LLMClient
from ..pipeline import RAGService
from . import metrics as M
from .judge import AnswerJudge

log = logging.getLogger("rag.eval")

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


def label_relevance(questions: list[dict], chunks: list[dict]) -> dict[str, dict]:
    """For each question: the set of relevant chunk IDs and, per evidence span, the chunks containing it."""
    labels = {}
    for q in questions:
        per_span = []
        for ev in q["evidence"]:
            ids = {c["chunk_id"] for c in chunks if c["doc_id"] == ev["doc"] and M.contains_span(c["content"], ev["span"])}
            per_span.append(ids)
        relevant = set().union(*per_span) if per_span else set()
        labels[q["id"]] = {"relevant": relevant, "per_span": per_span,
                           "unreachable_spans": [q["evidence"][i]["span"] for i, s in enumerate(per_span) if not s]}
    return labels


# ------------------------------------------------------------------ layer 1: retrieval
def eval_retrieval(svc: RAGService, questions: list[dict], labels: dict, k: int) -> tuple[dict, list[dict]]:
    rows = []
    for q in questions:
        hits, _ = svc.retrieve(q["question"], k, q.get("doc_type"))
        lab = labels[q["id"]]
        row = {"id": q["id"], "type": q["type"], "top_score": round(hits[0].score, 4) if hits else None,
               "retrieved": [{"doc_id": h.doc_id, "chunk_id": h.chunk_id, "score": round(h.score, 4),
                              "relevant": h.chunk_id in lab["relevant"]} for h in hits]}
        if q["type"] != "unanswerable":
            rels = [h.chunk_id in lab["relevant"] for h in hits]
            in_ctx = [r for h, r in zip(hits, rels) if h.score >= svc.settings.min_score]
            row.update({
                "n_relevant_in_index": len(lab["relevant"]),
                "hit": M.hit_rate(rels),
                "recall": M.evidence_recall([bool(ids & {h.chunk_id for h in hits}) for ids in lab["per_span"]]),
                "rr": M.reciprocal_rank(rels),
                "ndcg": M.ndcg_at_k(rels, len(lab["relevant"])),
                "precision": M.precision_at_k(rels),
                "ap": M.average_precision(rels, len(lab["relevant"])),
                "in_context_precision": M.precision_at_k(in_ctx) if in_ctx else 0.0,
                "unreachable_spans": lab["unreachable_spans"],
            })
        else:
            row["rejected_by_threshold"] = all(h.score < svc.settings.min_score for h in hits)
        rows.append(row)

    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]

    def agg(rs):
        return {"n": len(rs),
                "hit_rate": M.mean([r["hit"] for r in rs]), "recall": M.mean([r["recall"] for r in rs]),
                "mrr": M.mean([r["rr"] for r in rs]), "ndcg": M.mean([r["ndcg"] for r in rs]),
                "precision_at_k": M.mean([r["precision"] for r in rs]),
                "context_precision_ap": M.mean([r["ap"] for r in rs]),
                "in_context_precision": M.mean([r["in_context_precision"] for r in rs])}

    summary = {"k": k, "min_score": svc.settings.min_score, **agg(ans),
               "by_type": {t: agg([r for r in ans if r["type"] == t]) for t in sorted({r["type"] for r in ans})},
               "unanswerable_rejected_by_threshold": M.mean([float(r["rejected_by_threshold"]) for r in una]) if una else None,
               "relevant_top_score_mean": M.mean([r["top_score"] for r in ans if r["hit"]]),
               "unanswerable_top_score_mean": M.mean([r["top_score"] for r in una]) if una else None,
               "questions_with_unreachable_spans": [r["id"] for r in ans if r["unreachable_spans"]]}
    return summary, rows


# ------------------------------------------------------------------ layer 3: latency
def eval_latency(svc: RAGService, questions: list[dict], k: int, reps: int) -> dict:
    for q in questions[:3]:  # warm-up: model kernels, DB connection, planner caches
        svc.retrieve(q["question"], k, q.get("doc_type"))
    embed, search, both = [], [], []
    for _ in range(reps):
        for q in questions:
            _, t = svc.retrieve(q["question"], k, q.get("doc_type"))
            embed.append(t["embed_ms"])
            search.append(t["retrieval_ms"])
            both.append(t["embed_ms"] + t["retrieval_ms"])

    def pct(xs):
        return {"p50": round(M.percentile(xs, 50), 2), "p95": round(M.percentile(xs, 95), 2),
                "mean": round(M.mean(xs), 2), "n": len(xs)}

    return {"vector_search_ms": pct(search), "query_embedding_ms": pct(embed), "embed_plus_search_ms": pct(both),
            "k": k, "vectors_in_index": svc.store.stats()["vectors"], "reps_per_question": reps}


# ------------------------------------------------------------------ layer 2: answers
def eval_answers(svc: RAGService, questions: list[dict], labels: dict, k: int, judge: AnswerJudge | None,
                 gen_interval: float, judge_interval: float) -> tuple[dict, list[dict]]:
    rows = []
    for i, q in enumerate(questions, 1):
        resp = svc.answer(q["question"], k, q.get("doc_type"))
        log.info("[%d/%d] %s no_answer=%s", i, len(questions), q["id"], resp["no_answer"])
        relevant = labels[q["id"]]["relevant"]
        row = {"id": q["id"], "type": q["type"], "question": q["question"], "gold": q["gold"],
               "answer": resp["answer"], "no_answer": resp["no_answer"], "no_answer_reason": resp["no_answer_reason"],
               "citations": [c["doc_id"] for c in resp["citations"]],
               "latency_ms": resp["latency_ms"], "usage": resp["usage"]}
        if q["type"] != "unanswerable":
            row["em"] = M.exact_match(resp["answer"], q["gold"])
            row["contains_gold"] = M.contains_gold(resp["answer"], q["gold"])
            row["f1"] = M.token_f1(resp["answer"], q["gold"])
            row["citation_correct"] = any(c["chunk_id"] in relevant for c in resp["citations"]) if resp["citations"] else False
            row["context_passages"] = [r["text"] for r in resp["retrieved"] if r["in_context"]]
            if judge and not resp["no_answer"]:
                row["judge"] = judge.judge(q["id"], q["question"], resp["answer"], q["gold"], row["context_passages"])
                time.sleep(judge_interval)
        rows.append(row)
        if resp["usage"]["total_tokens"]:
            time.sleep(gen_interval)

    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    answered = [r for r in ans if not r["no_answer"]]
    judged = [r["judge"] for r in answered if r.get("judge") and not r["judge"].get("judge_error")]
    e2e = [r["latency_ms"]["total_ms"] for r in rows if r["usage"]["total_tokens"]]
    gen = [r["latency_ms"]["generation_ms"] for r in rows if r["usage"]["total_tokens"]]

    summary = {
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "exact_match": M.mean([r["em"] for r in ans]),
        "contains_gold": M.mean([r["contains_gold"] for r in ans]),
        "token_f1": M.mean([r["f1"] for r in ans]),
        "citation_accuracy": M.mean([float(r["citation_correct"]) for r in answered]) if answered else None,
        "answerable_false_abstention_rate": M.mean([float(r["no_answer"]) for r in ans]),
        "unanswerable_correct_abstention_rate": M.mean([float(r["no_answer"]) for r in una]) if una else None,
        "unanswerable_abstention_by": dict(_count(r["no_answer_reason"] for r in una)),
        "judge": None,
        "end_to_end_ms": {"p50": round(M.percentile(e2e, 50), 1), "p95": round(M.percentile(e2e, 95), 1), "n": len(e2e)},
        "generation_ms": {"p50": round(M.percentile(gen, 50), 1), "p95": round(M.percentile(gen, 95), 1)},
        "mean_tokens_per_llm_query": {
            "prompt": M.mean([r["usage"]["prompt_tokens"] for r in rows if r["usage"]["total_tokens"]]),
            "completion": M.mean([r["usage"]["completion_tokens"] for r in rows if r["usage"]["total_tokens"]])},
    }
    if judge:
        summary["judge"] = judge_summary(rows, judge)
    return summary, rows


def judge_summary(rows: list[dict], judge: AnswerJudge) -> dict:
    answered = [r for r in rows if r["type"] != "unanswerable" and not r["no_answer"]]
    judged = [r["judge"] for r in answered if r.get("judge") and not r["judge"].get("judge_error")]
    return {
        "model": f"{judge.client.provider}/{judge.client.model}",
        "n_answered": len(answered), "n_judged": len(judged),
        "judge_errors": sum(1 for r in answered if (r.get("judge") or {}).get("judge_error")),
        "faithfulness": M.mean([j["faithfulness"] for j in judged]),
        "answers_with_unsupported_claim": M.mean([float(j["n_supported"] < j["n_claims"]) for j in judged]),
        "answer_relevance_1to5": M.mean([j["answer_relevance"] for j in judged]),
        "answer_relevance_norm": M.mean([(j["answer_relevance"] - 1) / 4 for j in judged]),
        "correctness_1to5": M.mean([j["correctness"] for j in judged]),
        "judge_calls": judge.calls, "judge_tokens": judge.tokens,
    }


def make_judge(svc: RAGService) -> AnswerJudge:
    s = svc.settings
    return AnswerJudge(LLMClient(s.judge_provider, s.judge_model, s.judge_api_key, max_tokens=8192),
                       s.log_dir / "judge_calls.jsonl")


def rejudge(svc: RAGService, judge_interval: float = 7.0, out_dir: Path = RESULTS_DIR, tag: str = "eval",
            all_rows: bool = False) -> dict:
    """Judge answers from a saved run whose verdict is missing or failed (or all, with all_rows),
    without regenerating them. Passages are re-retrieved when an older run did not save them."""
    path = out_dir / f"{tag}_results.json"
    full = json.loads(path.read_text(encoding="utf-8"))
    rows = full["per_question"]["answers"]
    questions = {q["id"]: q for q in load_questions()}
    judge = make_judge(svc)
    for row in rows:
        if row["type"] == "unanswerable" or row["no_answer"]:
            continue
        if not all_rows and row.get("judge") and not row["judge"].get("judge_error"):
            continue
        q = questions[row["id"]]
        if "context_passages" not in row:
            hits, _ = svc.retrieve(q["question"], full["config"]["k"], q.get("doc_type"))
            row["context_passages"] = [h.content for h in hits if h.score >= svc.settings.min_score]
        log.info("judging %s", row["id"])
        row["judge"] = judge.judge(row["id"], q["question"], row["answer"], q["gold"], row["context_passages"])
        time.sleep(judge_interval)
    full["answers"]["judge"] = judge_summary(rows, judge)
    path.write_text(json.dumps(full, indent=2, default=float), encoding="utf-8")
    (out_dir / f"{tag}_report.md").write_text(render_report(full, rows), encoding="utf-8")
    return full["answers"]["judge"]


def _count(items):
    d = defaultdict(int)
    for x in items:
        d[str(x)] += 1
    return d


# ------------------------------------------------------------------ orchestration
def run(svc: RAGService, k: int | None = None, use_judge: bool = True, retrieval_only: bool = False,
        latency_reps: int = 10, gen_interval: float = 2.0, judge_interval: float = 7.0,
        out_dir: Path = RESULTS_DIR, tag: str = "eval") -> dict:
    k = k or svc.settings.top_k
    questions = load_questions()
    chunks = svc.store.all_chunks()
    labels = label_relevance(questions, chunks)
    stats = svc.store.stats()

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"k": k, "min_score": svc.settings.min_score, "chunk_size": svc.settings.chunk_size,
                   "chunk_overlap": svc.settings.chunk_overlap, "embedding_model": svc.embedder.model_name,
                   "embedding_dim": svc.embedder.dim, "llm": f"{svc.settings.llm_provider}/{svc.settings.llm_model}",
                   "vector_store": "postgres+pgvector (HNSW, cosine)", "documents": stats["documents"],
                   "vectors": stats["vectors"], "n_questions": len(questions)},
    }
    log.info("retrieval layer: %d questions, k=%d", len(questions), k)
    result["retrieval"], retrieval_rows = eval_retrieval(svc, questions, labels, k)
    log.info("latency layer: %d reps", latency_reps)
    result["latency"] = eval_latency(svc, questions, k, latency_reps)
    answer_rows = []
    if not retrieval_only:
        judge = make_judge(svc) if use_judge else None
        log.info("answer layer (judge=%s)", bool(judge))
        result["answers"], answer_rows = eval_answers(svc, questions, labels, k, judge, gen_interval, judge_interval)

    out_dir.mkdir(parents=True, exist_ok=True)
    full = {**result, "per_question": {"retrieval": retrieval_rows, "answers": answer_rows}}
    (out_dir / f"{tag}_results.json").write_text(json.dumps(full, indent=2, default=float), encoding="utf-8")
    if not retrieval_only:
        (out_dir / f"{tag}_report.md").write_text(render_report(result, answer_rows), encoding="utf-8")
    return result


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_report(r: dict, answer_rows: list[dict]) -> str:
    c, ret, lat, ans = r["config"], r["retrieval"], r["latency"], r["answers"]
    j = ans.get("judge") or {}
    lines = [
        "# RAG evaluation report", "",
        f"Generated {r['timestamp']} — {c['n_questions']} questions ({ans['n_answerable']} answerable, "
        f"{ans['n_unanswerable']} unanswerable), {c['documents']} documents / {c['vectors']} vectors.", "",
        f"Config: k={c['k']}, chunk {c['chunk_size']}/{c['chunk_overlap']} tokens, embeddings {c['embedding_model']} "
        f"({c['embedding_dim']}-d), generator {c['llm']}, judge {j.get('model', 'none')}, min_score={c['min_score']}.", "",
        "## 1. Retrieval (answerable questions)", "",
        "| Metric | Value |", "|---|---|",
        f"| Hit rate@{c['k']} | {_f(ret['hit_rate'])} |",
        f"| Recall@{c['k']} (evidence spans) | {_f(ret['recall'])} |",
        f"| MRR | {_f(ret['mrr'])} |",
        f"| nDCG@{c['k']} | {_f(ret['ndcg'])} |",
        f"| Precision@{c['k']} | {_f(ret['precision_at_k'])} |",
        f"| Context precision (rank-aware AP) | {_f(ret['context_precision_ap'])} |",
        f"| Precision of chunks passed to the LLM (score ≥ min_score) | {_f(ret['in_context_precision'])} |",
        f"| Unanswerable rejected by similarity threshold | {_f(ret['unanswerable_rejected_by_threshold'])} |", "",
        "## 2. Answers", "",
        "| Metric | Value |", "|---|---|",
        f"| Faithfulness (supported claims / claims, judge) | {_f(j.get('faithfulness'))} |",
        f"| Answers with ≥1 unsupported claim | {_f(j.get('answers_with_unsupported_claim'))} |",
        f"| Answer relevance (1–5, judge) | {_f(j.get('answer_relevance_1to5'), 2)} |",
        f"| Correctness vs gold (1–5, judge) | {_f(j.get('correctness_1to5'), 2)} |",
        f"| Exact match | {_f(ans['exact_match'])} |",
        f"| Gold answer contained in response | {_f(ans['contains_gold'])} |",
        f"| Token F1 | {_f(ans['token_f1'])} |",
        f"| Citation accuracy (cites a relevant chunk) | {_f(ans['citation_accuracy'])} |",
        f"| False abstention on answerable | {_f(ans['answerable_false_abstention_rate'])} |",
        f"| Correct abstention on unanswerable | {_f(ans['unanswerable_correct_abstention_rate'])} |", "",
        "## 3. Latency", "",
        "| Stage | p50 ms | p95 ms |", "|---|---|---|",
        f"| Vector search (SQL, k={lat['k']}, {lat['vectors_in_index']} vectors) | {lat['vector_search_ms']['p50']} | {lat['vector_search_ms']['p95']} |",
        f"| Query embedding (CPU) | {lat['query_embedding_ms']['p50']} | {lat['query_embedding_ms']['p95']} |",
        f"| Retrieval total (embed + search) | {lat['embed_plus_search_ms']['p50']} | {lat['embed_plus_search_ms']['p95']} |",
        f"| LLM generation | {ans['generation_ms']['p50']} | {ans['generation_ms']['p95']} |",
        f"| End-to-end (LLM-answered queries) | {ans['end_to_end_ms']['p50']} | {ans['end_to_end_ms']['p95']} |", "",
        f"Judge: {j.get('n_judged', 0)}/{j.get('n_answered', 0)} answered questions judged, "
        f"{j.get('judge_errors', 0)} judge errors, {j.get('judge_calls', 0)} calls, {j.get('judge_tokens', 0)} tokens.", "",
        "## Per-question answers", "",
        "| id | type | gold | answer | F1 | faithful | correct |", "|---|---|---|---|---|---|---|",
    ]
    for row in answer_rows:
        jj = row.get("judge") or {}
        answer = row["answer"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['id']} | {row['type']} | {row['gold'] or '—'} | {answer} | "
                     f"{_f(row.get('f1'), 2)} | {_f(jj.get('faithfulness'), 2)} | {jj.get('correctness', '—')} |")
    return "\n".join(lines) + "\n"
