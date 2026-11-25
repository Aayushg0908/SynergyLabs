"""Retrieval-only sweeps (no LLM calls): chunk size/overlap, then k.

For every chunking config the corpus is re-ingested (idempotency logic sees new
params and replaces each document's chunks), relevance labels are recomputed from the
evidence spans, and the retrieval metrics are measured. The default config is restored
at the end.

Run:  python eval/sweep.py
Out:  results/sweep_results.json, results/sweep_report.md
"""

import json
import logging
import statistics

from rag_app.config import PROJECT_ROOT
from rag_app.evaluation.harness import eval_retrieval, label_relevance, load_questions
from rag_app.pipeline import RAGService

# bge-small truncates input at 512 tokens; the title is prepended for embedding, so stay below that.
CHUNK_CONFIGS = [(96, 16), (128, 16), (256, 0), (256, 32), (256, 64), (384, 48), (480, 64)]
K_VALUES = [1, 3, 5, 10]
METRICS = ["hit_rate", "recall", "mrr", "ndcg", "precision_at_k", "context_precision_ap"]


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    svc = RAGService()
    questions = load_questions()
    default = (svc.settings.chunk_size, svc.settings.chunk_overlap)
    out = {"chunking": [], "k": []}

    for size, overlap in CHUNK_CONFIGS:
        svc.ingest(chunk_size=size, chunk_overlap=overlap)
        chunks = svc.store.all_chunks()
        labels = label_relevance(questions, chunks)
        summary, _ = eval_retrieval(svc, questions, labels, k=5)
        out["chunking"].append({
            "chunk_size": size, "chunk_overlap": overlap, "vectors": len(chunks),
            "mean_chunk_tokens": round(statistics.mean(len(svc.embedder.tokenizer.encode(c["content"])) for c in chunks), 1),
            "multi_doc_recall": summary["by_type"]["multi_doc"]["recall"],
            "unreachable": summary["questions_with_unreachable_spans"],
            **{m: summary[m] for m in METRICS},
        })
        print(f"chunk {size}/{overlap}: vectors={len(chunks)} " + " ".join(f"{m}={summary[m]:.3f}" for m in METRICS))

    svc.ingest(chunk_size=default[0], chunk_overlap=default[1])
    labels = label_relevance(questions, svc.store.all_chunks())
    for k in K_VALUES:
        summary, _ = eval_retrieval(svc, questions, labels, k=k)
        out["k"].append({"k": k, "multi_doc_recall": summary["by_type"]["multi_doc"]["recall"],
                         **{m: summary[m] for m in METRICS}})
        print(f"k={k}: " + " ".join(f"{m}={summary[m]:.3f}" for m in METRICS))

    results = PROJECT_ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "sweep_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    hdr = "| " + " | ".join(METRICS) + " | multi-doc recall |"
    sep = "|" + "---|" * (len(METRICS) + 1)
    lines = ["# Retrieval sweeps (k=5 unless stated; no LLM involved)", "",
             "## Chunk size / overlap (tokens)", "", "| size/overlap | vectors | mean tokens " + hdr[1:],
             "|---|---|" + sep]
    for r in out["chunking"]:
        lines.append(f"| {r['chunk_size']}/{r['chunk_overlap']} | {r['vectors']} | {r['mean_chunk_tokens']} | "
                     + " | ".join(f"{r[m]:.3f}" for m in METRICS) + f" | {r['multi_doc_recall']:.3f} |")
    lines += ["", f"## k (chunking {default[0]}/{default[1]})", "", "| k " + hdr, "|---" + sep]
    for r in out["k"]:
        lines.append(f"| {r['k']} | " + " | ".join(f"{r[m]:.3f}" for m in METRICS) + f" | {r['multi_doc_recall']:.3f} |")
    (results / "sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
