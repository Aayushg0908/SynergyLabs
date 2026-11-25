"""Ingestion and question answering."""

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .chunking import Chunker
from .config import PROJECT_ROOT, Settings, get_settings
from .embeddings import Embedder
from .llm import LLMClient
from .loaders import discover, file_sha256, load_document
from .store import Hit, VectorStore

log = logging.getLogger("rag")

NO_CONTEXT_SENTINEL = "INSUFFICIENT_CONTEXT"
NO_CONTEXT_ANSWER = "I couldn't find information about this in the available documents."

SYSTEM_PROMPT = f"""You answer questions about Kestrel Labs using ONLY the numbered context passages provided.

Rules:
- Use only facts stated in the passages. Never use outside knowledge or guess.
- Cite every factual sentence with the number of the passage it came from, in square brackets, e.g. [2] or [1][3].
- Be concise: answer the question directly in 1-4 sentences.
- If the passages do not contain the information needed to answer, reply with exactly: {NO_CONTEXT_SENTINEL}
- If the passages answer only part of the question, answer that part and say what is not covered."""


_CITE_VARIANTS = re.compile(r"[【\[]\s*(\d+(?:\s*[,;]\s*\d+)*)\s*(?:†[^】\]]*)?[】\]]")


def normalize_citations(text: str) -> str:
    """Rewrite citation variants models emit (【1】, 【1†source】, [1, 3]) into canonical [1][3]."""
    return _CITE_VARIANTS.sub(lambda m: "".join(f"[{n}]" for n in re.split(r"\s*[,;]\s*", m.group(1))), text)


def chunk_id(doc_id: str, content_hash: str, chunk_size: int, chunk_overlap: int, index: int) -> str:
    """Deterministic chunk ID: same file + same chunking params -> same IDs, so re-ingesting can never duplicate."""
    key = f"{doc_id}|{content_hash}|{chunk_size}|{chunk_overlap}|{index}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class RAGService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        s = self.settings
        self.embedder = Embedder(s.embedding_model, s.embedding_query_prefix)
        self.store = VectorStore(s.database_url, self.embedder.dim)
        self._llm: LLMClient | None = None
        s.log_dir.mkdir(parents=True, exist_ok=True)
        self._query_log = s.log_dir / "queries.jsonl"

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:  # lazy: ingestion does not need an API key
            s = self.settings
            self._llm = LLMClient(s.llm_provider, s.llm_model, s.llm_api_key, s.llm_max_tokens, s.llm_temperature)
        return self._llm

    # ------------------------------------------------------------------ ingestion
    def _doc_id(self, path: Path) -> str:
        path = path.resolve()
        for base in (self.settings.corpus_dir.resolve(), PROJECT_ROOT):
            if path.is_relative_to(base):
                return path.relative_to(base).as_posix()
        return path.as_posix()

    def _doc_type(self, path: Path) -> str:
        parent = path.resolve().parent
        return "general" if parent == self.settings.corpus_dir.resolve() else parent.name

    def ingest(self, root: Path | None = None, chunk_size: int | None = None,
               chunk_overlap: int | None = None, prune: bool = False) -> dict:
        root = Path(root or self.settings.corpus_dir)
        size = chunk_size or self.settings.chunk_size
        overlap = self.settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        chunker = Chunker(self.embedder.tokenizer, size, overlap)
        start = time.perf_counter()

        summary = {"files": 0, "new": 0, "updated": 0, "unchanged": 0, "pruned": 0, "chunks_written": 0,
                   "chunk_size": size, "chunk_overlap": overlap, "embedding_model": self.embedder.model_name,
                   "embedding_dim": self.embedder.dim, "documents": []}
        seen = set()
        for path in discover(root):
            summary["files"] += 1
            doc_id = self._doc_id(path)
            seen.add(doc_id)
            existing = self.store.get_document(doc_id)
            content_hash = file_sha256(path)
            if existing and existing["content_hash"] == content_hash and existing["chunk_size"] == size \
                    and existing["chunk_overlap"] == overlap and existing["embedding_model"] == self.embedder.model_name:
                summary["unchanged"] += 1
                summary["documents"].append({"doc_id": doc_id, "status": "unchanged"})
                continue

            doc = load_document(path)
            chunks = chunker.split(doc.text)
            # The title is prepended for embedding only: it gives each chunk document-level context.
            vectors = self.embedder.embed_documents([f"{doc.title}\n{c.text}" for c in chunks])
            rows = [(chunk_id(doc_id, content_hash, size, overlap, c.index), c.index, c.text, c.token_count, v)
                    for c, v in zip(chunks, vectors)]
            self.store.replace_document(
                {"doc_id": doc_id, "title": doc.title, "doc_type": self._doc_type(path),
                 "source_format": doc.source_format, "content_hash": content_hash, "chunk_size": size,
                 "chunk_overlap": overlap, "embedding_model": self.embedder.model_name},
                rows,
            )
            status = "updated" if existing else "new"
            summary[status] += 1
            summary["chunks_written"] += len(rows)
            summary["documents"].append({"doc_id": doc_id, "status": status, "chunks": len(rows)})

        if prune and root.is_dir():
            prefix = "" if root.resolve() == self.settings.corpus_dir.resolve() else self._doc_id(root) + "/"
            for doc_id in self.store.list_doc_ids():
                if doc_id.startswith(prefix) and doc_id not in seen:
                    self.store.delete_document(doc_id)
                    summary["pruned"] += 1
                    summary["documents"].append({"doc_id": doc_id, "status": "pruned"})

        summary["seconds"] = round(time.perf_counter() - start, 2)
        summary["store"] = self.store.stats()
        log.info("ingest files=%d new=%d updated=%d unchanged=%d pruned=%d chunks_written=%d vectors_total=%d",
                 summary["files"], summary["new"], summary["updated"], summary["unchanged"], summary["pruned"],
                 summary["chunks_written"], summary["store"]["vectors"])
        return summary

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, question: str, k: int | None = None, doc_types: list[str] | None = None) -> tuple[list[Hit], dict]:
        k = k or self.settings.top_k
        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(question)
        t1 = time.perf_counter()
        hits = self.store.search(qvec, k, doc_types)
        t2 = time.perf_counter()
        return hits, {"embed_ms": (t1 - t0) * 1000, "retrieval_ms": (t2 - t1) * 1000}

    # ------------------------------------------------------------------ answering
    @staticmethod
    def _build_prompt(question: str, hits: list[Hit]) -> str:
        blocks = [f"[{i}] (source: {h.doc_id} — {h.title})\n{h.content}" for i, h in enumerate(hits, 1)]
        return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"

    def answer(self, question: str, k: int | None = None, doc_types: list[str] | None = None) -> dict:
        k = k or self.settings.top_k
        t_start = time.perf_counter()
        hits, timings = self.retrieve(question, k, doc_types)
        context = [h for h in hits if h.score >= self.settings.min_score]

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        generate_ms = 0.0
        citations: list[dict] = []
        if not context:
            answer, no_answer, reason = NO_CONTEXT_ANSWER, True, "no_chunk_above_min_score"
        else:
            result = self.llm.complete(SYSTEM_PROMPT, self._build_prompt(question, context))
            generate_ms = result.latency_ms
            usage = {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                     "total_tokens": result.total_tokens}
            if NO_CONTEXT_SENTINEL in result.text or not result.text:
                answer, no_answer, reason = NO_CONTEXT_ANSWER, True, "llm_found_context_insufficient"
            else:
                answer, no_answer, reason = normalize_citations(result.text), False, None
                cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(context)})
                citations = [{"ref": n, "chunk_id": context[n - 1].chunk_id, "doc_id": context[n - 1].doc_id,
                              "title": context[n - 1].title, "score": round(context[n - 1].score, 4)} for n in cited]

        latency = {"embed_ms": round(timings["embed_ms"], 2), "retrieval_ms": round(timings["retrieval_ms"], 2),
                   "generation_ms": round(generate_ms, 2),
                   "total_ms": round((time.perf_counter() - t_start) * 1000, 2)}
        response = {
            "question": question,
            "answer": answer,
            "no_answer": no_answer,
            "no_answer_reason": reason,
            "citations": citations,
            "retrieved": [{"rank": i, "chunk_id": h.chunk_id, "doc_id": h.doc_id, "doc_type": h.doc_type,
                           "score": round(h.score, 4), "in_context": h.score >= self.settings.min_score,
                           "text": h.content}
                          for i, h in enumerate(hits, 1)],
            "k": k,
            "doc_type_filter": doc_types,
            "chunks_in_context": len(context),
            "usage": usage,
            "latency_ms": latency,
            "model": {"llm": f"{self.settings.llm_provider}/{self.settings.llm_model}",
                      "embedding": self.embedder.model_name, "embedding_dim": self.embedder.dim},
        }
        self._log_query(response)
        return response

    def _log_query(self, r: dict) -> None:
        lat, use = r["latency_ms"], r["usage"]
        log.info("query k=%d retrieved=%d in_context=%d cited=%d no_answer=%s tokens(prompt/completion/total)=%d/%d/%d "
                 "latency_ms(embed/retrieve/generate/total)=%.1f/%.1f/%.1f/%.1f",
                 r["k"], len(r["retrieved"]), r["chunks_in_context"], len(r["citations"]), r["no_answer"],
                 use["prompt_tokens"], use["completion_tokens"], use["total_tokens"],
                 lat["embed_ms"], lat["retrieval_ms"], lat["generation_ms"], lat["total_ms"])
        record = {"ts": datetime.now(timezone.utc).isoformat(), "question": r["question"], "k": r["k"],
                  "doc_type_filter": r["doc_type_filter"], "chunks_retrieved": len(r["retrieved"]),
                  "chunks_in_context": r["chunks_in_context"], "chunks_cited": len(r["citations"]),
                  "no_answer": r["no_answer"], "usage": r["usage"], "latency_ms": r["latency_ms"]}
        with self._query_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
