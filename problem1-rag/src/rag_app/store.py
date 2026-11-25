"""Postgres + pgvector storage: documents table (one row per file) and chunks table (one row per vector)."""

from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,          -- path relative to the corpus root
    title           TEXT NOT NULL,
    doc_type        TEXT NOT NULL,             -- metadata used for filtering
    source_format   TEXT NOT NULL,             -- pdf | html | md
    content_hash    TEXT NOT NULL,             -- sha256 of file bytes (idempotency key)
    chunk_size      INT  NOT NULL,
    chunk_overlap   INT  NOT NULL,
    embedding_model TEXT NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,              -- deterministic hash, see pipeline.chunk_id
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INT  NOT NULL,
    content     TEXT NOT NULL,
    token_count INT  NOT NULL,
    doc_type    TEXT NOT NULL,                 -- denormalised so filtering needs no join
    embedding   vector({dim}) NOT NULL,
    UNIQUE (doc_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_doc_type ON chunks (doc_type);
"""


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    doc_type: str
    chunk_index: int
    content: str
    score: float  # cosine similarity, 1 = identical


class VectorStore:
    def __init__(self, dsn: str, dim: int):
        self.dim = dim
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(_SCHEMA.format(dim=dim))
            existing = conn.execute(
                "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
            ).fetchone()[0]
            if existing != dim:
                raise RuntimeError(
                    f"chunks.embedding is vector({existing}) but the embedding model has dim {dim}; run `rag reset`"
                )
        self.pool = ConnectionPool(dsn, min_size=1, max_size=4, configure=self._configure, open=True)

    @staticmethod
    def _configure(conn: psycopg.Connection) -> None:
        register_vector(conn)
        # pgvector >= 0.8: keep scanning the HNSW graph when a metadata filter removes
        # candidates, so filtered queries still return k results.
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
        conn.commit()

    def close(self) -> None:
        self.pool.close()

    # ---------- documents ----------
    def get_document(self, doc_id: str) -> dict | None:
        with self.pool.connection() as conn:
            return conn.cursor(row_factory=dict_row).execute(
                "SELECT * FROM documents WHERE doc_id = %s", (doc_id,)
            ).fetchone()

    def list_doc_ids(self) -> list[str]:
        with self.pool.connection() as conn:
            return [r[0] for r in conn.execute("SELECT doc_id FROM documents ORDER BY doc_id")]

    def replace_document(self, doc: dict, rows: list[tuple[str, int, str, int, np.ndarray]]) -> None:
        """Atomically replace a document and all of its chunks.

        rows: (chunk_id, chunk_index, content, token_count, embedding)
        """
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc["doc_id"],))
            conn.execute(
                """
                INSERT INTO documents (doc_id, title, doc_type, source_format, content_hash,
                                       chunk_size, chunk_overlap, embedding_model, ingested_at)
                VALUES (%(doc_id)s, %(title)s, %(doc_type)s, %(source_format)s, %(content_hash)s,
                        %(chunk_size)s, %(chunk_overlap)s, %(embedding_model)s, now())
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title, doc_type = EXCLUDED.doc_type,
                    source_format = EXCLUDED.source_format, content_hash = EXCLUDED.content_hash,
                    chunk_size = EXCLUDED.chunk_size, chunk_overlap = EXCLUDED.chunk_overlap,
                    embedding_model = EXCLUDED.embedding_model, ingested_at = now()
                """,
                doc,
            )
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (chunk_id, doc_id, chunk_index, content, token_count, doc_type, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO NOTHING
                    """,
                    [(cid, doc["doc_id"], idx, text, ntok, doc["doc_type"], emb) for cid, idx, text, ntok, emb in rows],
                )

    def delete_document(self, doc_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

    # ---------- search ----------
    def search(self, query_vec: np.ndarray, k: int, doc_types: list[str] | None = None) -> list[Hit]:
        where = "WHERE doc_type = ANY(%(doc_types)s)" if doc_types else ""
        # Inner query is a plain ORDER BY distance LIMIT k on one table so the planner can
        # use the HNSW index; the join to documents happens on the k rows only.
        sql = f"""
            SELECT c.chunk_id, c.doc_id, d.title, c.doc_type, c.chunk_index, c.content,
                   1 - c.distance AS score
            FROM (
                SELECT chunk_id, doc_id, doc_type, chunk_index, content,
                       embedding <=> %(q)s AS distance
                FROM chunks {where}
                ORDER BY embedding <=> %(q)s
                LIMIT %(k)s
            ) c JOIN documents d USING (doc_id)
            ORDER BY c.distance
        """
        with self.pool.connection() as conn:
            rows = conn.cursor(row_factory=dict_row).execute(
                sql, {"q": query_vec, "k": k, "doc_types": doc_types}
            ).fetchall()
        return [Hit(**{**r, "score": float(r["score"])}) for r in rows]

    # ---------- admin ----------
    def stats(self) -> dict:
        with self.pool.connection() as conn:
            docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
            vecs = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            by_type = dict(conn.execute("SELECT doc_type, count(*) FROM chunks GROUP BY doc_type ORDER BY 1").fetchall())
            size = conn.execute("SELECT pg_size_pretty(pg_total_relation_size('chunks'))").fetchone()[0]
        return {"documents": docs, "vectors": vecs, "vectors_by_doc_type": by_type, "chunks_table_size": size, "dim": self.dim}

    @staticmethod
    def reset(dsn: str) -> None:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS documents;")
