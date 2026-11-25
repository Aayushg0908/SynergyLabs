"""HTTP API.  Run:  uvicorn rag_app.api:app --port 8000"""

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .pipeline import RAGService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
for _noisy in ("httpx", "httpx2", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


@lru_cache
def service() -> RAGService:
    return RAGService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service().llm  # load the embedding model, DB pool and LLM client once at startup
    yield
    service().store.close()


app = FastAPI(title="Kestrel RAG", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=50, description="top-k chunks to retrieve (default TOP_K)")
    doc_type: list[str] | None = Field(default=None, description="metadata filter, e.g. ['policies']")


class IngestRequest(BaseModel):
    path: str | None = Field(default=None, description="file or folder; default CORPUS_DIR")
    chunk_size: int | None = Field(default=None, ge=32, le=512)
    chunk_overlap: int | None = Field(default=None, ge=0, le=256)
    prune: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    return service().store.stats()


@app.post("/query")
def query(req: QueryRequest) -> dict:
    return service().answer(req.question, req.k, req.doc_type)


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    return service().ingest(Path(req.path) if req.path else None, req.chunk_size, req.chunk_overlap, req.prune)
