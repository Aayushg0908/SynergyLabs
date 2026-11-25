"""Local sentence-transformers embedder (no API cost, runs on CPU)."""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, query_prefix: str = ""):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name, device="cpu")
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim: int = get_dim()

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True)[0]
