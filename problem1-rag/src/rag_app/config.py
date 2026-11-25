"""All configuration comes from environment variables (or a local, git-ignored .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Generator LLM
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_key: str = ""
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0

    # Judge LLM (used by the evaluation harness only)
    judge_provider: str = "gemini"
    judge_model: str = "gemini-2.5-flash"
    judge_api_key: str = ""

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge-*-v1.5 models expect this instruction on queries (not on passages)
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Vector store
    database_url: str = "postgresql://rag:rag@localhost:5544/rag"

    # Chunking (token counts, measured with the embedding model's tokenizer)
    chunk_size: int = 256
    chunk_overlap: int = 32

    # Retrieval
    top_k: int = 5
    # Chunks below this cosine similarity are treated as irrelevant. If none pass,
    # the service answers "no relevant context" without calling the LLM.
    min_score: float = 0.5

    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    log_dir: Path = PROJECT_ROOT / "logs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
