"""Settings come from environment variables / the git-ignored .env file.

Judge and generator are configured independently: each has its own provider + model, and
provider API keys are looked up by provider name, so any role can use any provider.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROVIDER_URLS = {
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "groq": "https://api.groq.com/openai/v1",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    nvidia_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""

    generator_provider: str = "groq"
    generator_model: str = "openai/gpt-oss-20b"

    judge_provider: str = "nvidia"
    judge_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    judge_temperature: float = 0.0

    judge2_provider: str = "groq"
    judge2_model: str = "qwen/qwen3.8-27b"

    # Parallel judge calls. Nemotron reasons for ~25 s per call (p50), so 8 in flight is ~20 req/min,
    # under the NVIDIA free tier's ~40 req/min.
    max_concurrency: int = 8
    log_dir: Path = PROJECT_ROOT / "logs"
    results_dir: Path = PROJECT_ROOT / "results"

    def api_key(self, provider: str) -> str:
        key = getattr(self, f"{provider}_api_key", "")
        if not key:
            raise RuntimeError(f"{provider.upper()}_API_KEY is not set in .env")
        return key


@lru_cache
def get_settings() -> Settings:
    return Settings()
