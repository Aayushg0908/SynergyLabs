"""Thin provider-agnostic chat client (Groq or Gemini) that reports token usage and latency."""

import time
from dataclasses import dataclass

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    provider: str
    model: str


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in ("429", "rate limit", "resource_exhausted", "503", "unavailable", "timeout", "overloaded"))


class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str, max_tokens: int = 1024, temperature: float = 0.0):
        if not api_key:
            raise RuntimeError(f"no API key configured for {provider}")
        self.provider = provider.lower()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        if self.provider == "groq":
            from groq import Groq

            self._client = Groq(api_key=api_key, max_retries=0)
        elif self.provider == "gemini":
            from google import genai
            from google.genai import types

            # v1beta is required for system_instruction with this key type (v1 rejects it)
            self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
        else:
            raise ValueError(f"unknown provider: {provider}")

    @retry(retry=retry_if_exception(_is_transient), stop=stop_after_attempt(6),
           wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResult:
        start = time.perf_counter()
        if self.provider == "groq":
            kwargs = {}
            if self.model.startswith("openai/gpt-oss"):
                kwargs["reasoning_effort"] = "low"  # keep hidden reasoning short: lower latency + cost
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                **kwargs,
            )
            u = resp.usage
            return LLMResult(
                text=(resp.choices[0].message.content or "").strip(),
                prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens, total_tokens=u.total_tokens,
                latency_ms=(time.perf_counter() - start) * 1000, provider=self.provider, model=self.model,
            )

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json" if json_mode else None,
        )
        resp = self._client.models.generate_content(model=self.model, contents=user, config=config)
        um = resp.usage_metadata
        prompt = um.prompt_token_count or 0
        total = um.total_token_count or 0
        return LLMResult(
            text=(resp.text or "").strip(),
            prompt_tokens=prompt, completion_tokens=total - prompt, total_tokens=total,
            latency_ms=(time.perf_counter() - start) * 1000, provider=self.provider, model=self.model,
        )
