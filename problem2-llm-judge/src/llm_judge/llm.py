"""OpenAI-compatible chat client (NVIDIA NIM, Groq) with an audit log and replay cache.

Every call — judge or generator — is appended to logs/calls.jsonl with the full messages,
the raw response text, token usage, latency and a deterministic cache key. The same file is
the replay cache: a repeated call with the same key (provider, model, messages, temperature,
json mode, sample index) returns the logged response without hitting the API, so any report
can be regenerated offline from the log (`LLM_JUDGE_REPLAY=only` refuses network calls).
`sample` distinguishes deliberate repeats (test-retest), which must not hit the cache.
"""

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import PROVIDER_URLS, get_settings

_LOCK = threading.Lock()


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    provider: str
    model: str
    cached: bool
    call_id: str
    reasoning_tokens: int = 0


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_tag: dict = field(default_factory=dict)

    def add(self, r: ChatResult, tag: str) -> None:
        with _LOCK:
            t = self.by_tag.setdefault(tag or "untagged", {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0})
            if r.cached:
                self.cached_calls += 1
                t["cached"] += 1
                return
            self.calls += 1
            self.prompt_tokens += r.prompt_tokens
            self.completion_tokens += r.completion_tokens
            t["calls"] += 1
            t["prompt_tokens"] += r.prompt_tokens
            t["completion_tokens"] += r.completion_tokens


class CallLog:
    """Append-only JSONL log that doubles as a replay cache."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("error") is None and rec.get("cache_key"):
                    self.cache[rec["cache_key"]] = rec

    def write(self, rec: dict) -> None:
        with _LOCK, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if rec.get("error") is None:
            self.cache[rec["cache_key"]] = rec


_LOGS: dict[Path, CallLog] = {}


def call_log(path: Path | None = None) -> CallLog:
    path = path or get_settings().log_dir / "calls.jsonl"
    with _LOCK:
        if path not in _LOGS:
            _LOGS[path] = CallLog(path)
        return _LOGS[path]


def _transient(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (429, 500, 502, 503, 504)


class ChatClient:
    def __init__(self, provider: str, model: str, temperature: float = 0.0, max_tokens: int = 4096,
                 role: str = "judge", usage: Usage | None = None, log: CallLog | None = None):
        if provider not in PROVIDER_URLS:
            raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PROVIDER_URLS)}")
        self.provider, self.model, self.temperature, self.max_tokens, self.role = provider, model, temperature, max_tokens, role
        self.usage = usage or Usage()
        self.log = log or call_log()
        self.replay_only = os.environ.get("LLM_JUDGE_REPLAY") == "only"
        self._http = None if self.replay_only else httpx.Client(
            base_url=PROVIDER_URLS[provider], timeout=180,
            headers={"Authorization": f"Bearer {get_settings().api_key(provider)}"})

    @property
    def name(self) -> str:
        return f"{self.provider}/{self.model}"

    def _body(self, messages: list[dict], json_mode: bool) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.provider == "groq" and self.model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"
        if self.provider == "groq" and self.model.startswith("qwen/"):
            body["reasoning_format"] = "hidden"
        return body

    def cache_key(self, messages: list[dict], json_mode: bool, sample: int) -> str:
        blob = json.dumps([self.provider, self.model, messages, self.temperature, json_mode, sample], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    @retry(retry=retry_if_exception(_transient), stop=stop_after_attempt(6),
           wait=wait_exponential(multiplier=2, min=4, max=90), reraise=True)
    def _post(self, body: dict) -> dict:
        r = self._http.post("/chat/completions", json=body)
        r.raise_for_status()
        return r.json()

    def chat(self, messages: list[dict], json_mode: bool = False, tag: str = "", sample: int = 0,
             meta: dict | None = None) -> ChatResult:
        key = self.cache_key(messages, json_mode, sample)
        hit = self.log.cache.get(key)
        if hit:
            u = hit["usage"]
            res = ChatResult(hit["raw_response"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                             hit["latency_ms"], self.provider, self.model, True, hit["call_id"],
                             u.get("reasoning_tokens", 0))
            self.usage.add(res, tag)
            return res
        if self.replay_only:
            raise RuntimeError(f"replay-only mode and no logged response for {tag} ({key[:12]})")

        body = self._body(messages, json_mode)
        call_id, start, error, data = uuid.uuid4().hex[:12], time.perf_counter(), None, {}
        try:
            data = self._post(body)
        except Exception as exc:  # logged, then re-raised to the caller
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
        latency = (time.perf_counter() - start) * 1000
        msg = (data.get("choices") or [{}])[0].get("message", {}) if data else {}
        usage = data.get("usage", {}) if data else {}
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        rec = {"call_id": call_id, "ts": datetime.now(timezone.utc).isoformat(), "role": self.role, "tag": tag,
               "meta": meta or {}, "provider": self.provider, "model": self.model, "temperature": self.temperature,
               "json_mode": json_mode, "sample": sample, "cache_key": key, "messages": messages,
               "raw_response": msg.get("content") or "", "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason") if data else None,
               "usage": {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
                         "reasoning_tokens": reasoning},
               "latency_ms": round(latency, 1), "error": error}
        self.log.write(rec)
        if error:
            raise RuntimeError(error)
        res = ChatResult(rec["raw_response"], rec["usage"]["prompt_tokens"], rec["usage"]["completion_tokens"],
                         rec["latency_ms"], self.provider, self.model, False, call_id, reasoning)
        self.usage.add(res, tag)
        return res


def usage_dict(u: Usage) -> dict:
    return asdict(u)
