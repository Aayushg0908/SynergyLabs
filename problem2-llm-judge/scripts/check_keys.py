"""Check every provider key in .env without printing it: auth, available models, one tiny chat call.

Usage:  python scripts/check_keys.py [nvidia-model]
"""

import sys
from pathlib import Path

import httpx
from dotenv import dotenv_values

ENV = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
PROVIDERS = {
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}
WANT = ("llama", "mistral", "mixtral", "deepseek", "qwen", "gemma", "nemotron", "phi")


def check(name: str, chat_model: str | None) -> None:
    base, var = PROVIDERS[name]
    key = (ENV.get(var) or "").strip()
    print(f"\n== {name}: {var} set={bool(key)} length={len(key)} prefix={key[:4]!r}")
    if not key:
        return
    c = httpx.Client(base_url=base, headers={"Authorization": f"Bearer {key}"}, timeout=60)
    r = c.get("/models")
    models = sorted(m["id"] for m in r.json().get("data", [])) if r.is_success else []
    print(f"GET /models -> {r.status_code}, {len(models)} models")
    if name == "nvidia":
        for fam in WANT:
            hits = [m for m in models if fam in m.lower()]
            if hits:
                print(f"  {fam}: {', '.join(hits[:8])}{' …' if len(hits) > 8 else ''}")
    if chat_model:
        r = c.post("/chat/completions", json={"model": chat_model, "max_tokens": 10,
                                              "messages": [{"role": "user", "content": "Say ok"}]})
        print(f"POST /chat/completions {chat_model} -> {r.status_code}: {r.text[:200]}")
        for h, v in r.headers.items():
            if "ratelimit" in h.lower():
                print(f"  {h}: {v}")


if __name__ == "__main__":
    check("nvidia", sys.argv[1] if len(sys.argv) > 1 else None)
    check("groq", "qwen/qwen3.8-27b")
