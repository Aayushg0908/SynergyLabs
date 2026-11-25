"""Diagnose the Mistral API key without printing it.

Usage:  python scripts/check_mistral.py [model]
"""

import sys

import httpx
from dotenv import dotenv_values

from rag_app.config import PROJECT_ROOT

key = (dotenv_values(PROJECT_ROOT / ".env").get("JUDGE_API_KEY") or "").strip()
model = sys.argv[1] if len(sys.argv) > 1 else "mistral-small-latest"
print(f"key loaded: {bool(key)}  length={len(key)}  prefix={key[:4]!r}")

client = httpx.Client(base_url="https://api.mistral.ai/v1", headers={"Authorization": f"Bearer {key}"}, timeout=30)

r = client.get("/models")
print(f"\nGET /models -> {r.status_code}  ({len(r.json().get('data', [])) if r.is_success else 0} models)")

r = client.post("/chat/completions", json={
    "model": model, "max_tokens": 5,
    "messages": [{"role": "user", "content": "Say ok"}],
})
print(f"\nPOST /chat/completions model={model} -> {r.status_code}")
print("body:", r.text[:300])
print("rate-limit headers:")
for name, value in r.headers.items():
    if "ratelimit" in name.lower() or "retry" in name.lower():
        print(f"  {name}: {value}")
