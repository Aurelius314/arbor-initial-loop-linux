"""Protected OpenAI-compatible solver backend."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


def call_openai_compatible(*, messages: list[dict[str, str]], **_: Any) -> str:
    key = os.environ.get("COP_SOLVER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("set COP_SOLVER_API_KEY or OPENROUTER_API_KEY")
    base = os.environ.get("COP_SOLVER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    model = os.environ.get("COP_SOLVER_MODEL", "deepseek/deepseek-chat")
    payload = json.dumps({"model": model, "temperature": 1.0, "messages": messages}).encode()
    request = urllib.request.Request(
        f"{base}/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.loads(response.read().decode())
    return body["choices"][0]["message"]["content"]
