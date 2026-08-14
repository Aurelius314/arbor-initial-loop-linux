"""Protected OpenAI-compatible solver backend."""

from __future__ import annotations

import json
import logging
import os
import socket
import ssl
import time
from http.client import HTTPException, RemoteDisconnected
from urllib.error import HTTPError, URLError
import urllib.request
from typing import Any


LOGGER = logging.getLogger(__name__)


class TransientAPIError(RuntimeError):
    """Raised when a temporary provider/network failure exhausts retries."""


def call_openai_compatible(*, messages: list[dict[str, str]], **_: Any) -> str:
    key = os.environ.get("COP_SOLVER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("set COP_SOLVER_API_KEY or OPENROUTER_API_KEY")
    base = os.environ.get("COP_SOLVER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    model = os.environ.get("COP_SOLVER_MODEL", "deepseek/deepseek-chat")
    payload = json.dumps({"model": model, "temperature": 1.0, "messages": messages}).encode()
    max_attempts = max(1, int(os.environ.get("COP_SOLVER_API_ATTEMPTS", "4")))
    timeout = float(os.environ.get("COP_SOLVER_API_TIMEOUT", "300"))
    error: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            # Build a fresh request on every attempt. Some transports leave a
            # request/connection unusable after a partial response.
            request = urllib.request.Request(
                f"{base}/chat/completions", data=payload,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode())
            return body["choices"][0]["message"]["content"]
        except HTTPError as exc:
            if exc.code not in {408, 409, 425, 429} and not 500 <= exc.code < 600:
                raise
            error = exc
        except (
            RemoteDisconnected,
            HTTPException,
            URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            ssl.SSLError,
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as exc:
            error = exc
        if attempt + 1 < max_attempts:
            LOGGER.warning(
                "Transient solver API failure (%s: %s); retrying attempt %d/%d",
                type(error).__name__, error, attempt + 2, max_attempts,
            )
            time.sleep(min(2 ** attempt, 8))
    assert error is not None
    raise TransientAPIError(
        f"API request failed after {max_attempts} attempts: {type(error).__name__}: {error}"
    ) from error
