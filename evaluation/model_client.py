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
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


class InfrastructureAPIError(RuntimeError):
    """Raised when provider infrastructure or authentication invalidates an eval."""


class TransientAPIError(InfrastructureAPIError):
    """Raised when a temporary provider/network failure exhausts retries."""


class AuthenticationAPIError(InfrastructureAPIError):
    """Raised for rejected credentials; never treat it as a candidate failure."""


def _record_empty_content_response(body: dict[str, Any], diagnostics_dir: Any) -> None:
    """Save a provider response whose final content is empty for offline diagnosis."""
    if not diagnostics_dir:
        return
    try:
        target_dir = Path(diagnostics_dir) / "invalid_response"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"api_empty_content_{time.time_ns()}.json"
        target.write_text(
            json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.warning("Empty model content; full API response saved: %s", target)
    except Exception as exc:
        # Diagnostics must never change evaluation behavior.
        LOGGER.warning("Failed to save empty-content API response: %s", exc)


def _record_provider_error_response(body: dict[str, Any], diagnostics_dir: Any) -> None:
    """Persist a provider completion that ended with finish_reason=error."""
    if not diagnostics_dir:
        return
    try:
        target_dir = Path(diagnostics_dir) / "invalid_response"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"api_provider_error_{time.time_ns()}.json"
        target.write_text(
            json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.warning("Provider-error API response saved: %s", target)
    except Exception as exc:
        LOGGER.warning("Failed to save provider-error API response: %s", exc)


def call_openai_compatible(
    *,
    messages: list[dict[str, str]],
    diagnostics_dir: Any = None,
    response_mode: str = "text",
    **_: Any,
) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("set OPENROUTER_API_KEY")
    base = "https://openrouter.ai/api/v1"
    # base = os.environ.get("COP_SOLVER_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = "deepseek/deepseek-v4-flash-0731"
    # model = os.environ.get("COP_SOLVER_MODEL", "deepseek-v4-flash")
    request_body: dict[str, Any] = {
        "model": model,
        "temperature": 1.0,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Only route to endpoints that honor requested parameters such as
        # reasoning.effort="none".
        "provider": {"require_parameters": True},
    }
    if response_mode == "solution":
        # All seven COP outputs are either integer lists or lists of integer
        # lists. Problem-specific feasibility remains enforced by evaluators.
        request_body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "cop_candidate",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "solution": {
                            "anyOf": [
                                {"type": "array", "items": {"type": "integer"}},
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                    },
                                },
                            ]
                        },
                        "description": {"type": "string"},
                    },
                    "required": ["solution", "description"],
                    "additionalProperties": False,
                },
            },
        }
    elif response_mode != "text":
        raise ValueError(f"unsupported response_mode: {response_mode}")
    # This solver must produce a concise candidate, not spend the request
    # budget in an unbounded reasoning-only stream. OpenRouter uses "none" to
    # disable reasoning generation ("exclude" would only hide it).
    reasoning_effort = "none"
    request_body["reasoning"] = {"effort": reasoning_effort}
    max_tokens = os.environ.get("COP_SOLVER_MAX_TOKENS")
    if max_tokens:
        request_body["max_tokens"] = int(max_tokens)
    payload = json.dumps(request_body).encode()
    max_attempts = max(1, int(os.environ.get("COP_SOLVER_API_ATTEMPTS", "4")))
    timeout = float(os.environ.get("COP_SOLVER_API_TIMEOUT", "300"))
    wall_timeout = float(os.environ.get("COP_SOLVER_API_WALL_TIMEOUT", "300"))
    error: BaseException | None = None
    for attempt in range(max_attempts):
        started = time.monotonic()
        deadline = started + wall_timeout
        try:
            # Build a fresh request on every attempt. Some transports leave a
            # request/connection unusable after a partial response.
            request = urllib.request.Request(
                f"{base}/chat/completions", data=payload,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            reasoning_details: list[Any] = []
            finish_reason = None
            native_finish_reason = None
            provider = None
            response_model = model
            usage = None
            first_token_seconds = None
            saw_done = False
            with urllib.request.urlopen(
                request, timeout=min(timeout, wall_timeout)
            ) as response:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"solver API wall-clock timeout after {wall_timeout:.1f}s"
                        )
                    # urllib's timeout is normally per socket operation. Tighten
                    # it to the remaining wall-clock budget when possible.
                    try:
                        response.fp.raw._sock.settimeout(max(0.1, min(timeout, remaining)))
                    except (AttributeError, OSError):
                        pass
                    raw_line = response.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        saw_done = True
                        break
                    chunk = json.loads(line)
                    response_model = chunk.get("model", response_model)
                    provider = chunk.get("provider", provider)
                    if chunk.get("usage") is not None:
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or choice.get("message") or {}
                    content_piece = delta.get("content")
                    reasoning_piece = (
                        delta.get("reasoning") or delta.get("reasoning_content")
                    )
                    if isinstance(content_piece, str):
                        content_parts.append(content_piece)
                    if isinstance(reasoning_piece, str):
                        reasoning_parts.append(reasoning_piece)
                    if isinstance(delta.get("reasoning_details"), list):
                        reasoning_details.extend(delta["reasoning_details"])
                    finish_reason = choice.get("finish_reason", finish_reason)
                    native_finish_reason = choice.get(
                        "native_finish_reason", native_finish_reason
                    )
                    if first_token_seconds is None and (
                        content_piece or reasoning_piece or delta.get("reasoning_details")
                    ):
                        first_token_seconds = time.monotonic() - started
                        LOGGER.info(
                            "Solver stream first token after %.3fs (model=%s, provider=%s)",
                            first_token_seconds,
                            response_model,
                            provider,
                        )

            if not saw_done and finish_reason is None:
                raise ConnectionError("solver API stream ended before completion")
            content = "".join(content_parts)
            reasoning = "".join(reasoning_parts)
            elapsed = time.monotonic() - started
            if str(finish_reason).lower() == "error" or str(
                native_finish_reason
            ).lower() == "error":
                body = {
                    "model": response_model,
                    "provider": provider,
                    "choices": [{
                        "finish_reason": finish_reason,
                        "native_finish_reason": native_finish_reason,
                        "message": {
                            "content": content,
                            "reasoning": reasoning,
                            "reasoning_details": reasoning_details,
                        },
                    }],
                    "usage": usage,
                    "elapsed_seconds": elapsed,
                }
                _record_provider_error_response(body, diagnostics_dir)
                raise ConnectionError(
                    "provider stream completed with finish_reason=error "
                    f"(provider={provider})"
                )
            if not content.strip():
                body = {
                    "model": response_model,
                    "provider": provider,
                    "choices": [{
                        "finish_reason": finish_reason,
                        "native_finish_reason": native_finish_reason,
                        "message": {
                            "content": content,
                            "reasoning": reasoning,
                            "reasoning_details": reasoning_details,
                        },
                    }],
                    "usage": usage,
                    "elapsed_seconds": elapsed,
                }
                _record_empty_content_response(body, diagnostics_dir)
                LOGGER.warning(
                    "Model returned empty content (model=%s, provider=%s, "
                    "finish_reason=%s, native_finish_reason=%s, reasoning_chars=%d)",
                    response_model,
                    provider,
                    finish_reason,
                    native_finish_reason,
                    len(reasoning),
                )
                return ""
            return content
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise AuthenticationAPIError(
                    f"solver API authentication failed with HTTP {exc.code}"
                ) from exc
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
