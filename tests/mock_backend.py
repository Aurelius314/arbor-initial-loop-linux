"""Test-only deterministic backend; never exposed by the protected evaluator CLI."""

from __future__ import annotations

import re
from typing import Any


def mock_backend(*, messages: list[dict[str, str]], **_: Any) -> str:
    text = "\n".join(message["content"] for message in messages)
    indices = [int(x) for x in re.findall(r"(?:Node\s+|^)(\d+)(?=[,:])", text, re.MULTILINE)]
    n = max(indices) + 1 if indices else 0
    return f"Route: [{', '.join(map(str, range(n)))}]\n<description>index order</description>"
