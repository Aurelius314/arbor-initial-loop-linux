"""Protected scoring and solver-call budget."""

from __future__ import annotations

import math
import re
from typing import Any, Callable


MAX_CALLS_PER_INSTANCE = 5  # four candidates plus compression after round 2


class CallBudget:
    def __init__(self, backend: Callable[..., str], limit: int = MAX_CALLS_PER_INSTANCE):
        self.backend, self.limit, self.calls = backend, limit, 0

    def __call__(self, **kwargs: Any) -> str:
        if self.calls >= self.limit:
            raise RuntimeError(f"model call budget exceeded ({self.limit})")
        self.calls += 1
        return self.backend(**kwargs)


def validate_route(route: Any, n: int) -> list[int]:
    if not isinstance(route, list) or len(route) != n or any(type(x) is not int for x in route):
        raise ValueError(f"route must be an integer list of length {n}")
    if set(route) != set(range(n)):
        raise ValueError("route must be a permutation of node indices")
    return route


def tour_cost(coords: list[list[float]], route: list[int]) -> float:
    return sum(math.hypot(
        coords[route[i]][0] - coords[route[(i + 1) % len(route)]][0],
        coords[route[i]][1] - coords[route[(i + 1) % len(route)]][1],
    ) for i in range(len(route)))


def reference_objective(instance: dict[str, Any]) -> float:
    match = re.search(r"Objective\s*:\s*([-+0-9.eE]+)", str(instance.get("output", "")), re.IGNORECASE)
    if not match:
        raise ValueError("reference Objective missing")
    return float(match.group(1))


def gap_percent(cost: float, reference: float) -> float:
    return 100.0 * (cost - reference) / abs(reference)
