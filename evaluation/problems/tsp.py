"""Protected TSP-specific validation and objective calculation."""

from __future__ import annotations

import math
from typing import Any

OBJ_TYPE = "min"


def public_instance(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_type": "tsp",
        **{key: instance[key] for key in ("num_nodes", "instruction", "input", "instance")},
    }


def validate_solution(solution: Any, instance: dict[str, Any]) -> list[int]:
    n = int(instance["num_nodes"])
    if not isinstance(solution, list) or len(solution) != n or any(type(x) is not int for x in solution):
        raise ValueError(f"route must be an integer list of length {n}")
    if set(solution) != set(range(n)):
        raise ValueError("route must be a permutation of node indices")
    return solution


def objective(instance: dict[str, Any], route: list[int]) -> float:
    coords = instance["instance"]
    return sum(math.hypot(
        coords[route[i]][0] - coords[route[(i + 1) % len(route)]][0],
        coords[route[i]][1] - coords[route[(i + 1) % len(route)]][1],
    ) for i in range(len(route)))
