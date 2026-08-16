"""Protected Orienteering Problem validation and objective calculation."""

from __future__ import annotations

import math
from typing import Any

OBJ_TYPE = "max"


def public_instance(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_type": "op",
        **{key: instance[key] for key in (
            "num_nodes", "max_route_length", "start_node", "instruction", "input", "instance"
        )},
    }


def _route_length(coords: list[list[float]], route: list[int]) -> float:
    return sum(math.hypot(
        coords[route[i]][0] - coords[route[i + 1]][0],
        coords[route[i]][1] - coords[route[i + 1]][1],
    ) for i in range(len(route) - 1))


def validate_solution(solution: Any, instance: dict[str, Any]) -> list[int]:
    n = int(instance["num_nodes"])
    if not isinstance(solution, list) or not solution or any(type(node) is not int for node in solution):
        raise ValueError("route must be a non-empty integer list")
    if solution[0] != int(instance["start_node"]):
        raise ValueError("route must start at the configured start node")
    if len(set(solution)) != len(solution):
        raise ValueError("each node may be visited at most once")
    if any(node < 0 or node >= n for node in solution):
        raise ValueError("node index out of range")
    coords, prizes, max_length = instance["instance"]
    if len(coords) != n or len(prizes) != n:
        raise ValueError("OP instance size is inconsistent")
    if _route_length(coords, solution) > float(max_length) + 1e-9:
        raise ValueError("maximum route length exceeded")
    return solution


def objective(instance: dict[str, Any], route: list[int]) -> float:
    prizes = instance["instance"][1]
    return float(sum(float(prizes[node]) for node in route))
