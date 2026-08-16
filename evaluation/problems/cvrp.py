"""Protected CVRP validation and objective calculation."""

from __future__ import annotations

import math
from typing import Any

OBJ_TYPE = "min"


def public_instance(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_type": "cvrp",
        **{key: instance[key] for key in (
            "num_nodes", "vehicle_capacity", "instruction", "input", "instance"
        )},
    }


def validate_solution(solution: Any, instance: dict[str, Any]) -> list[list[int]]:
    n = int(instance["num_nodes"])
    coords, demands, capacity = instance["instance"]
    if len(coords) != n or len(demands) != n:
        raise ValueError("CVRP instance size is inconsistent")
    if not isinstance(solution, list) or not solution:
        raise ValueError("routes must be a non-empty list of routes")

    customers: list[int] = []
    for route in solution:
        if not isinstance(route, list) or len(route) < 2:
            raise ValueError("each route must contain a start and end depot")
        if any(type(node) is not int for node in route):
            raise ValueError("route nodes must be integers")
        if route[0] != 0 or route[-1] != 0:
            raise ValueError("each route must start and end at depot 0")
        if any(node <= 0 or node >= n for node in route[1:-1]):
            raise ValueError("customer index out of range or depot used inside route")
        if sum(float(demands[node]) for node in route[1:-1]) > float(capacity) + 1e-9:
            raise ValueError("vehicle capacity exceeded")
        customers.extend(route[1:-1])

    if len(customers) != n - 1 or set(customers) != set(range(1, n)):
        raise ValueError("every customer must be visited exactly once")
    return solution


def objective(instance: dict[str, Any], routes: list[list[int]]) -> float:
    coords = instance["instance"][0]
    return sum(
        math.hypot(
            coords[route[i]][0] - coords[route[i + 1]][0],
            coords[route[i]][1] - coords[route[i + 1]][1],
        )
        for route in routes
        for i in range(len(route) - 1)
    )
