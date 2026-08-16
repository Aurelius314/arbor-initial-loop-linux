"""Protected Maximum Independent Set validation and objective calculation."""

from __future__ import annotations

from typing import Any

OBJ_TYPE = "max"


def public_instance(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_type": "mis",
        **{key: instance[key] for key in ("num_nodes", "instruction", "input", "instance")},
    }


def validate_solution(solution: Any, instance: dict[str, Any]) -> list[int]:
    n = int(instance["num_nodes"])
    if not isinstance(solution, list) or any(type(node) is not int for node in solution):
        raise ValueError("independent set must be an integer list")
    if len(set(solution)) != len(solution):
        raise ValueError("independent set contains duplicate vertices")
    if any(node < 0 or node >= n for node in solution):
        raise ValueError("vertex index out of range")
    chosen = set(solution)
    if any(u in chosen and v in chosen for u, v in instance["instance"]["edges"]):
        raise ValueError("selected vertices are not independent")
    return solution


def objective(instance: dict[str, Any], independent_set: list[int]) -> float:
    return float(len(independent_set))
