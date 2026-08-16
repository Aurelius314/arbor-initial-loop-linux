"""Protected Minimum Vertex Cover validation and objective calculation."""

from __future__ import annotations

from typing import Any

OBJ_TYPE = "min"


def public_instance(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_type": "mvc",
        **{key: instance[key] for key in ("num_nodes", "instruction", "input", "instance")},
    }


def validate_solution(solution: Any, instance: dict[str, Any]) -> list[int]:
    n = int(instance["num_nodes"])
    if not isinstance(solution, list) or any(type(node) is not int for node in solution):
        raise ValueError("vertex cover must be an integer list")
    if len(set(solution)) != len(solution):
        raise ValueError("vertex cover contains duplicate vertices")
    if any(node < 0 or node >= n for node in solution):
        raise ValueError("vertex index out of range")
    cover = set(solution)
    if any(u not in cover and v not in cover for u, v in instance["instance"]["edges"]):
        raise ValueError("vertex cover leaves an edge uncovered")
    return solution


def objective(instance: dict[str, Any], vertex_cover: list[int]) -> float:
    return float(len(vertex_cover))
