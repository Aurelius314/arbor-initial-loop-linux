"""Protected PFSP validation and makespan calculation."""

from __future__ import annotations

from typing import Any

OBJ_TYPE = "min"


def public_instance(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_type": "pfsp",
        **{key: instance[key] for key in ("n", "m", "instruction", "input", "instance")},
    }


def validate_solution(solution: Any, instance: dict[str, Any]) -> list[int]:
    n = int(instance["n"])
    if not isinstance(solution, list) or len(solution) != n or any(type(job) is not int for job in solution):
        raise ValueError(f"job order must be an integer list of length {n}")
    if set(solution) != set(range(1, n + 1)):
        raise ValueError("job order must be a permutation of 1-indexed jobs")
    return solution


def objective(instance: dict[str, Any], job_order: list[int]) -> float:
    processing_times = instance["instance"]
    n, m = int(instance["n"]), int(instance["m"])
    if len(processing_times) != n or any(len(row) != m for row in processing_times):
        raise ValueError("PFSP processing-time matrix shape is inconsistent")
    completion = [0.0] * m
    for job in job_order:
        for machine in range(m):
            previous_machine = completion[machine - 1] if machine else 0.0
            completion[machine] = max(completion[machine], previous_machine) + float(
                processing_times[job - 1][machine]
            )
    return completion[-1]
