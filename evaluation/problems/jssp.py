"""Protected JSSP validation and precedence-aware makespan calculation."""

from __future__ import annotations

from collections import deque
from typing import Any

OBJ_TYPE = "min"


def public_instance(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_type": "jssp",
        **{key: instance[key] for key in ("n", "m", "instruction", "input", "instance")},
    }


def validate_solution(solution: Any, instance: dict[str, Any]) -> list[list[int]]:
    n, m = int(instance["n"]), int(instance["m"])
    expected_jobs = set(range(n))
    if not isinstance(solution, list) or len(solution) != m:
        raise ValueError(f"schedule must contain one job order for each of {m} machines")
    for machine_order in solution:
        if (
            not isinstance(machine_order, list)
            or len(machine_order) != n
            or any(type(job) is not int for job in machine_order)
            or set(machine_order) != expected_jobs
        ):
            raise ValueError("each machine order must be a permutation of 0-indexed jobs")
    return solution


def objective(instance: dict[str, Any], schedule: list[list[int]]) -> float:
    raw = instance["instance"]
    n, m = int(instance["n"]), int(instance["m"])
    if len(raw) != n or any(len(row) != 2 * m for row in raw):
        raise ValueError("JSSP operation matrix shape is inconsistent")

    duration: dict[tuple[int, int], float] = {}
    machine_operation: dict[tuple[int, int], tuple[int, int]] = {}
    successors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    indegree: dict[tuple[int, int], int] = {}

    def add_edge(source: tuple[int, int], target: tuple[int, int]) -> None:
        successors[source].append(target)
        indegree[target] += 1

    for job, row in enumerate(raw):
        machines = [int(row[2 * operation]) for operation in range(m)]
        if set(machines) != set(range(m)):
            raise ValueError("each JSSP job must visit every machine exactly once")
        for operation, machine in enumerate(machines):
            node = (job, operation)
            duration[node] = float(row[2 * operation + 1])
            machine_operation[(machine, job)] = node
            successors[node] = []
            indegree[node] = 0
        for operation in range(m - 1):
            add_edge((job, operation), (job, operation + 1))

    for machine, job_order in enumerate(schedule):
        for before, after in zip(job_order, job_order[1:]):
            add_edge(machine_operation[(machine, before)], machine_operation[(machine, after)])

    ready = deque(node for node, degree in indegree.items() if degree == 0)
    finish = {node: duration[node] for node in ready}
    processed = 0
    while ready:
        source = ready.popleft()
        processed += 1
        for target in successors[source]:
            finish[target] = max(finish.get(target, 0.0), finish[source] + duration[target])
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if processed != n * m:
        raise ValueError("machine orders and job precedence form an infeasible cycle")
    return max(finish.values(), default=0.0)
