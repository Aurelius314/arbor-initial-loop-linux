"""Protected Arbor evaluator: mean TSP optimality gap (lower is better)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from evaluation.model_client import call_openai_compatible
from evaluation.protocol import CallBudget, gap_percent, reference_objective, tour_cost, validate_route
from initial_loop import solve_instance

ROOT = Path(__file__).resolve().parent


def mock_backend(*, messages: list[dict[str, str]], **_: Any) -> str:
    """Deterministic valid-route backend for an API-free smoke test."""
    text = "\n".join(message["content"] for message in messages)
    import re
    indices = [int(x) for x in re.findall(r"(?:Node\s+|^)(\d+)(?=[,:])", text, re.MULTILINE)]
    n = max(indices) + 1 if indices else 0
    return f"Route: [{', '.join(map(str, range(n)))}]\n<description>index order</description>"


def public_instance(instance: dict[str, Any]) -> dict[str, Any]:
    return {key: instance[key] for key in ("num_nodes", "instruction", "input", "instance")}


def evaluate(instances: list[dict[str, Any]], backend: Callable[..., str], invalid_gap: float = 100.0):
    rows = []
    for index, instance in enumerate(instances):
        budget = CallBudget(backend)
        try:
            route = validate_route(solve_instance(public_instance(instance), budget), int(instance["num_nodes"]))
            objective = tour_cost(instance["instance"], route)
            gap = gap_percent(objective, reference_objective(instance))
            status = "ok"
        except Exception as exc:
            objective, gap = None, invalid_gap
            status = f"error: {type(exc).__name__}: {exc}"
        rows.append({"instance": index, "objective": objective, "gap": gap, "calls": budget.calls, "status": status})
    return sum(row["gap"] for row in rows) / len(rows), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--backend", choices=("api", "mock"), default="api")
    parser.add_argument("--invalid-gap", type=float, default=100.0)
    args = parser.parse_args()
    instances = json.loads((ROOT / "data" / "tsp" / f"{args.split}.json").read_text(encoding="utf-8"))
    score, rows = evaluate(instances, call_openai_compatible if args.backend == "api" else mock_backend, args.invalid_gap)
    for row in rows:
        print(f"instance={row['instance']} gap={row['gap']:.6f} calls={row['calls']} status={row['status']}")
    print(f"split: {args.split}")
    print(f"instances: {len(rows)}")
    print(f"mean_optimality_gap: {score:.6f}")
    print(f"score: {score:.6f}")


if __name__ == "__main__":
    main()
