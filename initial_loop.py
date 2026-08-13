"""Editable SimpleEvol harness for one Euclidean TSP instance.

Arbor is expected to evolve this file.  Dataset loading, reference objectives,
scoring and the model-call ceiling deliberately live outside this module.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable


ModelCall = Callable[..., str]


@dataclass(frozen=True)
class LoopConfig:
    max_experiments: int = 4
    compress_every: int = 2


def solve_instance(
    instance: dict[str, Any],
    call_model: ModelCall,
    config: LoopConfig = LoopConfig(),
) -> list[int]:
    """Run the harness and return its best feasible route.

    ``instance`` contains public task fields only.  In particular, the
    protected evaluator never passes the reference route/objective here.
    """
    coords = instance.get("instance")
    instruction = instance.get("instruction", "Solve this Euclidean TSP.")
    problem_input = instance.get("input") or _render_coordinates(coords)
    if not isinstance(coords, list) or not coords:
        raise ValueError("instance must contain non-empty coordinates")

    n = len(coords)
    initial = [
        {
            "role": "system",
            "content": (
                "You iteratively construct Euclidean TSP tours. Minimize the cyclic "
                "tour length. Return `Route: [..]` followed by "
                "`<description>...</description>`. Include every node exactly once; "
                "the closing edge is implicit."
            ),
        },
        {
            "role": "user",
            "content": f"### Instruction\n{instruction}\n\n### Input\n{problem_input}",
        },
    ]
    messages = list(initial)
    records: list[dict[str, Any]] = []
    best_route: list[int] | None = None
    best_cost = math.inf

    for experiment in range(1, config.max_experiments + 1):
        response = _call(call_model, messages)
        route = parse_route(response, n)
        if route is None:
            messages.append({"role": "user", "content": "Invalid format. Return a valid Route list."})
            continue

        cost = tour_cost(coords, route)
        if cost < best_cost:
            best_route, best_cost = route, cost
        records.append({"experiment": experiment, "cost": cost, "description": parse_description(response)})
        messages.extend([
            {"role": "assistant", "content": response},
            {
                "role": "user",
                "content": (
                    f"Evaluator feedback: candidate cost={cost:.6f}; "
                    f"best cost={best_cost:.6f}; best route={best_route}. "
                    "Reflect internally and return the next complete candidate."
                ),
            },
        ])

        if config.compress_every > 0 and experiment < config.max_experiments and experiment % config.compress_every == 0:
            summary = _call(call_model, [
                {"role": "system", "content": "Compress the experiment history into concise TSP search notes."},
                {"role": "user", "content": json.dumps(records, ensure_ascii=False)},
            ])
            messages = initial + [
                {"role": "assistant", "content": f"Compressed history:\n{summary}"},
                {"role": "user", "content": f"Best route={best_route}; best cost={best_cost:.6f}. Continue."},
            ]

    if best_route is None:
        raise RuntimeError("no feasible route produced")
    return best_route


def parse_route(text: str, n: int) -> list[int] | None:
    match = re.search(r"Route\s*:\s*\[([^\]]*)\]", text, re.IGNORECASE)
    if not match:
        return None
    try:
        route = [int(value.strip()) for value in match.group(1).split(",") if value.strip()]
    except ValueError:
        return None
    if len(route) == n + 1 and route[0] == route[-1]:
        route = route[:-1]
    return route if len(route) == n and set(route) == set(range(n)) else None


def parse_description(text: str) -> str:
    match = re.search(r"<description>(.*?)</description>", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def tour_cost(coords: list[list[float]], route: list[int]) -> float:
    return sum(
        math.hypot(
            coords[route[i]][0] - coords[route[(i + 1) % len(route)]][0],
            coords[route[i]][1] - coords[route[(i + 1) % len(route)]][1],
        )
        for i in range(len(route))
    )


def _render_coordinates(coords: Any) -> str:
    if not isinstance(coords, list):
        return ""
    return "\n".join(f"{i}: {point}" for i, point in enumerate(coords))


def _call(call_model: ModelCall, messages: list[dict[str, str]]) -> str:
    try:
        response = call_model(messages=messages)
    except TypeError:
        response = call_model(messages)
    return response if isinstance(response, str) else str(response)
