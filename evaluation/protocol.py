"""Protected scoring and solver-call budget."""

from __future__ import annotations

import re
from typing import Any, Callable


class ReferenceObjectiveViolation(ValueError):
    """A feasible candidate contradicts the dataset's declared optimum."""


class CallBudget:
    def __init__(self, backend: Callable[..., str], limit: int):
        self.backend, self.limit, self.calls = backend, limit, 0

    def __call__(self, **kwargs: Any) -> str:
        if self.calls >= self.limit:
            raise RuntimeError(f"model call budget exceeded ({self.limit})")
        self.calls += 1
        return self.backend(**kwargs)


def reference_objective(instance: dict[str, Any]) -> float:
    match = re.search(
        r"(?:Objective|Makespan)\s*:\s*([-+0-9.eE]+)",
        str(instance.get("output", "")),
        re.IGNORECASE,
    )
    if not match:
        raise ValueError("reference Objective/Makespan missing")
    return float(match.group(1))


def gap_percent(candidate: float, reference: float, obj_type: str) -> float:
    tolerance = 1e-9 * max(1.0, abs(reference))
    if obj_type == "min":
        if candidate < reference - tolerance:
            raise ReferenceObjectiveViolation(
                f"min candidate objective {candidate} is better than declared optimum {reference}"
            )
        return 100.0 * (candidate - reference) / abs(reference)
    if obj_type == "max":
        if candidate > reference + tolerance:
            raise ReferenceObjectiveViolation(
                f"max candidate objective {candidate} is better than declared optimum {reference}"
            )
        return 100.0 * (reference - candidate) / abs(reference)
    raise ValueError("obj_type must be 'min' or 'max'")


def macro_average(problem_scores: dict[str, float | None]) -> float | None:
    """Equal-weight average of problem-level mean gaps."""
    if not problem_scores:
        raise ValueError("at least one problem score is required")
    if any(score is None for score in problem_scores.values()):
        return None
    valid_scores = [score for score in problem_scores.values() if score is not None]
    return sum(valid_scores) / len(valid_scores)
