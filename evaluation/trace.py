"""Protected structured telemetry for SimpleEvol development trajectories.

The trace is diagnostic evidence only. It never participates in score
calculation, and the existing ``exp_*.txt`` and ``run.log`` files remain the
complete human-audit records.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _json_safe(value: Any, *, max_text: int = 6000) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_text else value[:max_text] + "...[truncated]"
    if isinstance(value, dict):
        return {str(key): _json_safe(item, max_text=max_text) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, max_text=max_text) for item in value[:200]]
    return _json_safe(str(value), max_text=max_text)


class DevTraceCollector:
    """Collect one instance trajectory in memory and append it to JSONL."""

    def __init__(self, path: Path, *, problem: str, instance: int) -> None:
        self.path = path
        self.problem = problem
        self.instance = instance
        self.events: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, payload: dict[str, Any] | None = None, **kwargs: Any) -> None:
        data = dict(payload or {})
        data.update(kwargs)
        # Raw model responses may contain complete or partial candidate solutions.
        # Never persist or expose them through the Arbor trajectory channel, even if
        # an evolved harness attempts to add this legacy field again.
        data.pop("response_excerpt", None)
        record = {
            "seq": len(self.events) + 1,
            "event": str(event),
            "problem": self.problem,
            "instance": self.instance,
            "data": _json_safe(data),
        }
        self.events.append(record)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def digest(self) -> dict[str, Any]:
        experiments = [item for item in self.events if item["event"] == "experiment_result"]
        valid = [
            item for item in experiments
            if item["data"].get("feasible") and item["data"].get("optimality_gap") is not None
        ]
        best = min(valid, key=lambda item: float(item["data"]["optimality_gap"])) if valid else None
        error_counts = Counter(
            str(item["data"].get("error") or "unknown")
            for item in experiments
            if not item["data"].get("feasible")
        )
        summaries = [
            item["data"].get("summary", "")
            for item in self.events
            if item["event"] in {"context_summary", "final_summary"}
        ]
        return {
            "problem": self.problem,
            "instance": self.instance,
            "events": len(self.events),
            "generation_attempts": sum(
                item["event"] == "generation_attempt" for item in self.events
            ),
            "format_failures": sum(
                item["event"] == "invalid_format" for item in self.events
            ),
            "experiments": len(experiments),
            "feasible_experiments": len(valid),
            "objective_curve": [item["data"].get("objective") for item in experiments],
            "gap_curve": [item["data"].get("optimality_gap") for item in experiments],
            "error_counts": dict(error_counts),
            "best_experiment": best["data"] if best else None,
            "latest_summary": summaries[-1] if summaries else "",
        }


def render_digest(digests: list[dict[str, Any]], *, max_chars: int = 10000) -> str:
    """Render bounded factual telemetry for automatic Executor feedback."""
    lines = [
        "### SimpleEvol development trajectory digest",
        "Diagnostic evidence only; the protected evaluator score remains authoritative.",
    ]
    for digest in digests:
        lines.append(
            f"- {digest['problem']} instance {digest['instance']}: "
            f"{digest['feasible_experiments']}/{digest['experiments']} feasible, "
            f"{digest['format_failures']} format failures, "
            f"{digest['generation_attempts']} generation attempts"
        )
        lines.append(f"  objective_curve={digest['objective_curve']}")
        lines.append(f"  gap_curve={digest['gap_curve']}")
        if digest["error_counts"]:
            lines.append(f"  errors={digest['error_counts']}")
        best = digest.get("best_experiment")
        if best:
            description = str(best.get("description") or "").replace("\n", " ")
            if len(description) > 800:
                description = description[:800] + "...[truncated]"
            lines.append(
                f"  best=experiment {best.get('experiment')}, "
                f"objective={best.get('objective')}, gap={best.get('optimality_gap')}"
            )
            lines.append(f"  best_description={description}")
        summary = str(digest.get("latest_summary") or "").strip()
        if summary:
            if len(summary) > 1200:
                summary = summary[:1200] + "...[truncated]"
            lines.append("  latest_summary=" + summary.replace("\n", " "))
    rendered = "\n".join(lines)
    return rendered if len(rendered) <= max_chars else rendered[:max_chars] + "\n...[digest truncated]"
