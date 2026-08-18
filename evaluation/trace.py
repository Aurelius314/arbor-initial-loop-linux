"""Protected structured telemetry for SimpleEvol development trajectories.

The trace is diagnostic evidence only. It never participates in score
calculation, and the existing ``exp_*.txt`` and ``run.log`` files remain the
complete human-audit records.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


def _compact_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "...[truncated]"


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
        strategy_notes = [
            item["data"].get("notes", item["data"].get("summary", ""))
            for item in self.events
            if item["event"] in {"strategy_notes", "final_summary"}
        ]
        starts = [item for item in self.events if item["event"] == "trajectory_start"]
        results = [item for item in self.events if item["event"] == "evaluator_result"]
        start = starts[-1]["data"] if starts else {}
        result = results[-1]["data"] if results else {}
        worst = max(valid, key=lambda item: float(item["data"]["optimality_gap"])) if valid else None
        target_experiments = int(start.get("max_experiments", len(experiments)))
        verified_statistics = {
            "source": "structured_events",
            "target_experiments": target_experiments,
            "completed_experiments": len(experiments),
            "completion_rate": (
                len(experiments) / target_experiments if target_experiments else None
            ),
            "feasible_experiments": len(valid),
            "experiment_feasibility_rate": (
                len(valid) / len(experiments) if experiments else 0.0
            ),
            "generation_attempts": sum(
                item["event"] == "generation_attempt" for item in self.events
            ),
            "format_failures": sum(
                item["event"] == "invalid_format" for item in self.events
            ),
            "model_calls": int(result.get("calls", 0)),
            "selected_attempt_calls": int(
                result.get("selected_attempt_calls", result.get("calls", 0))
            ),
            "instance_attempts": int(result.get("instance_attempts", 1)),
            "status": str(result.get("status", "unknown")),
            "final_objective": result.get("objective"),
            "final_gap": result.get("optimality_gap"),
            "best_experiment": best["data"].get("experiment") if best else None,
            "best_objective": best["data"].get("objective") if best else None,
            "best_gap": best["data"].get("optimality_gap") if best else None,
            "worst_feasible_experiment": (
                worst["data"].get("experiment") if worst else None
            ),
            "worst_feasible_objective": worst["data"].get("objective") if worst else None,
            "worst_feasible_gap": worst["data"].get("optimality_gap") if worst else None,
            "first_feasible_experiment": (
                valid[0]["data"].get("experiment") if valid else None
            ),
            "last_feasible_experiment": (
                valid[-1]["data"].get("experiment") if valid else None
            ),
            "error_counts": dict(error_counts),
        }
        return {
            "schema_version": 4,
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
            "verified_statistics": verified_statistics,
            # Qualitative model-authored notes are retained for human audit and
            # optional inspection, but never rendered as factual Arbor evidence.
            "unverified_strategy_notes": strategy_notes[-1] if strategy_notes else "",
        }


def build_problem_digest(
    problem: str,
    rows: list[dict[str, Any]],
    *,
    invalid_gap: float = 100.0,
    baseline_gaps: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Aggregate one completed problem batch across its distinct instances."""
    if not rows:
        raise ValueError("problem digest requires at least one instance result")
    if any(row.get("problem") != problem for row in rows):
        raise ValueError("problem digest rows must belong to exactly one problem")
    instance_ids = [int(row["instance"]) for row in rows]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("problem digest received duplicate instance results")

    trajectories = [
        row["_trajectory_digest"]
        for row in rows
        if isinstance(row.get("_trajectory_digest"), dict)
    ]
    final_gaps = [row.get("gap") for row in rows]
    numeric_final_gaps = [float(gap) for gap in final_gaps if gap is not None]
    infrastructure_valid = len(numeric_final_gaps) == len(rows)
    problem_mean_gap = (
        sum(numeric_final_gaps) / len(numeric_final_gaps)
        if infrastructure_valid else None
    )
    feasible_instances = sum(row.get("objective") is not None for row in rows)

    error_counts: Counter[str] = Counter()
    for trajectory in trajectories:
        error_counts.update({
            str(error): int(count)
            for error, count in trajectory.get("error_counts", {}).items()
        })

    max_experiments = max(
        (len(trajectory.get("gap_curve", [])) for trajectory in trajectories),
        default=0,
    )
    experiment_progress = []
    for experiment in range(1, max_experiments + 1):
        best_so_far = []
        instances_attempted = 0
        instances_feasible = 0
        for trajectory in trajectories:
            curve = trajectory.get("gap_curve", [])
            if len(curve) >= experiment:
                instances_attempted += 1
            valid_prefix = [float(gap) for gap in curve[:experiment] if gap is not None]
            if valid_prefix:
                instances_feasible += 1
                best_so_far.append(min(valid_prefix))
            else:
                best_so_far.append(float(invalid_gap))
        if best_so_far:
            experiment_progress.append({
                "experiment": experiment,
                "instance_coverage": instances_attempted / len(rows),
                "feasible_so_far_rate": instances_feasible / len(rows),
                "mean_best_so_far_gap": sum(best_so_far) / len(best_so_far),
            })

    trajectory_by_instance = {
        int(trajectory["instance"]): trajectory for trajectory in trajectories
    }
    instance_trajectory_index = []
    for row in sorted(rows, key=lambda item: int(item["instance"])):
        instance_id = int(row["instance"])
        trajectory = trajectory_by_instance.get(int(row["instance"]), {})
        statistics = trajectory.get("verified_statistics", {})
        best = trajectory.get("best_experiment") or {}
        final_gap = None if row.get("gap") is None else float(row["gap"])
        baseline_gap = (
            float(baseline_gaps[instance_id])
            if baseline_gaps is not None and instance_id in baseline_gaps
            else None
        )
        gap_regression = (
            final_gap - baseline_gap
            if final_gap is not None and baseline_gap is not None else None
        )
        instance_trajectory_index.append({
            "instance": instance_id,
            "final_gap": final_gap,
            "baseline_gap": baseline_gap,
            # Positive regression means worse; positive improvement means better.
            "gap_regression_vs_baseline": gap_regression,
            "gap_improvement_vs_baseline": (
                -gap_regression if gap_regression is not None else None
            ),
            "status": str(row.get("status", "unknown")),
            "model_calls": int(statistics.get("model_calls", row.get("calls", 0))),
            "events": int(trajectory.get("events", 0)),
            "generation_attempts": int(trajectory.get("generation_attempts", 0)),
            "format_failures": int(trajectory.get("format_failures", 0)),
            "experiments": int(trajectory.get("experiments", 0)),
            "feasible_experiments": int(trajectory.get("feasible_experiments", 0)),
            "best_experiment": best.get("experiment"),
            "best_gap": best.get("optimality_gap"),
            # These are abstract, solution-free excerpts. Full instance summaries
            # and experiment events remain available through ReadDevTrace.
            "unverified_best_description": _compact_text(best.get("description"), 180),
        })
    comparable_instances = [
        item for item in instance_trajectory_index
        if item["gap_regression_vs_baseline"] is not None
    ]
    baseline_comparison_complete = len(comparable_instances) == len(rows)

    def strategy_example(item: dict[str, Any]) -> dict[str, Any]:
        trajectory = trajectory_by_instance.get(int(item["instance"]), {})
        best = trajectory.get("best_experiment") or {}
        description = str(best.get("description") or "").replace("\n", " ").strip()
        if len(description) > 400:
            description = description[:400].rstrip() + "...[truncated]"
        return {
            "instance": int(item["instance"]),
            "final_gap": item["final_gap"],
            "baseline_gap": item["baseline_gap"],
            "gap_regression_vs_baseline": item["gap_regression_vs_baseline"],
            "gap_improvement_vs_baseline": item["gap_improvement_vs_baseline"],
            "best_experiment": best.get("experiment"),
            "unverified_description": description,
        }

    strong_examples: list[dict[str, Any]] = []
    weak_examples: list[dict[str, Any]] = []
    if baseline_comparison_complete:
        strong_ranked = sorted(
            comparable_instances,
            key=lambda item: float(item["gap_improvement_vs_baseline"]),
            reverse=True,
        )
        weak_ranked = sorted(
            comparable_instances,
            key=lambda item: float(item["gap_regression_vs_baseline"]),
            reverse=True,
        )
        strong_examples = [strategy_example(item) for item in strong_ranked[:2]]
        weak_examples = [strategy_example(item) for item in weak_ranked[:2]]

    return {
        "schema_version": 4,
        "problem": problem,
        "instance_count": len(rows),
        "distinct_instance_count": len(set(instance_ids)),
        "problem_mean_gap": problem_mean_gap,
        "median_gap": median(numeric_final_gaps) if infrastructure_valid else None,
        "min_gap": min(numeric_final_gaps) if infrastructure_valid else None,
        "max_gap": max(numeric_final_gaps) if infrastructure_valid else None,
        "feasible_instances": feasible_instances,
        "feasibility_rate": feasible_instances / len(rows),
        "mean_model_calls": sum(float(row.get("calls", 0)) for row in rows) / len(rows),
        "mean_generation_attempts": (
            sum(float(item.get("generation_attempts", 0)) for item in trajectories)
            / len(trajectories) if trajectories else None
        ),
        "mean_completed_experiments": (
            sum(float(item.get("experiments", 0)) for item in trajectories)
            / len(trajectories) if trajectories else None
        ),
        "total_completed_experiments": sum(
            int(item.get("experiments", 0)) for item in trajectories
        ),
        "total_feasible_experiments": sum(
            int(item.get("feasible_experiments", 0)) for item in trajectories
        ),
        "experiment_feasibility_rate": (
            sum(int(item.get("feasible_experiments", 0)) for item in trajectories)
            / sum(int(item.get("experiments", 0)) for item in trajectories)
            if sum(int(item.get("experiments", 0)) for item in trajectories) else 0.0
        ),
        "format_failures": sum(int(item.get("format_failures", 0)) for item in trajectories),
        "status_counts": dict(Counter(str(row.get("status", "unknown")) for row in rows)),
        "error_counts": dict(error_counts),
        "baseline_comparison_complete": baseline_comparison_complete,
        "instance_trajectory_index": instance_trajectory_index,
        "experiment_progress": experiment_progress,
        "strong_strategy_examples": strong_examples,
        "weak_strategy_examples": weak_examples,
    }


def render_problem_digests(
    digests: list[dict[str, Any]], *, max_chars: int = 10000
) -> str:
    """Render compact problem-level evidence after every instance batch finishes."""
    lines = [
        "### SimpleEvol problem-level trajectory digest",
        "Each row aggregates all distinct instances completed for one problem; no solution content is included.",
    ]
    for digest in digests:
        mean_gap = digest.get("problem_mean_gap")
        rendered_gap = "invalid" if mean_gap is None else f"{float(mean_gap):.6f}"
        lines.append(
            f"- {digest['problem']}: instances={digest['distinct_instance_count']}, "
            f"mean_gap={rendered_gap}, feasibility={digest['feasibility_rate']:.2%}, "
            f"experiment_feasibility={digest.get('experiment_feasibility_rate', 0.0):.2%}, "
            f"mean_calls={digest['mean_model_calls']:.2f}, "
            f"format_failures={digest['format_failures']}"
        )
    lines.append(
        "Compact instance index legend: i=instance, g=final gap, "
        "d=current gap minus baseline gap (positive means regression), "
        "f=feasible/completed experiments, b=best experiment, x=format failures."
    )
    for digest in digests:
        index = digest.get("instance_trajectory_index", [])
        if not index:
            continue
        entries = []
        for item in index:
            gap = item.get("final_gap")
            rendered_instance_gap = "invalid" if gap is None else f"{float(gap):.3f}"
            delta = item.get("gap_regression_vs_baseline")
            rendered_delta = "na" if delta is None else f"{float(delta):+.3f}"
            entries.append(
                f"{item['instance']}:g={rendered_instance_gap},"
                f"d={rendered_delta},"
                f"f={item['feasible_experiments']}/{item['experiments']},"
                f"b={item.get('best_experiment')},x={item['format_failures']}"
            )
        lines.append(f"- {digest['problem']} instance_trajectory_index=" + ";".join(entries))
    lines.append("Additional problem-level trajectory evidence:")
    for digest in digests:
        progress = digest.get("experiment_progress", [])
        if progress:
            curve = [round(float(item["mean_best_so_far_gap"]), 6) for item in progress]
            lines.append(f"- {digest['problem']} mean_best_so_far_gap_curve={curve}")
        if digest.get("error_counts"):
            lines.append(f"  recurring_errors={digest['error_counts']}")
        examples = [
            item for item in digest.get("strong_strategy_examples", [])
            if item.get("unverified_description")
        ][:1]
        if examples:
            example = examples[0]
            lines.append(
                f"  strong_strategy_evidence=instance {example['instance']}, "
                f"gap={example['final_gap']}, "
                f"improvement_vs_baseline={example['gap_improvement_vs_baseline']}: "
                f"unverified_model_description={example['unverified_description']}"
            )
        weak = [
            item for item in digest.get("weak_strategy_examples", [])
            if item.get("unverified_description")
        ][:1]
        if weak:
            example = weak[0]
            lines.append(
                f"  weak_strategy_evidence=instance {example['instance']}, "
                f"gap={example['final_gap']}, "
                f"regression_vs_baseline={example['gap_regression_vs_baseline']}: "
                f"unverified_model_description={example['unverified_description']}"
            )
    rendered = "\n".join(lines)
    return rendered if len(rendered) <= max_chars else rendered[:max_chars] + "\n...[problem digest truncated]"


def render_digest(digests: list[dict[str, Any]], *, max_chars: int = 10000) -> str:
    """Render bounded factual telemetry for automatic Executor feedback."""
    lines = [
        "### SimpleEvol development trajectory digest",
        "Diagnostic evidence only; the protected evaluator score remains authoritative.",
    ]
    for digest in digests:
        statistics = digest.get("verified_statistics", {})
        lines.append(
            f"- {digest['problem']} instance {digest['instance']}: "
            f"{statistics.get('feasible_experiments', digest['feasible_experiments'])}/"
            f"{statistics.get('completed_experiments', digest['experiments'])} feasible, "
            f"{statistics.get('format_failures', digest['format_failures'])} format failures, "
            f"{statistics.get('generation_attempts', digest['generation_attempts'])} generation attempts"
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
            lines.append(f"  unverified_model_description={description}")
    rendered = "\n".join(lines)
    return rendered if len(rendered) <= max_chars else rendered[:max_chars] + "\n...[digest truncated]"
