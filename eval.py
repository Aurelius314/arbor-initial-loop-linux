"""Protected Arbor evaluator: mean COP optimality gap (lower is better)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from datetime import datetime

from evaluation.model_client import (
    InfrastructureAPIError,
    TransientAPIError,
    call_openai_compatible,
)
from evaluation.trace import DevTraceCollector, render_digest
from evaluation.problems import PROBLEM_NAMES, get_problem
from evaluation.protocol import CallBudget, gap_percent, macro_average, reference_objective
from initial_loop import SimpleEvol

ROOT = Path(__file__).resolve().parent


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Client:
    """Adapt the protected backend to SimpleEvol's original client interface."""

    def __init__(self, backend: Callable[..., str], limit: int):
        self.call = CallBudget(backend, limit=limit)

    def chat_completion(self, n: int, messages: list[dict[str, str]]):
        return [_Choice(self.call(messages=messages)) for _ in range(n)]


class _EvalTool:
    """Expose evaluation without placing reference labels in public instances."""

    def __init__(self, problem: Any, protected_instances: list[dict[str, Any]]):
        self.problem = problem
        self.instances = protected_instances

    def evaluate(self, solution: Any, public_instance: dict[str, Any]):
        try:
            solution = self.problem.validate_solution(solution, public_instance)
            return {"feasible": True, "obj": self.problem.objective(public_instance, solution), "error": None}
        except Exception as exc:
            return {"feasible": False, "obj": None, "error": str(exc)}

    def reference_objective(self, index: int) -> float:
        return reference_objective(self.instances[index])


def evaluate_problem(
    problem_name: str,
    instances: list[dict[str, Any]],
    backend: Callable[..., str],
    invalid_gap: float = 100.0,
    records_dir: Path | None = None,
):
    problem = get_problem(problem_name)
    rows = []
    instance_attempts = max(1, int(os.environ.get("COP_EVAL_NETWORK_ATTEMPTS", "3")))
    solver_cfg = SimpleNamespace(
        max_experiments=16,
        compress_every=5,
        max_generation_attempts=24,
    )
    compression_calls = (
        (solver_cfg.max_experiments - 1) // solver_cfg.compress_every
        if solver_cfg.compress_every > 0 else 0
    )
    # Generation attempts + scheduled compression calls + final summary.
    call_limit = solver_cfg.max_generation_attempts + compression_calls + 1
    for index, instance in enumerate(instances):
        instance_records = records_dir / f"instance_{index:03d}" if records_dir else None
        trace = DevTraceCollector(
            instance_records / "trajectory.jsonl",
            problem=problem_name,
            instance=index,
        ) if instance_records else None
        total_calls = 0
        objective = None
        failure = None
        for instance_attempt in range(1, instance_attempts + 1):
            if trace:
                trace.emit("evaluator_attempt", attempt=instance_attempt)
            # Each retry receives a fresh client and call budget so a network
            # outage cannot consume the next attempt's generation budget.
            client = _Client(backend, limit=call_limit)
            solver = SimpleEvol(
                cfg=solver_cfg,
                root_dir=ROOT,
                client=client,
                eval_dataset=[problem.public_instance(instance)],
                eval_tool=_EvalTool(problem, [instance]),
                problem_name=problem_name,
                obj_type=problem.OBJ_TYPE,
                exp_records_dir=instance_records,
                record_instance_index=index,
            )
            if trace:
                solver.trace_sink = trace.emit
            try:
                _solution, objective = solver.evolve()
                failure = None if objective is not None else "error: no feasible solution produced"
            except InfrastructureAPIError as exc:
                objective = None
                failure = f"infrastructure_error: {exc}"
                if trace:
                    trace.emit(
                        "evaluator_attempt_failed",
                        attempt=instance_attempt,
                        category="infrastructure",
                        error=str(exc),
                    )
                if instance_attempt < instance_attempts:
                    logging.getLogger(__name__).warning(
                        "Network failure on %s instance %d; restarting instance attempt %d/%d",
                        problem_name, index, instance_attempt + 1, instance_attempts,
                    )
                    time.sleep(min(2 ** (instance_attempt - 1), 8))
            except Exception as exc:
                objective = None
                failure = f"error: {type(exc).__name__}: {exc}"
                if trace:
                    trace.emit(
                        "evaluator_attempt_failed",
                        attempt=instance_attempt,
                        category="candidate_or_code",
                        error=f"{type(exc).__name__}: {exc}",
                    )
            finally:
                total_calls += client.call.calls
            if not failure or not failure.startswith("infrastructure_error:"):
                break
        gap = None if failure and failure.startswith("infrastructure_error:") else (
            invalid_gap if objective is None else gap_percent(
                objective, reference_objective(instance), problem.OBJ_TYPE
            )
        )
        if trace:
            trace.emit(
                "evaluator_result",
                objective=objective,
                optimality_gap=gap,
                calls=total_calls,
                status=failure or "ok",
            )
        row = {
            "problem": problem_name, "instance": index, "objective": objective,
            "gap": gap, "calls": total_calls, "status": failure or "ok",
        }
        if trace:
            row["_trajectory_digest"] = trace.digest()
        rows.append(row)
    if any(row["gap"] is None for row in rows):
        return None, rows
    return sum(row["gap"] for row in rows) / len(rows), rows


def evaluate_split(split: str, backend: Callable[..., str], invalid_gap: float = 100.0):
    problem_scores: dict[str, float | None] = {}
    all_rows = []
    arbor_run_id = Path(
        os.environ.get("ARBOR_RUN_ID", f"standalone_{datetime.now():%Y%m%d_%H%M%S}")
    ).name
    project_root = Path(os.environ.get("ARBOR_PROJECT_ROOT", str(ROOT))).resolve()
    explicit_event = os.environ.get("ARBOR_EVAL_EVENT", "").strip()
    node_id = os.environ.get("ARBOR_NODE_ID", "").strip()
    if explicit_event:
        event = explicit_event
    elif node_id:
        event = f"{'heldout' if split == 'test' else 'candidate'}_{node_id}"
    elif split == "test":
        event = "heldout"
    elif "ARBOR_RUN_ID" not in os.environ:
        event = "standalone"
    elif ROOT.resolve() == project_root:
        event = "baseline"
    else:
        event = "candidate"
    event = re.sub(r"[^A-Za-z0-9_.-]+", "_", event).strip("_.-") or "unknown"
    evaluation_id = f"eval_{event}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    data_root = ROOT / "data"
    available = {
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and (path / f"{split}.json").exists()
    }
    expected = set(PROBLEM_NAMES)
    if available != expected:
        missing = sorted(expected - available)
        unexpected = sorted(available - expected)
        raise ValueError(
            f"{split} must contain exactly the seven COP datasets; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for problem_name in PROBLEM_NAMES:
        problem_dir = data_root / problem_name
        data_path = problem_dir / f"{split}.json"
        instances = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(instances, list) or not instances:
            raise ValueError(f"{data_path} must contain a non-empty JSON list")
        records_dir = (
            project_root / "arbor-bin" / "experiment_records" / arbor_run_id /
            split / problem_dir.name / evaluation_id
        )
        records_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(records_dir / "run.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root_logger = logging.getLogger()
        previous_log_level = root_logger.level
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        try:
            score, rows = evaluate_problem(problem_name, instances, backend, invalid_gap, records_dir)
        finally:
            root_logger.removeHandler(file_handler)
            root_logger.setLevel(previous_log_level)
            file_handler.close()
        problem_scores[problem_name] = score
        all_rows.extend(rows)
    return macro_average(problem_scores), all_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--invalid-gap", type=float, default=100.0)
    args = parser.parse_args()
    score, rows = evaluate_split(args.split, call_openai_compatible, args.invalid_gap)
    for row in rows:
        gap = "invalid" if row["gap"] is None else f"{row['gap']:.6f}"
        print(f"problem={row['problem']} instance={row['instance']} gap={gap} calls={row['calls']} status={row['status']}")
    problem_scores = {}
    for problem_name in PROBLEM_NAMES:
        problem_gaps = [row["gap"] for row in rows if row["problem"] == problem_name]
        problem_score = (
            None
            if not problem_gaps or any(gap is None for gap in problem_gaps)
            else sum(problem_gaps) / len(problem_gaps)
        )
        problem_scores[problem_name] = problem_score
        rendered = "invalid" if problem_score is None else f"{problem_score:.6f}"
        print(f"problem_mean_optimality_gap[{problem_name}]: {rendered}")
    digests = [row["_trajectory_digest"] for row in rows if row.get("_trajectory_digest")]
    if args.split == "dev" and digests:
        print(render_digest(digests))
    print(f"split: {args.split}")
    print(f"instances: {len(rows)}")
    if score is None:
        print("evaluation_invalid: API/network/authentication failure; rerun required")
        raise SystemExit(2)
    print("macro_formula: equal-weight mean of the seven problem_mean_optimality_gap values")
    print(f"macro_mean_optimality_gap: {score:.6f}")
    print(f"mean_optimality_gap: {score:.6f}")
    print(f"score: {score:.6f}")


if __name__ == "__main__":
    main()
