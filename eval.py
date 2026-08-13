"""Protected Arbor evaluator: mean COP optimality gap (lower is better)."""
# 评测入口，运行 initial_loop.py 的求解器，输出mean gap

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from datetime import datetime

from evaluation.model_client import call_openai_compatible
from evaluation.problems import get_problem
from evaluation.protocol import CallBudget, gap_percent, reference_objective
from initial_loop import SimpleEvol

ROOT = Path(__file__).resolve().parent


def mock_backend(*, messages: list[dict[str, str]], **_: Any) -> str:
    """Deterministic valid-route backend for an API-free smoke test."""
    text = "\n".join(message["content"] for message in messages)
    import re
    indices = [int(x) for x in re.findall(r"(?:Node\s+|^)(\d+)(?=[,:])", text, re.MULTILINE)]
    n = max(indices) + 1 if indices else 0
    return f"Route: [{', '.join(map(str, range(n)))}]\n<description>index order</description>"


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
    public_instances = [problem.public_instance(instance) for instance in instances]
    client = _Client(backend, limit=6 * len(instances))
    solver = SimpleEvol(
        cfg=SimpleNamespace(max_experiments=4, compress_every=2),
        root_dir=ROOT,
        client=client,
        eval_dataset=public_instances,
        eval_tool=_EvalTool(problem, instances),
        problem_name=problem_name,
        obj_type=problem.OBJ_TYPE,
        exp_records_dir=records_dir,
    )
    try:
        solutions, objectives = solver.evolve()
        if len(instances) == 1:
            solutions, objectives = [solutions], [objectives]
        failure = None
    except Exception as exc:
        solutions, objectives = [None] * len(instances), [None] * len(instances)
        failure = f"error: {type(exc).__name__}: {exc}"

    rows = []
    for index, (instance, objective) in enumerate(zip(instances, objectives)):
        gap = invalid_gap if objective is None else gap_percent(
            objective, reference_objective(instance), problem.OBJ_TYPE
        )
        rows.append({
            "problem": problem_name, "instance": index, "objective": objective,
            "gap": gap, "calls": client.call.calls, "status": failure or "ok",
        })
    return sum(row["gap"] for row in rows) / len(rows), rows


def evaluate_split(split: str, backend: Callable[..., str], invalid_gap: float = 100.0):
    problem_scores, all_rows = [], []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for problem_dir in sorted(path for path in (ROOT / "data").iterdir() if path.is_dir()):
        data_path = problem_dir / f"{split}.json"
        if not data_path.exists():
            continue
        instances = json.loads(data_path.read_text(encoding="utf-8"))
        records_dir = ROOT / "experiment_records" / split / problem_dir.name / run_id
        records_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(records_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)
        try:
            score, rows = evaluate_problem(problem_dir.name, instances, backend, invalid_gap, records_dir)
        finally:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
        problem_scores.append(score)
        all_rows.extend(rows)
    if not problem_scores:
        raise ValueError(f"no data/<problem>/{split}.json datasets found")
    return sum(problem_scores) / len(problem_scores), all_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--backend", choices=("api", "mock"), default="api")
    parser.add_argument("--invalid-gap", type=float, default=100.0)
    args = parser.parse_args()
    score, rows = evaluate_split(args.split, call_openai_compatible if args.backend == "api" else mock_backend, args.invalid_gap)
    for row in rows:
        print(f"problem={row['problem']} instance={row['instance']} gap={row['gap']:.6f} calls={row['calls']} status={row['status']}")
    print(f"split: {args.split}")
    print(f"instances: {len(rows)}")
    print(f"mean_optimality_gap: {score:.6f}")
    print(f"score: {score:.6f}")


if __name__ == "__main__":
    main()
