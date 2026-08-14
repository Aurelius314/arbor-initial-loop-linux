from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from mock_backend import mock_backend


ROOT = Path(__file__).resolve().parents[1]


def load_eval():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("initial_loop_eval", ROOT / "eval.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mock_dev_evaluation_is_finite_and_valid():
    module = load_eval()
    instances = json.loads((ROOT / "data/tsp/dev.json").read_text(encoding="utf-8"))
    score, rows = module.evaluate_problem("tsp", instances, mock_backend)
    assert len(rows) == 2
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["calls"] == 6 for row in rows)
    assert isinstance(score, float)


def test_scoring_labels_are_not_exposed_to_harness():
    module = load_eval()
    instance = json.loads((ROOT / "data/tsp/dev.json").read_text(encoding="utf-8"))[0]
    public = module.get_problem("tsp").public_instance(instance)
    assert set(public) == {"problem_type", "num_nodes", "instruction", "input", "instance"}
    assert "output" not in public


def test_call_budget_is_enforced():
    module = load_eval()
    budget = module.CallBudget(lambda **_: "ok", limit=1)
    assert budget(messages=[]) == "ok"
    try:
        budget(messages=[])
    except RuntimeError:
        pass
    else:
        raise AssertionError("budget accepted an extra call")


def test_gap_respects_objective_direction():
    module = load_eval()
    assert module.gap_percent(110.0, 100.0, "min") == 10.0
    assert module.gap_percent(90.0, 100.0, "max") == 10.0


def test_tsp_accepts_explicit_closed_route():
    module = load_eval()
    problem = module.get_problem("tsp")
    instance = {"num_nodes": 3, "instance": [[0, 0], [1, 0], [0, 1]]}
    assert problem.validate_solution([0, 1, 2, 0], instance) == [0, 1, 2]


def test_one_instance_failure_does_not_poison_the_next():
    module = load_eval()
    instances = json.loads((ROOT / "data/tsp/dev.json").read_text(encoding="utf-8"))
    calls = 0

    def backend(*, messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 14:
            return "invalid"
        return mock_backend(messages=messages, **kwargs)

    _score, rows = module.evaluate_problem("tsp", instances, backend)
    assert rows[0]["status"].startswith("error:")
    assert rows[1]["status"] == "ok"


def test_transient_api_failure_invalidates_evaluation_instead_of_scoring_100():
    module = load_eval()
    instances = json.loads((ROOT / "data/tsp/dev.json").read_text(encoding="utf-8"))[:1]

    def backend(**_kwargs):
        raise module.TransientAPIError("connection closed")

    score, rows = module.evaluate_problem("tsp", instances, backend)
    assert score is None
    assert rows[0]["gap"] is None
    assert rows[0]["status"].startswith("infrastructure_error:")


def test_public_evaluator_has_no_mock_backend_switch():
    source = (ROOT / "eval.py").read_text(encoding="utf-8")
    assert "--backend" not in source
    assert "mock_backend" not in source
