from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


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
    score, rows = module.evaluate(instances, module.mock_backend)
    assert len(rows) == 2
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["calls"] == 5 for row in rows)
    assert isinstance(score, float)


def test_scoring_labels_are_not_exposed_to_harness():
    module = load_eval()
    instance = json.loads((ROOT / "data/tsp/dev.json").read_text(encoding="utf-8"))[0]
    assert set(module.public_instance(instance)) == {"num_nodes", "instruction", "input", "instance"}
    assert "output" not in module.public_instance(instance)


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
