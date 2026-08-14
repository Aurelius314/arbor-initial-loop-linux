"""API-free evaluator smoke entry point kept outside the production CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import eval as evaluator  # noqa: E402
from mock_backend import mock_backend  # noqa: E402


def main() -> None:
    instances = json.loads((PROJECT / "data" / "tsp" / "dev.json").read_text(encoding="utf-8"))
    score, _rows = evaluator.evaluate_problem("tsp", instances, mock_backend)
    print(f"score: {score:.6f}")


if __name__ == "__main__":
    main()
