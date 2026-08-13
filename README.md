# Arbor Initial Loop - minimal runnable benchmark

Arbor evolves `initial_loop.py`. The protected evaluator calls that module on
all instances in the selected split and reports mean optimality gap; lower is
better. Invalid candidates receive a 100% gap for that instance.

## Split and edit contract

- Evolution data: `data/tsp/dev.json` (2 instances).
- Held-out evaluation data: `data/tsp/test.json` (4 instances).
- Editable artifact: `initial_loop.py` only.
- Protected: `eval.py`, `evaluation/`, `data/`, and `plugins/`.
- The evaluator strips `output` before passing an instance to the harness.
- Solver calls are capped at five per instance, matching the baseline's four
  candidate calls and one intermediate compression call.

## Smoke test

The mock backend checks the full protocol without an API call:

```powershell
python eval.py --split dev --backend mock
python -m pytest -q
```

## Real baseline and Arbor

Configure the solver used inside the evaluated harness independently from the
model used by Arbor:

```powershell
$env:OPENROUTER_API_KEY = "..."
$env:TSP_SOLVER_MODEL = "deepseek/deepseek-chat"
python eval.py --split dev
arbor run "Evolve initial_loop.py to minimize mean TSP optimality gap; use dev for evolution and test only for held-out evaluation" --yes --yes-cwd .
```

Arbor itself must already be installed/configured (`arbor setup`). The project
config loads the local `initial_loop_tsp` plugin and starts with a three-cycle
review-mode smoke profile.
