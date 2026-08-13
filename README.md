# Arbor Initial Loop - unified COP benchmark

Arbor evolves `initial_loop.py`. The protected evaluator discovers every
`data/<problem>/<split>.json`, evaluates it through
`evaluation/problems/<problem>.py`, and reports the problem-balanced mean
optimality gap; lower is better.

## Split and edit contract

- Evolution data: `data/<problem>/dev.json`.
- Held-out evaluation data: `data/<problem>/test.json`.
- Editable artifact: `initial_loop.py` only.
- Protected: `eval.py`, `evaluation/`, `data/`, and `plugins/`.
- The evaluator strips `output` before passing an instance to the harness.
- Solver calls are capped at six per instance, matching the baseline's four
  candidate calls, one intermediate compression call, and one final summary.
- Local traces are retained under
  `experiment_records/<split>/<problem>/<timestamp>/`; they are git-ignored
  and do not participate in scoring.

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
