"""Protected problem evaluators for the unified COP benchmark."""

from . import cvrp, jssp, mis, mvc, op, pfsp, tsp

PROBLEMS = {
    "cvrp": cvrp,
    "jssp": jssp,
    "mis": mis,
    "mvc": mvc,
    "op": op,
    "pfsp": pfsp,
    "tsp": tsp,
}
PROBLEM_NAMES = tuple(PROBLEMS)


def get_problem(name: str):
    try:
        return PROBLEMS[name]
    except KeyError:
        raise ValueError(f"unsupported COP problem: {name}") from None
