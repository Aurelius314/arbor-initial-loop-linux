"""Protected problem evaluators for the unified COP benchmark."""

from . import tsp

PROBLEMS = {"tsp": tsp}


def get_problem(name: str):
    try:
        return PROBLEMS[name]
    except KeyError:
        raise ValueError(f"unsupported COP problem: {name}") from None
