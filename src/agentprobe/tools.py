"""Deterministic tools the agent can call. Predictable and input-tolerant."""
from __future__ import annotations


def _num(x):
    """Coerce a tool argument to a float, or raise a clear error."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise ValueError(f"expected a number, got {type(x).__name__}: {x!r}")


def add(a, b) -> float:
    return _num(a) + _num(b)


def multiply(a, b) -> float:
    return _num(a) * _num(b)


REGISTRY = {"add": add, "multiply": multiply}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers a and b.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers a and b.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
]