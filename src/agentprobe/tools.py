"""Deterministic tools the agent can call. Predictable and input-tolerant.

P1-5: arguments are coerced safely (including negatives and scientific notation),
and non-finite values (inf, nan) are rejected with a clear error rather than
being treated as valid numbers.
"""
from __future__ import annotations
import math


def _num(x) -> float:
    """Coerce a tool argument to a FINITE float, or raise a clear error."""
    if isinstance(x, bool):
        # bool is a subclass of int; reject it so True/False are not treated as 1/0
        raise ValueError(f"expected a number, got bool: {x!r}")
    if isinstance(x, (int, float)):
        val = float(x)
    elif isinstance(x, str):
        try:
            val = float(x.strip())
        except ValueError:
            raise ValueError(f"not a valid number: {x!r}")
    else:
        raise ValueError(f"expected a number, got {type(x).__name__}: {x!r}")
    if not math.isfinite(val):
        raise ValueError(f"non-finite number rejected: {val!r}")
    return val


def add(a, b) -> float:
    result = _num(a) + _num(b)
    if not math.isfinite(result):
        raise ValueError("result is non-finite")
    return result


def multiply(a, b) -> float:
    result = _num(a) * _num(b)
    if not math.isfinite(result):
        raise ValueError("result is non-finite")
    return result


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