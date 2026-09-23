"""Validate persisted evidence before trusting it for replay or training."""
import math
from . import tools


def finite_number(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric observation")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("expected a finite numeric observation") from exc
    if not math.isfinite(number):
        raise ValueError("expected a finite numeric observation")
    return number


def validate_identity(record, task, answer_key="true_answer"):
    if record.get("task_id") != task.task_id or record.get("question") != task.question:
        raise ValueError("saved task ID/question does not match the selected dataset")
    if finite_number(record.get(answer_key)) != finite_number(task.answer):
        raise ValueError("saved reference answer does not match the selected dataset")
    if "family" in record and record["family"] != task.family:
        raise ValueError("saved task family does not match the selected dataset")


def validate_steps(steps, clean_only=False):
    if not isinstance(steps, list):
        raise ValueError("steps must be a list")
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("each step must be an object")
        status = step.get("status", "error" if str(step.get("result", "")).startswith("error:") else "ok")
        kind = step.get("kind", "tool")
        if status not in {"ok", "error"} or kind not in {"tool", "parse", "provider"}:
            raise ValueError("invalid step status or kind")
        if clean_only and (status != "ok" or kind != "tool"):
            raise ValueError("training requires clean tool steps")
        if kind != "tool":
            if status != "error":
                raise ValueError("non-tool steps must represent errors")
            continue
        if status == "error":
            continue
        args = step.get("args")
        name = step.get("tool")
        if not isinstance(name, str) or name not in tools.REGISTRY:
            raise ValueError("successful step contains an unknown tool")
        if not isinstance(args, dict) or set(args) != {"a", "b"}:
            raise ValueError("successful step requires arguments a and b")
        try:
            actual = tools.REGISTRY[name](**args)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("invalid successful tool arguments") from exc
        if not math.isclose(actual, finite_number(step.get("result")), rel_tol=0, abs_tol=1e-6):
            raise ValueError("saved tool result does not match recomputed arithmetic")
