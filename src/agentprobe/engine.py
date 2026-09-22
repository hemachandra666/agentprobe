"""One shared action-observation loop for every model (P0-3, hardened in P1-2).

Both teacher and student run here. The loop generates ONE action, validates it,
executes the REAL tool, appends the ACTUAL observation, and continues. Invalid
actions are recorded as failed steps, never silently dropped.

P1-2: independent budgets (max steps, max calls, wall-time). Provider failures
are caught and recorded as a failed step with a termination reason, so a crashed
model call never loses the run. Response shape is validated before use.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from . import tools

SYSTEM = ("You are a calculator agent. Use the add and multiply tools to compute "
          "step by step, then give the final number.")


@dataclass
class Action:
    """One parsed action from a model: a tool call, a final answer, or nothing."""
    tool: str | None          # tool name, or None
    args: dict | None         # args dict, or None
    final_answer: str | None  # set when the model is done
    raw: str                  # the raw text/response, for evidence


class Provider:
    """A backend must implement act(): given history, return the NEXT single Action."""
    def act(self, question: str, history: list) -> Action:
        raise NotImplementedError


def _execute(action: Action) -> str:
    """Run a real tool for a validated action, or return a typed error string."""
    fn = tools.REGISTRY.get(action.tool)
    if fn is None:
        return f"error: unknown tool {action.tool}"
    if not isinstance(action.args, dict):
        return f"error: bad arguments, expected an object"
    try:
        return str(fn(**action.args))
    except Exception as e:
        return f"error: {e}"


def _valid_action(action) -> bool:
    """A provider must return an Action with the expected shape."""
    return isinstance(action, Action)


def run(task_id: str, question: str, provider: Provider,
        max_steps: int = 8, max_calls: int = 8, max_seconds: float = 60.0):
    """Shared loop with independent budgets and provider-failure handling.

    history entries are tuples describing what happened, so each provider can
    rebuild a coherent conversation. Kinds:
      ("call", tool, args)   the model called a tool
      ("observation", result)the tool returned this
      ("assistant", text)    the model produced plain text
    """
    from .trajectory import Trajectory
    traj = Trajectory(task_id=task_id)
    history: list = []
    calls_made = 0
    termination = "max_steps"
    start = time.monotonic()

    for _ in range(max_steps):
        if calls_made >= max_calls:
            termination = "max_calls"
            break
        if time.monotonic() - start > max_seconds:
            termination = "timeout"
            break

        try:
            action = provider.act(question, history)
        except Exception as e:
            traj.add_step("(provider)", {}, f"error: provider failed: {e}")
            termination = "provider_error"
            break

        if not _valid_action(action):
            traj.add_step("(provider)", {}, "error: provider returned invalid action shape")
            termination = "invalid_action_shape"
            break

        # model signalled it is done
        if action.final_answer is not None and action.tool is None:
            traj.final_answer = (action.final_answer or "").strip()
            termination = "model_final"
            break

        # no usable action at all: record a failed step, keep going
        if action.tool is None:
            traj.add_step("(none)", {}, "error: no valid action produced")
            history.append(("assistant", action.raw or ""))
            calls_made += 1
            continue

        # validated tool call: execute the REAL tool, record the call AND the result
        result = _execute(action)
        traj.add_step(action.tool, action.args or {}, result)
        history.append(("call", action.tool, action.args or {}))
        history.append(("observation", result))
        calls_made += 1

    traj.termination = termination  # type: ignore[attr-defined]
    return traj