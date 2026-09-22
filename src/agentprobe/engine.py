"""P0-3: one shared action-observation loop for every model.

Both teacher and student run here. The loop generates ONE action, validates it,
executes the REAL tool, appends the ACTUAL observation, and continues. Invalid
actions are recorded as failed steps, never silently dropped. The model's own
final answer is recorded separately from tool results, and there are hard budgets.
"""
from __future__ import annotations
from dataclasses import dataclass
from . import tools
from .trajectory import Trajectory

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


def run(task_id: str, question: str, provider: Provider,
        max_steps: int = 8, max_calls: int = 8) -> Trajectory:
    """Shared loop. Returns a Trajectory with the model's own final answer recorded."""
    traj = Trajectory(task_id=task_id)
    history: list = []
    calls_made = 0
    termination = "max_steps"

    for _ in range(max_steps):
        if calls_made >= max_calls:
            termination = "max_calls"
            break

        action = provider.act(question, history)

        # model signalled it is done
        if action.final_answer is not None and action.tool is None:
            traj.final_answer = action.final_answer.strip()
            termination = "model_final"
            break

        # no usable action at all: record a failed step, keep going
        if action.tool is None:
            traj.add_step("(none)", {}, "error: no valid action produced")
            history.append(("assistant", action.raw))
            history.append(("tool", "error: no valid action produced"))
            calls_made += 1
            continue

        # validated tool call: execute the REAL tool, append the ACTUAL result
        result = _execute(action)
        traj.add_step(action.tool, action.args or {}, result)
        history.append(("assistant", action.raw))
        history.append(("tool", result))
        calls_made += 1

    traj.termination = termination  # type: ignore[attr-defined]
    return traj