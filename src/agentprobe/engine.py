"""One shared action-observation loop for every model (P0-3, hardened in P1-2, P1-3).

Both teacher and student run here. The loop generates ONE action, validates it,
executes the REAL tool, appends the ACTUAL observation, and continues. Invalid
actions are recorded as failed steps, never silently dropped.

P1-2: independent budgets (max steps, max calls, wall-time). Provider failures
are caught and recorded with a termination reason, so a crashed call never loses
the run. Response shape is validated before use.

P1-3: every run carries timing, a termination reason, and produces a complete
record (via run_record) even when it made zero tool calls or failed.
"""
from __future__ import annotations
import time, uuid
from dataclasses import dataclass
from . import tools

SCHEMA_VERSION = "1.0"

SYSTEM = ("You are a calculator agent. Use the add and multiply tools to compute "
          "step by step, then give the final number.")


@dataclass
class Action:
    """One parsed action from a model: a tool call, a final answer, or nothing."""
    tool: str | None
    args: dict | None
    final_answer: str | None
    raw: str


class Provider:
    """A backend must implement act(): given history, return the NEXT single Action."""
    def act(self, question: str, history: list) -> Action:
        raise NotImplementedError


def _execute(action: Action) -> str:
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
    return isinstance(action, Action)


def run(task_id: str, question: str, provider: Provider,
        max_steps: int = 8, max_calls: int = 8, max_seconds: float = 60.0):
    """Shared loop. Records timing, termination, and per-turn model text for P1-3.

    history entries are tuples describing what happened:
      ("call", tool, args)    the model called a tool
      ("observation", result) the tool returned this
      ("assistant", text)     the model produced plain text
    """
    from .trajectory import Trajectory
    traj = Trajectory(task_id=task_id)
    history: list = []
    responses: list = []          # every raw model response, for evidence (P1-3)
    calls_made = 0
    termination = "max_steps"
    start = time.time()
    start_mono = time.monotonic()

    for _ in range(max_steps):
        if calls_made >= max_calls:
            termination = "max_calls"
            break
        if time.monotonic() - start_mono > max_seconds:
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

        responses.append(action.raw)

        if action.final_answer is not None and action.tool is None:
            traj.final_answer = (action.final_answer or "").strip()
            termination = "model_final"
            break

        if action.tool is None:
            traj.add_step("(none)", {}, "error: no valid action produced")
            history.append(("assistant", action.raw or ""))
            calls_made += 1
            continue

        result = _execute(action)
        traj.add_step(action.tool, action.args or {}, result)
        history.append(("call", action.tool, action.args or {}))
        history.append(("observation", result))
        calls_made += 1

    end = time.time()
    traj.termination = termination                 # type: ignore[attr-defined]
    traj.responses = responses                     # type: ignore[attr-defined]
    traj.started_at = start                        # type: ignore[attr-defined]
    traj.ended_at = end                            # type: ignore[attr-defined]
    traj.run_id = uuid.uuid4().hex[:12]            # type: ignore[attr-defined]
    return traj


def run_record(traj, task, model_label, experiment_id, score):
    """A complete, saveable record for ONE attempted run (P1-3).

    Exists for EVERY attempt, including zero-call and failed runs.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_id": getattr(traj, "run_id", None),
        "model": model_label,
        "task_id": task.task_id,
        "family": task.family,
        "question": task.question,
        "true_answer": task.answer,
        "started_at": getattr(traj, "started_at", None),
        "ended_at": getattr(traj, "ended_at", None),
        "duration_s": round(getattr(traj, "ended_at", 0) - getattr(traj, "started_at", 0), 3),
        "termination": getattr(traj, "termination", None),
        "num_steps": traj.step_count,
        "steps": [{"tool": s.tool, "args": s.args, "result": str(s.result)} for s in traj.steps],
        "model_responses": getattr(traj, "responses", []),
        "final_answer": traj.final_answer,
        "score": score,
    }