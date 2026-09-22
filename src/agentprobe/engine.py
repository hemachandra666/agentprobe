"""Shared bounded action/observation loop. Never discard proposed actions."""
from __future__ import annotations
import time, queue, threading
from dataclasses import dataclass
from . import tools
from .trajectory import Trajectory

SCHEMA_VERSION = "2.0"
SYSTEM = ('You are a calculator agent. Use add(a, b) and multiply(a, b). '
          'Produce exactly ONE action per turn and wait for its real result. '
          'For text tool calls output only add(a=NUMBER, b=NUMBER) or '
          'multiply(a=NUMBER, b=NUMBER). When finished output only Answer: NUMBER. '
          'Do not invent tool results or output a full plan.')

@dataclass
class Action:
    tool: str | None
    args: dict | None
    final_answer: str | None
    raw: str
    error: str | None = None

class Provider:
    def act(self, question, history):
        raise NotImplementedError

def _execute(action):
    if not isinstance(action.tool, str) or action.tool not in tools.REGISTRY:
        return "unknown tool", "error"
    if not isinstance(action.args, dict) or set(action.args) != {"a", "b"}:
        return "expected arguments a and b", "error"
    try:
        return str(tools.REGISTRY[action.tool](**action.args)), "ok"
    except Exception as e:
        return str(e), "error"

def _act_before_deadline(provider, question, history, seconds):
    """Bound waiting; timed-out inference may continue in a daemon thread.

    Callers must stop the experiment after timeout to avoid overlapping GPU work.
    Tool execution occurs only in the main loop, never in this worker.
    """
    result = queue.Queue(maxsize=1)
    def invoke():
        try:
            result.put((True, provider.act(question, list(history))))
        except Exception as e:
            result.put((False, e))
    threading.Thread(target=invoke, daemon=True).start()
    try:
        ok, value = result.get(timeout=max(0, seconds))
    except queue.Empty:
        raise TimeoutError("provider exceeded run deadline")
    if not ok:
        raise value
    return value

def run(task_id, question, provider, max_steps=16, max_calls=8, max_seconds=60.0):
    if max_steps < 1 or max_calls < 1 or max_seconds <= 0:
        raise ValueError("budgets must be positive")
    traj = Trajectory(task_id)
    history = []
    calls = 0
    deadline = time.monotonic() + max_seconds
    traj.termination = "max_steps"
    for _ in range(max_steps):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            traj.termination = "timeout"
            break
        try:
            action = _act_before_deadline(provider, question, history, remaining)
        except TimeoutError:
            traj.termination = "timeout"
            break
        except Exception as e:
            traj.add_step("(provider)", {}, str(e), status="error", kind="provider")
            traj.termination = "provider_error"
            break
        if time.monotonic() >= deadline:
            traj.termination = "timeout"
            break
        if (not isinstance(action, Action) or not isinstance(action.raw, str)
                or (action.final_answer is not None and not isinstance(action.final_answer, str))):
            traj.add_step("(provider)", {}, "invalid action shape", status="error", kind="parse")
            traj.termination = "invalid_action_shape"
            break
        traj.responses.append(action.raw)
        if action.error or (action.tool is not None and action.final_answer is not None):
            traj.add_step("(parse)", {}, action.error or "ambiguous action", status="error", kind="parse")
            history.append(("assistant", action.raw))
            history.append(("observation", "error: output exactly one valid action"))
            continue
        if action.tool is None and action.final_answer is not None:
            traj.final_answer = action.final_answer.strip()
            traj.termination = "model_final" if traj.final_answer else "invalid_final"
            break
        if action.tool is None:
            traj.add_step("(parse)", {}, "no action", status="error", kind="parse")
            traj.termination = "invalid_action_shape"
            break
        if calls >= max_calls:
            traj.termination = "max_calls"
            break
        result, status = _execute(action)
        traj.add_step(action.tool, action.args, result, status=status)
        history.extend([("call", action.tool, action.args),
                        ("observation", result if status == "ok" else "error: " + result)])
        calls += 1
    traj.ended_at = time.time()
    return traj

def run_record(traj, task, model_label, experiment_id, score):
    return {"schema_version": SCHEMA_VERSION, "experiment_id": experiment_id,
            "model": model_label, "family": task.family, "question": task.question,
            "true_answer": task.answer, **traj.to_dict(), "num_steps": traj.step_count,
            "duration_s": round((traj.ended_at or traj.started_at) - traj.started_at, 3),
            "model_responses": traj.responses, "score": score}
