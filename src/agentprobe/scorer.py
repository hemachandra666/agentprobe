"""AgentProbe scorers.

P0-2 fix: the old trajectory_match only checked tool NAMES in order. It could
score add(100,200) then multiply(300,2) as a perfect 1.0 for "add 3 and 4, then
multiply by 2", even though the answer is 600 not 14. That metric is retained
but renamed tool_sequence_match, so it is never mistaken for correctness.
New metrics actually validate the task: final-answer correctness and task success.
"""
from __future__ import annotations
from .trajectory import Trajectory

# A "task" here is anything with .reference_tools, .min_steps, and .answer.
# The new generated tasks (task_gen.py) carry a true numeric .answer.


def _true_answer(task):
    """The task's correct final numeric answer, or None if unknown."""
    return getattr(task, "answer", None)


def tool_sequence_match(traj: Trajectory, task) -> float:
    """SHAPE ONLY: does the tool-name sequence match the reference, in order.

    This does NOT verify arguments or correctness. Renamed from the old
    'trajectory_match' precisely so it is not read as a correctness score.
    """
    ref = task.reference_tools
    got = traj.tool_sequence()
    if not ref:
        return 1.0
    correct = sum(1 for i, tool in enumerate(ref) if i < len(got) and got[i] == tool)
    extra = max(0, len(got) - len(ref))
    score = (correct - extra) / len(ref)
    return round(max(0.0, min(1.0, score)), 3)


def _final_numeric(traj: Trajectory):
    """The last successful tool result as a number, or None."""
    for s in reversed(traj.steps):
        r = str(s.result)
        if r.startswith("error:"):
            continue
        try:
            return float(r)
        except ValueError:
            continue
    return None


def answer_correct(traj: Trajectory, task) -> bool:
    """Did the agent's actual computed result equal the task's true answer?

    This is the check the old scorer lacked. It uses the last real tool result,
    which is what the tools actually computed, not the model's text.
    """
    true = _true_answer(task)
    if true is None:
        return False
    got = _final_numeric(traj)
    if got is None:
        return False
    return abs(got - true) < 1e-6


def task_success(traj: Trajectory, task) -> bool:
    """A task succeeds if it produced the correct final answer.

    A mid-run error is allowed IF the agent recovered and still reached the
    right answer (that recovered case is reported separately). What is never
    allowed: a wrong final answer (the 600-vs-14 case) or an empty run, or a
    run whose LAST step is still an error (it never recovered).
    """
    if traj.step_count == 0:
        return False
    if str(traj.steps[-1].result).startswith("error:"):
        return False  # ended on an error, did not recover
    return answer_correct(traj, task)


def step_efficiency(traj: Trajectory, task) -> float:
    """Steps vs minimum, but only meaningful for SUCCESSFUL tasks.

    Returns None for unsuccessful tasks so partial runs cannot look efficient
    (P0-4). Fewer-than-required steps no longer scores as fully efficient.
    """
    if not task_success(traj, task):
        return None
    if traj.step_count < task.min_steps:
        return None  # cannot succeed in fewer than the minimum, treat as invalid
    return round(min(1.0, task.min_steps / traj.step_count), 3)


def loop_count(traj: Trajectory) -> int:
    """Consecutive identical calls (same tool AND same args)."""
    loops = 0
    for prev, cur in zip(traj.steps, traj.steps[1:]):
        if prev.tool == cur.tool and prev.args == cur.args:
            loops += 1
    return loops


def has_loop(traj: Trajectory) -> bool:
    return loop_count(traj) > 0


def error_count(traj: Trajectory) -> int:
    return sum(1 for s in traj.steps if str(s.result).startswith("error:"))


def error_rate(traj: Trajectory) -> float:
    if traj.step_count == 0:
        return 0.0
    return round(error_count(traj) / traj.step_count, 3)


def recovered(traj: Trajectory, task):
    """Recovery requires: an error occurred AND the task still succeeded (P0-4).

    Returns None when no error was exercised, so clean runs do not inflate
    recovery to 1.0. The old version only checked the last step lacked 'error:'.
    """
    if error_count(traj) == 0:
        return None  # recovery not exercised
    return task_success(traj, task)


def score_one(traj: Trajectory, task) -> dict:
    return {
        "task_id": task.task_id,
        "task_success": task_success(traj, task),
        "answer_correct": answer_correct(traj, task),
        "tool_sequence_match": tool_sequence_match(traj, task),
        "step_efficiency": step_efficiency(traj, task),
        "loops": loop_count(traj),
        "error_rate": error_rate(traj),
        "recovered": recovered(traj, task),
        "steps": traj.step_count,
        "final_answer": traj.final_answer,
    }