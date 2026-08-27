"""AgentProbe scorers: tool selection, trajectory match, efficiency, loops, errors."""
from __future__ import annotations
from .trajectory import Trajectory
from .task_suite import Task


def tool_selection_accuracy(traj: Trajectory, task: Task) -> float:
    """Fraction of the reference tool path the agent got right, in order."""
    ref = task.reference_tools
    got = traj.tool_sequence()
    if not ref:
        return 1.0
    matches = sum(1 for i, tool in enumerate(ref) if i < len(got) and got[i] == tool)
    return matches / len(ref)


def step_efficiency(traj: Trajectory, task: Task) -> float:
    """1.0 means the agent used the minimum steps. Lower means it wandered."""
    if traj.step_count == 0:
        return 0.0
    return min(1.0, task.min_steps / traj.step_count)


def loop_count(traj: Trajectory) -> int:
    """Number of consecutive identical calls (same tool AND same args)."""
    loops = 0
    for prev, cur in zip(traj.steps, traj.steps[1:]):
        if prev.tool == cur.tool and prev.args == cur.args:
            loops += 1
    return loops


def has_loop(traj: Trajectory) -> bool:
    return loop_count(traj) > 0


def error_count(traj: Trajectory) -> int:
    """Number of steps whose result was an error."""
    return sum(1 for s in traj.steps if str(s.result).startswith("error:"))


def error_rate(traj: Trajectory) -> float:
    """Fraction of steps that returned an error. 0.0 means every call worked."""
    if traj.step_count == 0:
        return 0.0
    return round(error_count(traj) / traj.step_count, 3)


def recovered(traj: Trajectory) -> bool:
    """True if the agent hit an error but still finished with a clean final step.

    Captures the t7 pattern: failed calls, then a correct brute-force finish.
    """
    if error_count(traj) == 0:
        return False
    if not traj.steps:
        return False
    last = traj.steps[-1]
    return not str(last.result).startswith("error:")


def trajectory_match(traj: Trajectory, task: Task) -> float:
    """Rigorous path match: right tools in the right order with no wasted steps.

    1.0 means the agent's tool sequence equals the reference exactly.
    Extra, repeated, or errored steps lengthen the path and lower the score.
    """
    ref = task.reference_tools
    got = traj.tool_sequence()
    if not ref:
        return 1.0
    correct = sum(1 for i, tool in enumerate(ref) if i < len(got) and got[i] == tool)
    extra = max(0, len(got) - len(ref))
    score = (correct - extra) / len(ref)
    return round(max(0.0, min(1.0, score)), 3)


def score_one(traj: Trajectory, task: Task) -> dict:
    return {
        "task_id": task.task_id,
        "tool_selection": round(tool_selection_accuracy(traj, task), 3),
        "traj_match": trajectory_match(traj, task),
        "step_efficiency": round(step_efficiency(traj, task), 3),
        "loops": loop_count(traj),
        "error_rate": error_rate(traj),
        "recovered": recovered(traj),
        "steps": traj.step_count,
        "answer": traj.final_answer,
    }