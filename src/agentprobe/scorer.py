"""Two Week 1 scorers: tool-selection accuracy and step efficiency."""
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


def score_one(traj: Trajectory, task: Task) -> dict:
    return {
        "task_id": task.task_id,
        "tool_selection": round(tool_selection_accuracy(traj, task), 3),
        "step_efficiency": round(step_efficiency(traj, task), 3),
        "steps": traj.step_count,
        "answer": traj.final_answer,
    }