"""Unit tests for the scorers. These run in CI with no model needed."""
from __future__ import annotations
from agentprobe.trajectory import Trajectory
from agentprobe.task_suite import Task
from agentprobe import scorer


def make_traj(task_id, tool_calls, final="done"):
    """Build a trajectory from a list of (tool, args, result) tuples."""
    t = Trajectory(task_id=task_id)
    for tool, args, result in tool_calls:
        t.add_step(tool, args, result)
    t.final_answer = final
    return t


TASK = Task("t", "add 3 and 4, then multiply by 2", ["add", "multiply"], 2)


def test_perfect_path_scores_one():
    traj = make_traj("t", [("add", {"a": 3, "b": 4}, "7"), ("multiply", {"a": 7, "b": 2}, "14")])
    assert scorer.trajectory_match(traj, TASK) == 1.0
    assert scorer.step_efficiency(traj, TASK) == 1.0
    assert scorer.loop_count(traj) == 0
    assert scorer.error_rate(traj) == 0.0


def test_duplicate_call_is_a_loop():
    traj = make_traj("t", [
        ("add", {"a": 3, "b": 4}, "7"),
        ("add", {"a": 3, "b": 4}, "7"),
        ("multiply", {"a": 7, "b": 2}, "14"),
    ])
    assert scorer.loop_count(traj) == 1
    assert scorer.has_loop(traj) is True


def test_extra_steps_lower_trajectory_match():
    traj = make_traj("t", [
        ("add", {"a": 3, "b": 4}, "7"),
        ("add", {"a": 3, "b": 4}, "7"),
        ("multiply", {"a": 7, "b": 2}, "14"),
    ])
    assert scorer.trajectory_match(traj, TASK) < 1.0


def test_error_steps_counted():
    traj = make_traj("t", [
        ("add", {"a": 3, "b": 4}, "7"),
        ("multiply", {"a": 3}, "error: bad argument"),
    ])
    assert scorer.error_count(traj) == 1
    assert scorer.error_rate(traj) == 0.5


def test_recovery_detected():
    traj = make_traj("t", [
        ("multiply", {"a": 3}, "error: bad argument"),
        ("add", {"a": 3, "b": 4}, "7"),
        ("multiply", {"a": 7, "b": 2}, "14"),
    ])
    assert scorer.recovered(traj) is True