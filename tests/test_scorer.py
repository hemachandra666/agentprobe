"""Unit tests for the honest scorers (P0-2, P0-4). Run in CI, no model needed.

These test externally meaningful behavior, not the old implementation:
- a right-tool-names but wrong-answer trace must FAIL task success (the 600-vs-14 case)
- partial/too-few-step runs must not look efficient
- recovery requires an error AND eventual success
"""
from __future__ import annotations
from dataclasses import dataclass
from agentprobe.trajectory import Trajectory
from agentprobe import scorer


@dataclass
class Task:
    """Minimal task with a TRUE answer, matching the new task_gen structure."""
    task_id: str
    question: str
    reference_tools: list
    min_steps: int
    answer: float


def make_traj(task_id, tool_calls, final="done"):
    t = Trajectory(task_id=task_id)
    for tool, args, result in tool_calls:
        t.add_step(tool, args, result)
    t.final_answer = final
    return t


# "add 3 and 4, then multiply by 2"  ->  true answer 14
TASK = Task("t", "add 3 and 4, then multiply by 2", ["add", "multiply"], 2, answer=14.0)


def test_correct_run_succeeds():
    traj = make_traj("t", [("add", {"a": 3, "b": 4}, "7"), ("multiply", {"a": 7, "b": 2}, "14")])
    assert scorer.task_success(traj, TASK) is True
    assert scorer.answer_correct(traj, TASK) is True
    assert scorer.tool_sequence_match(traj, TASK) == 1.0
    assert scorer.step_efficiency(traj, TASK) == 1.0
    assert scorer.error_rate(traj) == 0.0


def test_right_tools_wrong_answer_fails():
    """The counterexample from the review: add(100,200) then multiply(300,2) = 600.

    Tool names match the reference exactly, so tool_sequence_match is 1.0,
    but the answer is 600 not 14, so the task must NOT be counted a success.
    """
    traj = make_traj("t", [("add", {"a": 100, "b": 200}, "300"), ("multiply", {"a": 300, "b": 2}, "600")])
    assert scorer.tool_sequence_match(traj, TASK) == 1.0      # shape matches
    assert scorer.answer_correct(traj, TASK) is False         # but answer is wrong
    assert scorer.task_success(traj, TASK) is False           # so it fails


def test_duplicate_call_is_a_loop():
    traj = make_traj("t", [
        ("add", {"a": 3, "b": 4}, "7"),
        ("add", {"a": 3, "b": 4}, "7"),
        ("multiply", {"a": 7, "b": 2}, "14"),
    ])
    assert scorer.loop_count(traj) == 1
    assert scorer.has_loop(traj) is True


def test_error_run_is_not_success():
    traj = make_traj("t", [
        ("add", {"a": 3, "b": 4}, "7"),
        ("multiply", {"a": 3}, "error: bad argument"),
    ])
    assert scorer.error_count(traj) == 1
    assert scorer.task_success(traj, TASK) is False


def test_partial_run_is_not_efficient():
    """Too few steps to actually complete: must not report as efficient (P0-4)."""
    traj = make_traj("t", [("add", {"a": 3, "b": 4}, "7")])  # only 1 of 2 steps
    assert scorer.task_success(traj, TASK) is False
    assert scorer.step_efficiency(traj, TASK) is None


def test_recovery_requires_error_and_success():
    """Recovery: an error occurred, then the task still succeeded correctly."""
    traj = make_traj("t", [
        ("multiply", {"a": 3}, "error: bad argument"),
        ("add", {"a": 3, "b": 4}, "7"),
        ("multiply", {"a": 7, "b": 2}, "14"),
    ])
    assert scorer.recovered(traj, TASK) is True


def test_clean_run_recovery_is_not_applicable():
    """No error occurred, so recovery is N/A, not 1.0 (P0-4)."""
    traj = make_traj("t", [("add", {"a": 3, "b": 4}, "7"), ("multiply", {"a": 7, "b": 2}, "14")])
    assert scorer.recovered(traj, TASK) is None