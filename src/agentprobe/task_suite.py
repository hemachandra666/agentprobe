"""Three fixed tasks with known-good reference trajectories."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Task:
    task_id: str
    question: str
    reference_tools: list[str]  # the ideal tool path
    min_steps: int              # fewest steps a perfect agent needs


TASKS = [
    Task("t1", "What is 6 times 7?", ["multiply"], 1),
    Task("t2", "What is 5 plus 9?", ["add"], 1),
    Task("t3", "Add 3 and 4, then multiply the result by 2.", ["add", "multiply"], 2),
]