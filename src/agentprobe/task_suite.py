"""Loads the task suite from tasks/suite.yaml."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Task:
    task_id: str
    question: str
    reference_tools: list[str]
    min_steps: int


SUITE_PATH = Path(__file__).resolve().parents[2] / "tasks" / "suite.yaml"


def load_tasks(path: Path = SUITE_PATH) -> list[Task]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [Task(**item) for item in raw]


TASKS = load_tasks()