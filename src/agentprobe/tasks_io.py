"""Load the generated task splits (with true answers) from data/."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


@dataclass
class Task:
    task_id: str
    family: str
    question: str
    reference_tools: list
    numbers: list
    min_steps: int
    answer: float


def _load(path: Path) -> list[Task]:
    tasks = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            tasks.append(Task(
                task_id=d["task_id"], family=d["family"], question=d["question"],
                reference_tools=d["reference_tools"], numbers=d["numbers"],
                min_steps=d["min_steps"], answer=d["answer"],
            ))
    return tasks


def load_train() -> list[Task]:
    return _load(DATA / "train_tasks.jsonl")


def load_test() -> list[Task]:
    return _load(DATA / "test_tasks.jsonl")