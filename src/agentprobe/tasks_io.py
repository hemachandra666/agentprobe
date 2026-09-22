"""Load task data from an explicit directory, working data, or packaged examples."""
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import json

@dataclass
class Task:
    task_id: str
    family: str
    question: str
    reference_tools: list
    numbers: list
    min_steps: int
    answer: float

def data_root(directory=None):
    if directory is not None:
        return Path(directory)
    local = Path.cwd() / "data"
    return local if local.is_dir() else files("agentprobe").joinpath("data")

def _load(name, directory=None):
    return [Task(**json.loads(line)) for line in data_root(directory).joinpath(name).read_text().splitlines() if line.strip()]

def load_train(directory=None):
    return _load("train_tasks.jsonl", directory)

def load_test(directory=None):
    return _load("test_tasks.jsonl", directory)
