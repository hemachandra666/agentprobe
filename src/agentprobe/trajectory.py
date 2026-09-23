"""Serializable run evidence, including failures and zero-call attempts."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json, time, uuid

@dataclass
class Step:
    index: int
    tool: str
    args: object
    result: str
    status: str = "ok"
    kind: str = "tool"

@dataclass
class Trajectory:
    task_id: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    termination: str = "unknown"
    responses: list[str] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def add_step(self, tool, args, result, *, status=None, kind="tool"):
        # Prefix inference is only a compatibility path for legacy traces.
        status = status or ("error" if str(result).startswith("error:") else "ok")
        self.steps.append(Step(len(self.steps), tool, args, str(result), status, kind))

    @property
    def step_count(self):
        return len(self.steps)

    def tool_sequence(self):
        return [s.tool for s in self.steps]

    def to_dict(self):
        return {**asdict(self), "step_count": self.step_count}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
