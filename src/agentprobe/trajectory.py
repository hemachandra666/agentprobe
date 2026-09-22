"""Records an agent run as an ordered trajectory of steps."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json, time


@dataclass
class Step:
    index: int              # order of this step, starting at 0
    tool: str               # which tool was called
    args: dict              # the arguments passed to it
    result: str             # what the tool returned


@dataclass
class Trajectory:
    task_id: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    started_at: float = field(default_factory=time.time)

    def add_step(self, tool: str, args: dict, result: str) -> None:
        self.steps.append(Step(index=len(self.steps), tool=tool, args=args, result=str(result)))

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def tool_sequence(self) -> list[str]:
        return [s.tool for s in self.steps]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["step_count"] = self.step_count
        return d
    
    def save(self, path: str) -> None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)