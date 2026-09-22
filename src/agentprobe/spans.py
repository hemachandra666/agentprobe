"""Writes agent steps as span-shaped JSON records to a JSONL stream.

Note: these are span-SHAPED records (trace_id, span_id, name, attributes), not a
real OpenTelemetry integration. They are readable and auditable, nothing more.

One span per step, one JSON object per line. This is the structured,
tool-readable trace format the dashboard will consume in Week 3.
"""
from __future__ import annotations
from pathlib import Path
import json, time, uuid

SPANS_PATH = Path(__file__).resolve().parents[2] / "traces" / "spans.jsonl"


def _span(trace_id: str, task_id: str, run_index: int, step, model: str) -> dict:
    """Shape one step as a span record with standard attributes."""
    return {
        "trace_id": trace_id,
        "span_id": uuid.uuid4().hex[:16],
        "name": f"tool.{step.tool}",
        "timestamp": time.time(),
        "attributes": {
            "task_id": task_id,
            "run_index": run_index,
            "model": model,
            "step_index": step.index,
            "tool": step.tool,
            "args": step.args,
            "result": str(step.result),
            "is_error": str(step.result).startswith("error:"),
        },
    }


def write_trajectory(traj, run_index: int, model: str, path: Path = SPANS_PATH) -> str:
    """Append every step of a trajectory to the JSONL span stream.

    Returns the trace_id so runs can be grouped later.
    """
    trace_id = uuid.uuid4().hex
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for step in traj.steps:
            record = _span(trace_id, traj.task_id, run_index, step, model)
            f.write(json.dumps(record) + "\n")
    return trace_id