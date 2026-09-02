"""Week 4: generate clean teacher trajectories as a distillation dataset.

Runs the teacher model over the task suite, keeps only CORRECT trajectories
(right tool path, no errors), and saves them as training examples.

Usage:
  uv run python -m agentprobe.generate_teacher
  uv run python -m agentprobe.generate_teacher --runs 20
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from . import agent, scorer
from .task_suite import TASKS

TEACHER = "qwen2.5:7b"
DATASET_PATH = Path(__file__).resolve().parents[2] / "teacher_data.jsonl"


def is_clean(traj, task) -> bool:
    """A trajectory is clean if it matches the reference path exactly and had no errors."""
    return (
        scorer.trajectory_match(traj, task) == 1.0
        and scorer.error_count(traj) == 0
        and traj.step_count == task.min_steps
    )


def to_training_example(traj, task) -> dict:
    """Turn a clean trajectory into a training record: the task and the ideal tool path."""
    return {
        "task_id": task.task_id,
        "question": task.question,
        "tool_sequence": traj.tool_sequence(),
        "steps": [
            {"tool": s.tool, "args": s.args, "result": str(s.result)}
            for s in traj.steps
        ],
        "final_answer": traj.final_answer,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Generate clean teacher trajectories.")
    p.add_argument("--runs", type=int, default=20, help="runs per task (default 20)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    kept, total = [], 0
    for task in TASKS:
        clean_for_task = 0
        for _ in range(args.runs):
            total += 1
            traj = agent.run(task.task_id, task.question, model=TEACHER)
            if is_clean(traj, task):
                kept.append(to_training_example(traj, task))
                clean_for_task += 1
        print(f"{task.task_id}: kept {clean_for_task}/{args.runs} clean")

    with open(DATASET_PATH, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")

    print(f"\nTeacher: {TEACHER}")
    print(f"Total runs: {total}, clean kept: {len(kept)} ({round(100*len(kept)/total)}%)")
    print(f"Saved dataset to {DATASET_PATH.name}")


if __name__ == "__main__":
    main()