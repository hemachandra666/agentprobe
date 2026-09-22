"""Generate honest teacher trajectories as a distillation dataset.

P0-2 fix: acceptance is gated on real task_success (correct final answer via the
shared engine), not on tool-name order. Runs the teacher through the SAME loop
(engine) both models use. Only correct, completed trajectories are kept.

Usage:
  uv run --no-sync python -m agentprobe.generate_teacher --runs 5
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from . import agent, scorer
from .tasks_io import load_train

TEACHER = "qwen2.5:7b"
DATASET_PATH = Path(__file__).resolve().parents[2] / "teacher_data.jsonl"


def to_training_example(traj, task) -> dict:
    return {
        "task_id": task.task_id,
        "family": task.family,
        "question": task.question,
        "answer": task.answer,
        "steps": [
            {"tool": s.tool, "args": s.args, "result": str(s.result)}
            for s in traj.steps
        ],
        "final_answer": traj.final_answer,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Generate honest teacher trajectories.")
    p.add_argument("--runs", type=int, default=3, help="runs per task (default 3)")
    p.add_argument("--max-tasks", type=int, default=200,
                   help="cap number of train tasks to sample (default 200)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_train()
    rng = random.Random(42)
    rng.shuffle(tasks)
    tasks = tasks[:args.max_tasks]

    kept, total, success = [], 0, 0
    for i, task in enumerate(tasks, 1):
        for _ in range(args.runs):
            total += 1
            traj = agent.run(task.task_id, task.question, model=TEACHER)
            if scorer.task_success(traj, task):  # HONEST gate: correct answer
                success += 1
                kept.append(to_training_example(traj, task))
        if i % 25 == 0:
            print(f"  {i}/{len(tasks)} tasks, {success} successful trajectories so far")

    with open(DATASET_PATH, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")

    rate = round(100 * success / total) if total else 0
    print(f"\nTeacher: {TEACHER}")
    print(f"Tasks sampled: {len(tasks)}  Total runs: {total}")
    print(f"Successful (correct-answer) trajectories kept: {len(kept)} ({rate}%)")
    print(f"Saved dataset to {DATASET_PATH.name}")


if __name__ == "__main__":
    main()