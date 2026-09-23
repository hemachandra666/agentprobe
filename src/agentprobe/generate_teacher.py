"""Generate honest teacher trajectories as a distillation dataset.

P0-2 fix: acceptance is gated on real task_success (correct final answer via the
shared engine), not on tool-name order. Runs the teacher through the SAME loop
(engine) both models use. Only correct, completed trajectories are kept.

Usage:
  uv run --no-sync python -m agentprobe.generate_teacher --runs 5
"""
from __future__ import annotations
import argparse, json, random, uuid, tempfile, os
from pathlib import Path
from . import agent, scorer, engine
from .tasks_io import load_train

TEACHER = "qwen2.5:7b"



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
    p.add_argument("--output", type=Path, default=Path("teacher_data.jsonl"))
    p.add_argument("--data-dir", type=Path)
    args = p.parse_args()
    if args.runs < 1 or args.max_tasks < 1:
        p.error("--runs and --max-tasks must be positive")
    return args


def main() -> None:
    args = parse_args()
    tasks = load_train(args.data_dir)
    rng = random.Random(42)
    rng.shuffle(tasks)
    tasks = tasks[:args.max_tasks]

    if not tasks:
        raise ValueError("training dataset is empty")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    experiment = "teacher-" + uuid.uuid4().hex
    evidence = output.parent / "runs" / experiment
    evidence.mkdir(parents=True)
    metadata = {"experiment_id": experiment, "status": "running", "teacher": TEACHER,
                "attempted": 0, "accepted": 0, "output": str(output)}
    try:
        with (evidence / "teacher_runs.jsonl").open("w") as traces, (evidence / "accepted.jsonl").open("w") as accepted:
            for i, task in enumerate(tasks, 1):
                for _ in range(args.runs):
                    traj = agent.run(task.task_id, task.question, model=TEACHER)
                    metadata["attempted"] += 1
                    # Save the attempt before scoring or deciding whether to continue.
                    record = engine.run_record(traj, task, TEACHER, experiment, {})
                    traces.write(json.dumps(record) + "\n")
                    traces.flush()
                    if traj.termination in {"timeout", "provider_error", "invalid_action_shape"}:
                        raise RuntimeError(f"teacher inference failed ({traj.termination}); stop process before retrying. Evidence: {evidence}")
                    if scorer.task_success(traj, task) and scorer.error_count(traj) == 0:
                        accepted.write(json.dumps(to_training_example(traj, task)) + "\n")
                        accepted.flush()
                        metadata["accepted"] += 1
                if i % 25 == 0:
                    print(f"  {i}/{len(tasks)} tasks, {metadata['accepted']} successful trajectories so far")
        if metadata["accepted"] == 0:
            raise ValueError("no clean successful trajectories; existing output was preserved")
        # Publish only a successful generation run; preserve old data on failures.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write((evidence / "accepted.jsonl").read_bytes())
            os.replace(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        metadata["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        metadata.update(status="failed", error=str(exc) or type(exc).__name__)
        raise
    finally:
        (evidence / "generation.json").write_text(json.dumps(metadata, indent=2))
        print("Teacher evidence:", evidence)
    rate = 100 * metadata["accepted"] / metadata["attempted"]
    print(f"Teacher: {TEACHER}")
    print(f"Tasks sampled: {len(tasks)}  Total runs: {metadata['attempted']}")
    print(f"Successful trajectories kept: {metadata['accepted']} ({rate:.1f}%)")
    print(f"Saved dataset to {output}")


if __name__ == "__main__":
    main()