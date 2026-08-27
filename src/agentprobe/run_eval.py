"""Week 2 entrypoint: run each task N times, aggregate, print, and emit spans.

Usage:
  uv run python -m agentprobe.run_eval                 # default 5 runs
  uv run python -m agentprobe.run_eval --runs 20       # 20 runs per task
  uv run python -m agentprobe.run_eval --runs 20 --clear   # wipe spans first
"""
from __future__ import annotations
import argparse
from . import agent, scorer, spans
from .task_suite import TASKS
from .model import DEFAULT_MODEL


def aggregate(task, trajs) -> dict:
    scores = [scorer.score_one(t, task) for t in trajs]
    n = len(scores)
    errored = [s for s in scores if s["error_rate"] > 0]
    return {
        "task_id": task.task_id,
        "avg_traj": round(sum(s["traj_match"] for s in scores) / n, 3),
        "avg_step_eff": round(sum(s["step_efficiency"] for s in scores) / n, 3),
        "loop_rate": round(sum(1 for s in scores if s["loops"] > 0) / n, 3),
        "avg_error_rate": round(sum(s["error_rate"] for s in scores) / n, 3),
        "recovery_rate": round(
            (sum(1 for s in errored if s["recovered"]) / len(errored)) if errored else 1.0, 3
        ),
        "runs": n,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run the AgentProbe evaluation suite.")
    p.add_argument("--runs", type=int, default=5, help="runs per task (default 5)")
    p.add_argument("--clear", action="store_true", help="wipe the span file before running")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = DEFAULT_MODEL

    if args.clear and spans.SPANS_PATH.exists():
        spans.SPANS_PATH.unlink()

    print(f"Running {len(TASKS)} tasks x {args.runs} runs on {model} ...")
    rows = []
    for task in TASKS:
        trajs = []
        for i in range(args.runs):
            traj = agent.run(task.task_id, task.question)
            traj.save(f"traces/{task.task_id}_run{i}.json")
            spans.write_trajectory(traj, run_index=i, model=model)
            trajs.append(traj)
        rows.append(aggregate(task, trajs))

    print(f"\n{'task':6} {'avg_traj':>9} {'avg_step':>9} {'loop_rt':>8} {'err_rt':>7} {'recov':>7} {'runs':>5}")
    print("-" * 60)
    for r in rows:
        print(f"{r['task_id']:6} {r['avg_traj']:>9} {r['avg_step_eff']:>9} "
              f"{r['loop_rate']:>8} {r['avg_error_rate']:>7} {r['recovery_rate']:>7} {r['runs']:>5}")

    n = len(rows)
    print("-" * 60)
    print(f"{'AVG':6} {round(sum(r['avg_traj'] for r in rows)/n,3):>9} "
          f"{round(sum(r['avg_step_eff'] for r in rows)/n,3):>9} "
          f"{round(sum(r['loop_rate'] for r in rows)/n,3):>8} "
          f"{round(sum(r['avg_error_rate'] for r in rows)/n,3):>7} "
          f"{round(sum(r['recovery_rate'] for r in rows)/n,3):>7}")


if __name__ == "__main__":
    main()