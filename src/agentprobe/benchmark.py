"""Multi-family benchmark: run the suite across several models, save results.

Usage:
  uv run python -m agentprobe.benchmark
  uv run python -m agentprobe.benchmark --runs 10
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from . import agent, scorer, spans
from .task_suite import TASKS

MODELS = ["qwen2.5:7b", "llama3.1:8b", "nemotron-mini:4b", "mistral:7b"]
RESULTS_PATH = Path(__file__).resolve().parents[2] / "results.json"


def bench_model(model: str, runs: int) -> dict:
    """Run the full suite `runs` times per task on one model, return aggregates."""
    all_scores = []
    for task in TASKS:
        for i in range(runs):
            traj = agent.run(task.task_id, task.question, model=model)
            spans.write_trajectory(traj, run_index=i, model=model)
            all_scores.append(scorer.score_one(traj, task))

    n = len(all_scores)
    errored = [s for s in all_scores if s["error_rate"] > 0]
    return {
        "model": model,
        "avg_traj": round(sum(s["traj_match"] for s in all_scores) / n, 3),
        "avg_step_eff": round(sum(s["step_efficiency"] for s in all_scores) / n, 3),
        "loop_rate": round(sum(1 for s in all_scores if s["loops"] > 0) / n, 3),
        "error_rate": round(sum(s["error_rate"] for s in all_scores) / n, 3),
        "recovery_rate": round(
            (sum(1 for s in errored if s["recovered"]) / len(errored)) if errored else 1.0, 3
        ),
        "total_runs": n,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark the suite across model families.")
    p.add_argument("--runs", type=int, default=5, help="runs per task per model (default 5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if spans.SPANS_PATH.exists():
        spans.SPANS_PATH.unlink()

    rows = []
    for m in MODELS:
        print(f"benchmarking {m} ({len(TASKS)} tasks x {args.runs} runs) ...")
        rows.append(bench_model(m, args.runs))

    # save results for the dashboard to read
    output = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs_per_task": args.runs,
        "num_tasks": len(TASKS),
        "models": rows,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))

    print(f"\n{'model':18} {'avg_traj':>9} {'avg_step':>9} {'loop_rt':>8} {'err_rt':>7} {'recov':>7} {'runs':>5}")
    print("-" * 72)
    for r in rows:
        print(f"{r['model']:18} {r['avg_traj']:>9} {r['avg_step_eff']:>9} "
              f"{r['loop_rate']:>8} {r['error_rate']:>7} {r['recovery_rate']:>7} {r['total_runs']:>5}")
    print(f"\nsaved results to {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()