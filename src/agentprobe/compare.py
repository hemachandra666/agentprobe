"""Head-to-head: distilled student vs qwen teacher on trajectory behavior.

Runs both through the SAME tasks and scorers, prints the comparison.
This is the headline result: did the small student keep the teacher's behavior?
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from . import agent, student_agent, scorer
from .task_suite import TASKS

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "comparison.json"
TEACHER = "qwen2.5:7b"


def eval_runner(run_fn, label, runs, model=None):
    """Run one agent (teacher or student) over all tasks, return aggregate scores."""
    all_scores = []
    for task in TASKS:
        for _ in range(runs):
            if model:
                traj = run_fn(task.task_id, task.question, model=model)
            else:
                traj = run_fn(task.task_id, task.question)
            all_scores.append(scorer.score_one(traj, task))
    n = len(all_scores)
    errored = [s for s in all_scores if s["error_rate"] > 0]
    return {
        "label": label,
        "avg_traj": round(sum(s["traj_match"] for s in all_scores) / n, 3),
        "avg_step_eff": round(sum(s["step_efficiency"] for s in all_scores) / n, 3),
        "loop_rate": round(sum(1 for s in all_scores if s["loops"] > 0) / n, 3),
        "error_rate": round(sum(s["error_rate"] for s in all_scores) / n, 3),
        "recovery_rate": round(
            (sum(1 for s in errored if s["recovered"]) / len(errored)) if errored else 1.0, 3),
        "total_runs": n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3, help="runs per task (default 3)")
    args = p.parse_args()

    print(f"Evaluating teacher ({TEACHER}) ...")
    teacher = eval_runner(agent.run, f"teacher ({TEACHER})", args.runs, model=TEACHER)

    print("Evaluating distilled student (1.5B) ...")
    student = eval_runner(student_agent.run, "student (1.5B distilled)", args.runs)

    rows = [teacher, student]
    RESULT_PATH.write_text(json.dumps({"models": rows, "runs_per_task": args.runs}, indent=2))

    print(f"\n{'model':26} {'traj':>6} {'step_eff':>9} {'loop':>6} {'err':>6} {'recov':>6}")
    print("-" * 66)
    for r in rows:
        print(f"{r['label']:26} {r['avg_traj']:>6} {r['avg_step_eff']:>9} "
              f"{r['loop_rate']:>6} {r['error_rate']:>6} {r['recovery_rate']:>6}")

    # the headline number
    kept = round(100 * student["avg_traj"] / teacher["avg_traj"]) if teacher["avg_traj"] else 0
    print(f"\nHeadline: the 1.5B student kept {kept}% of the teacher's trajectory quality.")
    print(f"Saved comparison to {RESULT_PATH.name}")


if __name__ == "__main__":
    main()