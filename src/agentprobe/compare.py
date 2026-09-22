"""P0-5: honest three-way comparison on UNSEEN tasks through the shared loop.

Runs teacher, UNTUNED student, and TUNED student on data/test_tasks.jsonl
(tasks never trained on, including two held-out families). Every model uses the
same engine loop and the same honest scorer. task_success = correct final answer.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from . import agent, student_agent, scorer, engine
from .tasks_io import load_test

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "comparison.json"
TEACHER = "qwen2.5:7b"


def eval_provider(provider_factory, label, tasks, runs):
    rows = []
    for task in tasks:
        for _ in range(runs):
            provider = provider_factory()
            traj = engine.run(task.task_id, task.question, provider)
            s = scorer.score_one(traj, task)
            s["family"] = task.family
            rows.append(s)
    n = len(rows)
    succ = sum(1 for r in rows if r["task_success"])
    # step efficiency only over successful runs (P0-4)
    effs = [r["step_efficiency"] for r in rows if r["step_efficiency"] is not None]
    return {
        "label": label,
        "task_success_rate": round(succ / n, 3),
        "answer_correct_rate": round(sum(1 for r in rows if r["answer_correct"]) / n, 3),
        "avg_step_efficiency": round(sum(effs) / len(effs), 3) if effs else None,
        "error_rate": round(sum(r["error_rate"] for r in rows) / n, 3),
        "loop_any_rate": round(sum(1 for r in rows if r["loops"] > 0) / n, 3),
        "total_runs": n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1, help="runs per task (default 1)")
    p.add_argument("--max-tasks", type=int, default=100, help="cap test tasks (default 100)")
    args = p.parse_args()

    tasks = load_test()
    random.Random(42).shuffle(tasks)
    tasks = tasks[:args.max_tasks]

    print(f"Evaluating on {len(tasks)} UNSEEN test tasks, {args.runs} run(s) each.\n")

    print(f"1/3 teacher ({TEACHER}) ...")
    teacher = eval_provider(lambda: agent.OllamaProvider(TEACHER), f"teacher ({TEACHER})", tasks, args.runs)

    print("2/3 untuned student (base 1.5B, no fine-tuning) ...")
    untuned = eval_provider(lambda: student_agent.BaseStudentProvider(), "untuned student (1.5B base)", tasks, args.runs)

    print("3/3 tuned student (distilled) ...")
    tuned = eval_provider(lambda: student_agent.StudentProvider(), "tuned student (1.5B distilled)", tasks, args.runs)

    rows = [teacher, untuned, tuned]
    RESULT_PATH.write_text(json.dumps({"models": rows, "runs_per_task": args.runs,
                                       "num_test_tasks": len(tasks)}, indent=2))

    print(f"\n{'model':32} {'success':>8} {'answer_ok':>10} {'step_eff':>9} {'err':>6} {'loop':>6}")
    print("-" * 76)
    for r in rows:
        eff = r["avg_step_efficiency"] if r["avg_step_efficiency"] is not None else "n/a"
        print(f"{r['label']:32} {r['task_success_rate']:>8} {r['answer_correct_rate']:>10} "
              f"{str(eff):>9} {r['error_rate']:>6} {r['loop_any_rate']:>6}")

    print(f"\nThe honest question: did fine-tuning move the student's success rate")
    print(f"from the untuned baseline toward the teacher? Compare column 'success'.")
    print(f"Saved to {RESULT_PATH.name}")


if __name__ == "__main__":
    main()