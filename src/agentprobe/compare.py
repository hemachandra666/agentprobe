"""Honest three-way comparison on UNSEEN tasks through the shared loop.

Runs teacher, UNTUNED student, and TUNED student on data/test_tasks.jsonl
(tasks never trained on, including two held-out families). Every model uses the
same engine loop and the same honest scorer. task_success = correct final answer.

P1-1: report BOTH macro-average error (mean of per-run error fractions) and
micro-average error (total failed calls / total calls), with explicit
denominators, since they differ when runs have different lengths. Recovery is
reported as N/A when no run actually errored.
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
    total_calls = 0
    total_failed_calls = 0
    errored_runs = 0
    recovered_runs = 0
    for task in tasks:
        for _ in range(runs):
            provider = provider_factory()
            traj = engine.run(task.task_id, task.question, provider)
            s = scorer.score_one(traj, task)
            s["family"] = task.family
            rows.append(s)
            # micro-average accounting: count actual calls and failures
            calls = traj.step_count
            failed = scorer.error_count(traj)
            total_calls += calls
            total_failed_calls += failed
            if failed > 0:
                errored_runs += 1
                if s["recovered"] is True:
                    recovered_runs += 1

    n = len(rows)
    succ = sum(1 for r in rows if r["task_success"])
    effs = [r["step_efficiency"] for r in rows if r["step_efficiency"] is not None]

    # recovery is only defined over runs that actually errored
    recovery = round(recovered_runs / errored_runs, 3) if errored_runs > 0 else None

    return {
        "label": label,
        "task_success_rate": round(succ / n, 3),
        "answer_correct_rate": round(sum(1 for r in rows if r["answer_correct"]) / n, 3),
        "avg_step_efficiency": round(sum(effs) / len(effs), 3) if effs else None,
        # macro: mean of per-run error fractions, denominator = number of runs
        "error_rate_macro": round(sum(r["error_rate"] for r in rows) / n, 3),
        "error_rate_macro_denominator_runs": n,
        # micro: total failed calls / total calls, denominator = number of calls
        "error_rate_micro": round(total_failed_calls / total_calls, 3) if total_calls else 0.0,
        "error_rate_micro_denominator_calls": total_calls,
        "loop_any_rate": round(sum(1 for r in rows if r["loops"] > 0) / n, 3),
        "recovery_rate": recovery,
        "errored_runs": errored_runs,
        "total_runs": n,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1, help="runs per task (default 1)")
    p.add_argument("--max-tasks", type=int, default=100, help="cap test tasks (default 100)")
    args = p.parse_args()
    if args.runs < 1:
        p.error("--runs must be >= 1")
    if args.max_tasks < 1:
        p.error("--max-tasks must be >= 1")
    return args


def main() -> None:
    args = parse_args()

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

    print(f"\n{'model':32} {'success':>8} {'ans_ok':>8} {'eff':>6} "
          f"{'err_macro':>10} {'err_micro':>10} {'recov':>7}")
    print("-" * 84)
    for r in rows:
        eff = r["avg_step_efficiency"] if r["avg_step_efficiency"] is not None else "n/a"
        recov = r["recovery_rate"] if r["recovery_rate"] is not None else "n/a"
        print(f"{r['label']:32} {r['task_success_rate']:>8} {r['answer_correct_rate']:>8} "
              f"{str(eff):>6} {r['error_rate_macro']:>10} {r['error_rate_micro']:>10} {str(recov):>7}")

    print(f"\nerr_macro = mean of per-run error fractions (denominator = runs).")
    print(f"err_micro = failed calls / total calls (denominator = calls). They differ")
    print(f"when runs have different lengths. recov = recovered / errored runs, N/A if none errored.")
    print(f"Saved to {RESULT_PATH.name}")


if __name__ == "__main__":
    main()