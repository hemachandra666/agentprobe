"""Honest three-way comparison on UNSEEN tasks through the shared loop.

Runs teacher, UNTUNED student, and TUNED student on data/test_tasks.jsonl
(tasks never trained on, including two held-out families). Every model uses the
same engine loop and the same honest scorer. task_success = correct final answer.

P1-1: report BOTH macro-average and micro-average error, with denominators.
P1-3: save one COMPLETE run record per attempted run (including failed/zero-call
runs) to traces/comparison_runs.jsonl, alongside the aggregate summary.
"""
from __future__ import annotations
import argparse, json, random, uuid
from pathlib import Path
from . import agent, student_agent, scorer, engine
from .tasks_io import load_test

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "comparison.json"
TRACES_PATH = ROOT / "traces" / "comparison_runs.jsonl"
TEACHER = "qwen2.5:7b"


def eval_provider(provider_factory, label, tasks, runs, experiment_id, trace_file):
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

            # P1-3: write a complete record for THIS run (even if it failed)
            record = engine.run_record(traj, task, label, experiment_id, s)
            trace_file.write(json.dumps(record) + "\n")

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
    recovery = round(recovered_runs / errored_runs, 3) if errored_runs > 0 else None

    return {
        "label": label,
        "task_success_rate": round(succ / n, 3),
        "answer_correct_rate": round(sum(1 for r in rows if r["answer_correct"]) / n, 3),
        "avg_step_efficiency": round(sum(effs) / len(effs), 3) if effs else None,
        "error_rate_macro": round(sum(r["error_rate"] for r in rows) / n, 3),
        "error_rate_macro_denominator_runs": n,
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
    experiment_id = uuid.uuid4().hex[:12]

    tasks = load_test()
    random.Random(42).shuffle(tasks)
    tasks = tasks[:args.max_tasks]

    TRACES_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Experiment {experiment_id}: {len(tasks)} UNSEEN test tasks, {args.runs} run(s) each.\n")

    with open(TRACES_PATH, "w") as tf:
        print(f"1/3 teacher ({TEACHER}) ...")
        teacher = eval_provider(lambda: agent.OllamaProvider(TEACHER),
                                f"teacher ({TEACHER})", tasks, args.runs, experiment_id, tf)

        print("2/3 untuned student (base 1.5B, no fine-tuning) ...")
        untuned = eval_provider(lambda: student_agent.BaseStudentProvider(),
                                "untuned student (1.5B base)", tasks, args.runs, experiment_id, tf)

        print("3/3 tuned student (distilled) ...")
        tuned = eval_provider(lambda: student_agent.StudentProvider(),
                              "tuned student (1.5B distilled)", tasks, args.runs, experiment_id, tf)

    rows = [teacher, untuned, tuned]
    RESULT_PATH.write_text(json.dumps({
        "experiment_id": experiment_id,
        "models": rows,
        "runs_per_task": args.runs,
        "num_test_tasks": len(tasks),
        "traces_file": TRACES_PATH.name,
    }, indent=2))

    print(f"\n{'model':32} {'success':>8} {'ans_ok':>8} {'eff':>6} "
          f"{'err_macro':>10} {'err_micro':>10} {'recov':>7}")
    print("-" * 84)
    for r in rows:
        eff = r["avg_step_efficiency"] if r["avg_step_efficiency"] is not None else "n/a"
        recov = r["recovery_rate"] if r["recovery_rate"] is not None else "n/a"
        print(f"{r['label']:32} {r['task_success_rate']:>8} {r['answer_correct_rate']:>8} "
              f"{str(eff):>6} {r['error_rate_macro']:>10} {r['error_rate_micro']:>10} {str(recov):>7}")

    total_records = sum(r["total_runs"] for r in rows)
    print(f"\nSaved aggregate to {RESULT_PATH.name}")
    print(f"Saved {total_records} complete run records to {TRACES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()