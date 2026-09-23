"""Versioned comparisons with complete evidence and separate metric meanings."""
from __future__ import annotations
import argparse, hashlib, json, random, uuid, platform
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from . import agent, student_agent, scorer, engine
from .tasks_io import load_test
from .trajectory import Trajectory
from .process_provider import ProcessProvider

SCORER_VERSION = "2.0"

def summarize(rows):
    if not rows:
        raise ValueError("evaluation requires at least one run")
    n = len(rows)
    eff = [r["step_efficiency"] for r in rows if r["step_efficiency"] is not None]
    recovered = [r["recovered"] for r in rows if r["recovered"] is not None]
    total_calls = sum(r.get("tool_calls", r["steps"]) for r in rows)
    failed = sum(r.get("failed_tool_calls", 0) for r in rows)
    return {
        "task_success_rate": sum(r["task_success"] for r in rows) / n,
        "answer_correct_rate": sum(r["answer_correct"] for r in rows) / n,
        "tool_result_correct_rate": sum(r["tool_result_correct"] for r in rows) / n,
        "completion_rate": sum(r["completed"] for r in rows) / n,
        "avg_step_efficiency": sum(eff) / len(eff) if eff else None,
        "error_rate_macro": sum(r["tool_error_rate"] for r in rows) / n,
        "error_rate_macro_denominator_runs": n,
        "error_rate_micro": failed / total_calls if total_calls else None,
        "error_rate_micro_denominator_calls": total_calls,
        "loop_any_rate": sum(bool(r["loops"] or r["cycle"]) for r in rows) / n,
        "recovery_rate": sum(recovered) / len(recovered) if recovered else None,
        "errored_runs": len(recovered),
        "parse_error_runs": sum(r.get("parse_errors", 0) > 0 for r in rows),
        "provider_error_runs": sum(r.get("termination") == "provider_error" for r in rows),
        "timeout_runs": sum(r.get("termination") == "timeout" for r in rows),
        "total_runs": n,
    }

def eval_provider(provider_factory, label, tasks, runs, experiment_id, trace_file):
    rows = []
    for task in tasks:
        for _ in range(runs):
            try:
                provider = provider_factory()
            except Exception as exc:
                traj = Trajectory(task.task_id, termination="provider_error")
                traj.add_step("(provider)", {}, str(exc), status="error", kind="provider")
                import time
                traj.ended_at = time.time()
            else:
                traj = engine.run(task.task_id, task.question, provider)
            try:
                score = scorer.score_one(traj, task)
            except Exception as e:
                trace_file.write(json.dumps(engine.run_record(traj, task, label, experiment_id,
                                                             {"scoring_error": str(e)})) + "\n")
                trace_file.flush()
                raise RuntimeError("scoring failed; run evidence was preserved") from e
            score["family"] = task.family
            score["termination"] = traj.termination
            tool_steps = [s for s in traj.steps if s.kind == "tool"]
            score["tool_calls"] = len(tool_steps)
            score["failed_tool_calls"] = sum(scorer.is_error(s) for s in tool_steps)
            score["tool_error_rate"] = score["failed_tool_calls"] / len(tool_steps) if tool_steps else 0
            score["parse_errors"] = sum(s.kind == "parse" for s in traj.steps)
            trace_file.write(json.dumps(engine.run_record(traj, task, label, experiment_id, score)) + "\n")
            trace_file.flush()
            if traj.termination in {"timeout", "provider_error", "invalid_action_shape"}:
                raise RuntimeError(
                    f"inference failed ({traj.termination}); trace saved. "
                    "Stop process before restarting inference")
            rows.append(score)
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    return {"label": label, **summarize(rows),
            "by_family": {name: summarize(group) for name, group in families.items()}}

def parse_args(default_models=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--max-tasks", type=int, default=100)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("runs"))
    p.add_argument("--models", nargs="+", default=default_models or ["teacher", "untuned", "tuned"])
    a = p.parse_args()
    if a.runs < 1 or a.max_tasks < 1:
        p.error("--runs and --max-tasks must be positive")
    return a

def main(default_models=None):
    args = parse_args(default_models)
    tasks = load_test(args.data_dir)
    random.Random(42).shuffle(tasks)
    tasks = tasks[:args.max_tasks]
    if not tasks:
        raise ValueError("test dataset is empty")
    experiment = uuid.uuid4().hex
    out = args.output_dir / experiment
    out.mkdir(parents=True)
    metadata = {"experiment_id": experiment, "scorer_version": SCORER_VERSION,
                "status": "running", "num_test_tasks": len(tasks), "runs_per_task": args.runs,
                "student_execution": "spawn_process",
                "python": platform.python_version(), "models_requested": args.models,
                "task_sha256": hashlib.sha256(json.dumps([asdict(t) for t in tasks], sort_keys=True).encode()).hexdigest(),
                "generation": {"ollama": {"temperature": 0, "seed": 42, "num_predict": 120},
                               "student": {"do_sample": False, "max_new_tokens": 120}},
                "traces_file": "comparison_runs.jsonl"}
    (out / "comparison.json").write_text(json.dumps(metadata, indent=2))
    results = []
    try:
        with (out / "comparison_runs.jsonl").open("w") as trace_file:
            for name in args.models:
                if name == "tuned":
                    factory, label = student_agent.StudentProvider, "tuned student (1.5B distilled)"
                elif name == "untuned":
                    factory, label = student_agent.BaseStudentProvider, "untuned student (1.5B base)"
                else:
                    model = "qwen2.5:7b" if name == "teacher" else name
                    factory, label = lambda m=model: agent.OllamaProvider(m), "teacher (qwen2.5:7b)" if name == "teacher" else model
                if name in {"tuned", "untuned"}:
                    with ProcessProvider(factory) as worker:
                        result = eval_provider(lambda: worker, label, tasks, args.runs, experiment, trace_file)
                else:
                    result = eval_provider(factory, label, tasks, args.runs, experiment, trace_file)
                results.append(result)
                print(label, result["task_success_rate"])
        metadata["status"] = "complete"
    except (Exception, KeyboardInterrupt) as e:
        metadata.update(status="failed", error=str(e))
        raise
    finally:
        (out / "comparison.json").write_text(json.dumps({**metadata, "models": results}, indent=2))
        print("Evidence:", out)

if __name__ == "__main__":
    main()
