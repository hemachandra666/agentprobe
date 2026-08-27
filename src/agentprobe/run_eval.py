"""Week 1 entrypoint: run every task, score it, print the report."""
from __future__ import annotations
from . import agent, scorer
from .task_suite import TASKS


def main() -> None:
    rows = []
    for task in TASKS:
        traj = agent.run(task.task_id, task.question)
        traj.save(f"traces/{task.task_id}.json")
        rows.append(scorer.score_one(traj, task))

    print(f"\n{'task':6} {'tool_sel':>9} {'step_eff':>9} {'steps':>6}  answer")
    print("-" * 60)
    for r in rows:
        print(f"{r['task_id']:6} {r['tool_selection']:>9} {r['step_efficiency']:>9} {r['steps']:>6}  {r['answer'][:30]}")

    n = len(rows)
    avg_tool = sum(r["tool_selection"] for r in rows) / n
    avg_step = sum(r["step_efficiency"] for r in rows) / n
    print("-" * 60)
    print(f"{'AVG':6} {round(avg_tool,3):>9} {round(avg_step,3):>9}")


if __name__ == "__main__":
    main()