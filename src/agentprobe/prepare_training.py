"""Format honest teacher trajectories into training conversations.

Uses the correct-answer-gated teacher_data.jsonl (P0-2). The real unseen-task
test set already lives in data/test_tasks.jsonl (P0-1), so this only turns the
teacher's SUCCESSFUL trajectories into training examples. No re-splitting here;
the task-level split was done before any data was generated, so there is no leakage.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEACHER_DATA = ROOT / "teacher_data.jsonl"
TRAIN_DATA = ROOT / "train_conversations.jsonl"

SYSTEM = ("You are a calculator agent. Use the add and multiply tools to compute step by step, "
          "then give the final number.")


def format_tool_calls(steps) -> str:
    lines = []
    for s in steps:
        args = ", ".join(f"{k}={v}" for k, v in s["args"].items())
        lines.append(f"{s['tool']}({args}) = {s['result']}")
    return "\n".join(lines)


def to_conversation(ex) -> dict:
    target = format_tool_calls(ex["steps"])
    answer = ex.get("final_answer") or f"The answer is {ex['answer']}."
    return {
        "task_id": ex["task_id"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": f"{target}\n\nAnswer: {answer}"},
        ],
    }


def main() -> None:
    examples = []
    with open(TEACHER_DATA) as f:
        for line in f:
            examples.append(to_conversation(json.loads(line)))

    with open(TRAIN_DATA, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} training conversations to {TRAIN_DATA.name}")
    if examples:
        print("\nExample:")
        print(json.dumps(examples[0]["messages"], indent=2))


if __name__ == "__main__":
    main()