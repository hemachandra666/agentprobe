"""Reshape teacher trajectories into tool-calling conversations, split train/test.

Teaches the student to emit the same tool-call sequence the teacher used.
Holds out ~15% as a fair exam: fresh examples spanning all difficulties.
"""
from __future__ import annotations
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEACHER_DATA = ROOT / "teacher_data.jsonl"
TRAIN_DATA = ROOT / "train_conversations.jsonl"
TEST_DATA = ROOT / "test_conversations.jsonl"

SYSTEM = ("You are a calculator agent. Use the add and multiply tools to compute step by step, "
          "then give the final number.")
TEST_FRACTION = 0.15
SEED = 42


def format_tool_calls(steps) -> str:
    lines = []
    for s in steps:
        args = ", ".join(f"{k}={v}" for k, v in s["args"].items())
        lines.append(f"{s['tool']}({args}) = {s['result']}")
    return "\n".join(lines)


def to_conversation(ex) -> dict:
    target = format_tool_calls(ex["steps"])
    return {
        "task_id": ex["task_id"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": f"{target}\n\nAnswer: {ex['final_answer']}"},
        ],
    }


def main() -> None:
    # group examples by task so we can hold out some from EVERY task (stratified)
    by_task = {}
    with open(TEACHER_DATA) as f:
        for line in f:
            ex = json.loads(line)
            by_task.setdefault(ex["task_id"], []).append(ex)

    rng = random.Random(SEED)
    train, test = [], []
    for task_id, rows in by_task.items():
        rng.shuffle(rows)
        n_test = max(1, round(len(rows) * TEST_FRACTION))
        test.extend(rows[:n_test])
        train.extend(rows[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)

    with open(TRAIN_DATA, "w") as f:
        for ex in train:
            f.write(json.dumps(to_conversation(ex)) + "\n")
    with open(TEST_DATA, "w") as f:
        for ex in test:
            f.write(json.dumps(to_conversation(ex)) + "\n")

    print(f"Train: {len(train)} examples -> {TRAIN_DATA.name}")
    print(f"Test:  {len(test)} examples (held out) -> {TEST_DATA.name}")
    # show the held-out task spread, to confirm it covers easy and hard
    spread = {}
    for ex in test:
        spread[ex["task_id"]] = spread.get(ex["task_id"], 0) + 1
    print("Held-out task spread:", dict(sorted(spread.items())))


if __name__ == "__main__":
    main()