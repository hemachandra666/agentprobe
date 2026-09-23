"""Create disjoint train/validation next-action examples; never use final-test data."""
import json, random, hashlib, argparse
from pathlib import Path
from .engine import SYSTEM
from .evidence import finite_number, validate_identity, validate_steps
from .scorer import final_numeric
from .tasks_io import load_test, load_train
from .student_agent import _history_to_text

ROOT = Path.cwd()

def validate_example(ex, task=None):
    if task is not None:
        validate_identity(ex, task, answer_key="answer")
    expected = finite_number(ex.get("answer"))
    if not ex.get("steps"):
        raise ValueError("training trajectory must contain a tool call")
    validate_steps(ex["steps"], clean_only=True)
    if (final_numeric(ex.get("final_answer", "")) != expected
            or finite_number(ex["steps"][-1]["result"]) != expected):
        raise ValueError("training final answer and tool result must match the reference")


def to_examples(ex):
    answer = final_numeric(ex.get("final_answer", ""))
    if answer is None or abs(answer - ex["answer"]) >= 1e-6:
        return []
    validate_example(ex)
    history, rows = [], []
    for step in ex["steps"]:
        if str(step["result"]).startswith("error:"):
            return []
        args = step["args"]
        call = f"{step['tool']}(a={args['a']}, b={args['b']})"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["question"]}]
        messages.extend(_history_to_text(history))
        rows.append({"task_id": ex["task_id"], "messages": messages + [{"role": "assistant", "content": call}]})
        history.extend([("call", step["tool"], args), ("observation", step["result"])])
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["question"]}]
    messages.extend(_history_to_text(history))
    rows.append({"task_id": ex["task_id"], "messages": messages + [{"role": "assistant", "content": f"Answer: {answer}"}]})
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    examples = [json.loads(line) for line in (ROOT / "teacher_data.jsonl").read_text().splitlines()]
    canonical = {t.task_id: t for t in load_train(args.data_dir)}
    allowed = {t.question for t in canonical.values()}
    forbidden = {t.question for t in load_test(args.data_dir)}
    if any(e["question"] not in allowed or e["question"] in forbidden for e in examples):
        raise ValueError("teacher data must contain training questions only")
    for ex in examples:
        if ex.get("task_id") not in canonical:
            raise ValueError("unknown training task ID")
        validate_example(ex, canonical[ex["task_id"]])
    questions = sorted({e["question"] for e in examples})
    if len(questions) < 2:
        raise ValueError("at least two unique training questions required")
    random.Random(42).shuffle(questions)
    validation = set(questions[:max(1, round(.15 * len(questions)))])
    groups = {"train": [], "validation": []}
    for ex in examples:
        groups["validation" if ex["question"] in validation else "train"].extend(to_examples(ex))
    if any(not rows for rows in groups.values()):
        raise ValueError("both training and validation need valid examples")
    for name, rows in groups.items():
        (ROOT / f"{name}_conversations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(name, len(rows), "next-action examples")
    (ROOT / "training_split.json").write_text(json.dumps({"schema_version": "2.0", "train_sha256": hashlib.sha256((ROOT / "train_conversations.jsonl").read_bytes()).hexdigest(), "validation_sha256": hashlib.sha256((ROOT / "validation_conversations.jsonl").read_bytes()).hexdigest(), "seed": 42, "validation_questions": sorted(validation),
                                                         "train_questions": sorted(set(questions)-validation)}, indent=2))

if __name__ == "__main__":
    main()
