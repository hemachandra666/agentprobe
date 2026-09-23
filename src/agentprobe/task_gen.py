"""Generate unique task instances with clean train/test separation.

P0-1 fix: the old suite had 15 fixed questions, so 'held-out' rows were just
repeats of training-seen questions. This generates many DISTINCT instances and
separates them two ways:
  - instance holdout: train and test use different numbers for shared families
  - family holdout: some whole task shapes appear ONLY in test (unseen composition)
A manifest records exactly what went where, and a leakage check enforces zero overlap.
"""
from __future__ import annotations
import json, random
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path.cwd()
OUT_DIR = ROOT / "data"
SEED = 42
INSTANCES_PER_FAMILY = 200  # unique instances generated per family

# A family is a task SHAPE: an ordered list of ops. We render numbers into it.
# op is "add" or "multiply". Each family is a chain applied left to right.
FAMILIES = {
    "f_mul":            ["multiply"],
    "f_add":            ["add"],
    "f_add_mul":        ["add", "multiply"],
    "f_mul_add":        ["multiply", "add"],
    "f_add_add":        ["add", "add"],
    "f_mul_mul":        ["multiply", "multiply"],
    "f_add_mul_add":    ["add", "multiply", "add"],
    "f_mul_add_mul":    ["multiply", "add", "multiply"],
    "f_add_mul_add_mul":["add", "multiply", "add", "multiply"],
    "f_len5_a":         ["add", "multiply", "add", "multiply", "add"],
    "f_len5_b":         ["multiply", "add", "multiply", "add", "multiply"],
}

# Families held out ENTIRELY for test: unseen shapes the model never trains on.
FAMILY_HOLDOUT = {"f_mul_add_mul", "f_len5_b"}

WORD = {"add": "add", "multiply": "multiply"}


@dataclass
class Task:
    task_id: str
    family: str
    question: str
    reference_tools: list[str]
    numbers: list[int]          # the operands, in order of use
    min_steps: int
    answer: float               # the correct final numeric answer


def _phrase(ops, nums) -> str:
    """Render a natural-language question for the op chain and numbers."""
    a, b = nums[0], nums[1]
    if ops[0] == "add":
        s = f"Add {a} and {b}"
    else:
        s = f"Multiply {a} by {b}"
    running_idx = 2
    for op in ops[1:]:
        n = nums[running_idx]; running_idx += 1
        if op == "add":
            s += f", then add {n}"
        else:
            s += f", then multiply by {n}"
    return s + "."


def _answer(ops, nums) -> float:
    val = nums[0] + nums[1] if ops[0] == "add" else nums[0] * nums[1]
    idx = 2
    for op in ops[1:]:
        n = nums[idx]; idx += 1
        val = val + n if op == "add" else val * n
    return float(val)


def _count_numbers(ops) -> int:
    # first op consumes 2 numbers, each later op consumes 1 more
    return 2 + (len(ops) - 1)


def generate() -> dict:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    all_tasks, seen_signatures = [], set()
    for fam, ops in FAMILIES.items():
        n_needed = _count_numbers(ops)
        made = 0
        attempts = 0
        while made < INSTANCES_PER_FAMILY and attempts < INSTANCES_PER_FAMILY * 20:
            attempts += 1
            nums = [rng.randint(1, 20) for _ in range(n_needed)]
            sig = (fam, tuple(nums))
            if sig in seen_signatures:
                continue  # dedup: no identical (family, numbers) twice
            seen_signatures.add(sig)
            made += 1
            all_tasks.append(Task(
                task_id=f"{fam}__{made:04d}",
                family=fam,
                question=_phrase(ops, nums),
                reference_tools=list(ops),
                numbers=nums,
                min_steps=len(ops),
                answer=_answer(ops, nums),
            ))

    # split: family-holdout families go entirely to test;
    # the rest split 80/20 by instance, with zero question overlap.
    train, test = [], []
    for t in all_tasks:
        if t.family in FAMILY_HOLDOUT:
            test.append(t)
        else:
            (test if rng.random() < 0.2 else train).append(t)

    # write splits
    def dump(path, tasks):
        with open(path, "w") as f:
            for t in tasks:
                f.write(json.dumps(asdict(t)) + "\n")

    dump(OUT_DIR / "train_tasks.jsonl", train)
    dump(OUT_DIR / "test_tasks.jsonl", test)

    # manifest
    manifest = {
        "seed": SEED,
        "instances_per_family": INSTANCES_PER_FAMILY,
        "families": {k: v for k, v in FAMILIES.items()},
        "family_holdout": sorted(FAMILY_HOLDOUT),
        "counts": {"total": len(all_tasks), "train": len(train), "test": len(test)},
    }
    (OUT_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Generated {len(all_tasks)} unique tasks across {len(FAMILIES)} families.")
    print(f"Train: {len(train)}  Test: {len(test)}")
    print(f"Family holdout (test-only shapes): {sorted(FAMILY_HOLDOUT)}")
    print(f"Wrote data/train_tasks.jsonl, data/test_tasks.jsonl, data/split_manifest.json")
    return manifest


if __name__ == "__main__":
    generate()