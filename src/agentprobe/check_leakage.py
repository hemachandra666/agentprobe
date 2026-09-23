"""Automated leakage check for P0-1. Fails loudly if train/test overlap."""
from __future__ import annotations
import json, sys
from pathlib import Path

from .tasks_io import data_root
DATA = data_root()


def load_questions(path):
    return [json.loads(l)["question"] for l in path.read_text().splitlines()]


def main() -> None:
    train = load_questions(DATA / "train_tasks.jsonl")
    test = load_questions(DATA / "test_tasks.jsonl")

    train_set, test_set = set(train), set(test)
    overlap = train_set & test_set
    train_dupes = len(train) - len(train_set)
    test_dupes = len(test) - len(test_set)

    print(f"train questions: {len(train)} ({len(train_set)} unique)")
    print(f"test questions:  {len(test)} ({len(test_set)} unique)")
    print(f"exact overlap:   {len(overlap)}")
    print(f"train duplicates:{train_dupes}   test duplicates: {test_dupes}")

    ok = (len(overlap) == 0 and train_dupes == 0 and test_dupes == 0)
    if ok:
        print("PASS: zero overlap, zero duplicates.")
    else:
        print("FAIL: leakage or duplicates found.")
        sys.exit(1)


if __name__ == "__main__":
    main()