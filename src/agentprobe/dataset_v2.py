"""Fresh arithmetic data with associative/commutative expression grouping."""
import argparse
import hashlib
import json
import random
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from .task_gen import FAMILIES, FAMILY_HOLDOUT, Task, _phrase, _answer
from .tasks_io import load_train, load_test


def expression_key(task):
    # Preserve expression structure; flatten/sort add and multiply subtrees.
    # Do not group merely by numeric answer or claim distributive equivalence.
    nums, ops = task.numbers, task.reference_tools
    if len(nums) != len(ops) + 1 or not ops:
        raise ValueError('invalid expression')
    if any(type(n) is not int for n in nums):
        raise ValueError('integer operands required')
    node = ('number', nums[0])
    for op, value in zip(ops, nums[1:]):
        if op not in {'add', 'multiply'}:
            raise ValueError('unknown operator')
        children = []
        for child in (node, ('number', value)):
            children.extend(child[1] if child[0] == op else [child])
        node = (op, tuple(sorted(children, key=repr)))
    return json.dumps(node, separators=(',', ':'))


def check_groups(train, test):
    if {expression_key(t) for t in train} & {expression_key(t) for t in test}:
        raise ValueError('expression groups overlap')


def generate(output, count=200, seed=43):
    output = Path(output)
    if output.exists():
        raise ValueError('output must be a new directory; existing data is preserved')
    if count < 2 or count > 1000:
        raise ValueError('instances per family must be between 2 and 1000')
    old_root = files('agentprobe').joinpath('data')
    old_tasks = load_train(old_root) + load_test(old_root)
    excluded = {expression_key(t) for t in old_tasks}
    seen = set(excluded)
    rng = random.Random(seed)
    train, test = [], []
    for family, ops in FAMILIES.items():
        tasks = []
        for attempt in range(count * 100):
            nums = [rng.randint(1, 100) for _ in range(len(ops) + 1)]
            task = Task(f'v2_{family}__{len(tasks)+1:04d}', family,
                        _phrase(ops, nums), list(ops), nums, len(ops), _answer(ops, nums))
            key = expression_key(task)
            if key in seen:
                continue
            seen.add(key)
            tasks.append(task)
            if len(tasks) == count:
                break
        if len(tasks) != count:
            raise ValueError('could not generate requested unique groups')
        rng.shuffle(tasks)
        n_test = count if family in FAMILY_HOLDOUT else max(1, round(.2 * count))
        test.extend(tasks[:n_test])
        train.extend(tasks[n_test:])
    check_groups(train, test)
    payloads = {name: ''.join(json.dumps(asdict(t)) + '\n' for t in rows)
                for name, rows in [('train', train), ('test', test)]}
    manifest = {'dataset_version': '2.0', 'seed': seed, 'operand_range': [1,100],
                'group_policy': 'associative_commutative_expression_v1',
                'excludes_packaged_v1_groups': True, 'family_holdout': sorted(FAMILY_HOLDOUT),
                'counts': {'train': len(train), 'test': len(test)},
                'sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name,text in payloads.items()}}
    output.mkdir(parents=True, exist_ok=False)
    for name,text in payloads.items():
        (output / f'{name}_tasks.jsonl').write_text(text)
    (output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--instances-per-family', type=int, default=200)
    p.add_argument('--seed', type=int, default=43)
    a = p.parse_args()
    generate(a.output_dir, a.instances_per_family, a.seed)


if __name__ == '__main__':
    main()
