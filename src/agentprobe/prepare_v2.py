"""Prepare v2 examples in a new directory; preserve v1 outputs."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from .dataset_v2 import expression_key, check_groups
from .tasks_io import load_train, load_test
from .prepare_training import validate_example, to_examples


def prepare(data_dir, teacher_file, output):
    data_dir, teacher_file, output = map(Path, (data_dir, teacher_file, output))
    if output.exists():
        raise ValueError('output must be a new directory')
    manifest = json.loads((data_dir / 'split_manifest.json').read_text())
    if manifest.get('dataset_version') != '2.0':
        raise ValueError('version-2 dataset required')
    for name in ('train', 'test'):
        if hashlib.sha256((data_dir / f'{name}_tasks.jsonl').read_bytes()).hexdigest() != manifest['sha256'][name]:
            raise ValueError('dataset hash mismatch')
    train, test = load_train(data_dir), load_test(data_dir)
    check_groups(train, test)
    canonical = {t.task_id: t for t in train}
    examples = [json.loads(line) for line in teacher_file.read_text().splitlines() if line.strip()]
    for ex in examples:
        if ex.get('task_id') not in canonical:
            raise ValueError('teacher task is not in training split')
        validate_example(ex, canonical[ex['task_id']])
    keys = sorted({expression_key(canonical[e['task_id']]) for e in examples})
    if len(keys) < 2:
        raise ValueError('at least two accepted expression groups required')
    random.Random(43).shuffle(keys)
    val = set(keys[:max(1, round(.15 * len(keys)))])
    groups = {'train': [], 'validation': []}
    for ex in examples:
        name = 'validation' if expression_key(canonical[ex['task_id']]) in val else 'train'
        groups[name].extend(to_examples(ex))
    if any(not rows for rows in groups.values()):
        raise ValueError('empty train or validation examples')
    payloads = {n: ''.join(json.dumps(r)+'\n' for r in rows) for n,rows in groups.items()}
    summary = {'schema_version': '2.0', 'dataset_version': '2.0', 'seed': 43,
               'dataset_manifest': manifest, 'teacher_sha256': hashlib.sha256(teacher_file.read_bytes()).hexdigest(),
               'validation_groups': sorted(val), 'train_groups': sorted(set(keys)-val),
               **{n+'_sha256': hashlib.sha256(s.encode()).hexdigest() for n,s in payloads.items()}}
    output.mkdir(parents=True, exist_ok=False)
    for n,s in payloads.items():
        (output / f'{n}_conversations.jsonl').write_text(s)
    (output / 'training_split.json').write_text(json.dumps(summary, indent=2)+'\n')
    print({n: len(r) for n,r in groups.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True, type=Path)
    p.add_argument('--teacher-file', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    a = p.parse_args()
    prepare(a.data_dir, a.teacher_file, a.output_dir)


if __name__ == '__main__':
    main()
