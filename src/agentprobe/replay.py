"""Replay evidence against the selected dataset without running a model."""
import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from .trajectory import Trajectory
from .tasks_io import load_test
from .scorer import score_one, is_error
from .compare import summarize
from .evidence import validate_identity, validate_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/replay_audit.json'))
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--summary', type=Path, help='Optional matching comparison.json; matching sibling is detected automatically')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("replay output must not overwrite input evidence")
    dataset = load_test(args.data_dir) if args.data_dir else load_test()
    tasks = {t.task_id: t for t in dataset}
    if len(tasks) != len(dataset):
        raise ValueError("duplicate task IDs in replay dataset")
    metadata = None
    summary_path = args.summary or args.input.with_name('comparison.json')
    if summary_path.is_file():
        candidate = json.loads(summary_path.read_text())
        matching = candidate.get('traces_file') == args.input.name
        if args.summary or (matching and candidate.get('scorer_version') == '2.0'):
            if args.output.resolve() == summary_path.resolve():
                raise ValueError("replay output must not overwrite original summary")
            if not matching:
                raise ValueError("summary does not name this trace file")
            metadata = candidate
    elif args.summary:
        raise ValueError("requested summary file does not exist")
    groups, family_groups, task_orders = defaultdict(list), defaultdict(lambda: defaultdict(list)), {}
    run_ids, experiments = set(), set()
    counts = defaultdict(lambda: defaultdict(int))
    for line_number, line in enumerate(args.input.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        task = tasks.get(record.get('task_id'))
        if task is None:
            raise ValueError(f"unknown task ID on line {line_number}")
        validate_identity(record, task)
        validate_steps(record['steps'])
        label = record.get('model')
        if not isinstance(label, str) or not label:
            raise ValueError("trace must identify its model")
        if 'run_id' in record:
            if record['run_id'] in run_ids:
                raise ValueError("duplicate run ID in replay evidence")
            run_ids.add(record['run_id'])
        if record.get('experiment_id'):
            experiments.add(record['experiment_id'])
        if len(experiments) > 1:
            raise ValueError("replay one experiment at a time")
        if metadata and record.get('experiment_id') != metadata.get('experiment_id'):
            raise ValueError("trace experiment does not match summary")
        task_orders.setdefault(label, {})[task.task_id] = task
        counts[label][task.task_id] += 1
        traj = Trajectory(task.task_id, final_answer=record.get('final_answer', ''),
                          termination=record.get('termination', 'unknown'))
        for step in record['steps']:
            traj.add_step(step['tool'], step['args'], step['result'],
                          status=step.get('status'), kind=step.get('kind', 'tool'))
        score = score_one(traj, task)
        tool_steps = [s for s in traj.steps if s.kind == 'tool']
        failed = sum(is_error(s) for s in tool_steps)
        score.update(tool_calls=len(tool_steps), failed_tool_calls=failed,
                     tool_error_rate=failed / len(tool_steps) if tool_steps else 0,
                     parse_errors=sum(s.kind == 'parse' for s in traj.steps),
                     termination=traj.termination)
        groups[label].append(score)
        family_groups[label][getattr(task, 'family', 'unknown')].append(score)
    if not groups:
        raise ValueError("replay input has no records")
    if metadata and metadata.get('status') == 'complete':
        if set(groups) != {row['label'] for row in metadata['models']}:
            raise ValueError("trace model coverage does not match summary")
        for label, selected in task_orders.items():
            if len(selected) != metadata['num_test_tasks'] or any(
                count != metadata['runs_per_task'] for count in counts[label].values()
            ):
                raise ValueError("trace task/run coverage does not match summary")
            digest = hashlib.sha256(json.dumps([asdict(t) for t in selected.values()], sort_keys=True).encode()).hexdigest()
            if digest != metadata['task_sha256']:
                raise ValueError("selected dataset hash does not match experiment")
    result = {
        'status': 'replay_only', 'scorer_version': '2.0',
        'identity_validation': 'task ID, question, reference answer and successful tool arithmetic',
        'summary_checked': bool(metadata),
        'warning': 'Replay is not fresh inference and cannot restore actions omitted by legacy traces.',
        'models': [
            {'label': label, **summarize(rows),
             'by_family': {family: summarize(items) for family, items in family_groups[label].items()}}
            for label, rows in groups.items()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for row in result['models']:
        print(row['label'], row['task_success_rate'])
    print(result['warning'])


if __name__ == '__main__':
    main()
