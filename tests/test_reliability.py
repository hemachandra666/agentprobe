"""Failure-path and evidence-integrity checks requiring no GPU or service."""
import copy
import json
import sys
from pathlib import Path

import pytest

from agentprobe import compare, engine, generate_teacher, prepare_training, replay
from agentprobe.evidence import validate_steps
from agentprobe.tasks_io import Task
from agentprobe.trajectory import Trajectory


TASK = Task('reliability_add', 'f_add', 'Add 3 and 4.', ['add'], [3, 4], 1, 7.0)
TASK2 = Task('reliability_mul', 'f_mul', 'Multiply 3 by 4.', ['multiply'], [3, 4], 1, 12.0)


def successful(task=TASK):
    traj = Trajectory(task.task_id, termination='model_final', final_answer=f'Answer: {task.answer}')
    traj.add_step(task.reference_tools[0], {'a': 3, 'b': 4}, str(task.answer))
    return traj


def trace(task=TASK):
    return engine.run_record(successful(task), task, 'fixture', 'test-experiment', {})


def example(task=TASK):
    return generate_teacher.to_training_example(successful(task), task)


@pytest.mark.parametrize('termination', ['provider_error', 'timeout', 'invalid_action_shape'])
def test_compare_aborts_and_preserves_failed_attempt(tmp_path, monkeypatch, termination):
    attempts = []
    def failed(*args, **kwargs):
        attempts.append(1)
        t = Trajectory(TASK.task_id, termination=termination)
        if termination == 'provider_error':
            t.add_step('(provider)', {}, 'offline', status='error', kind='provider')
        return t
    monkeypatch.setattr(compare, 'load_test', lambda directory: [TASK, TASK2])
    monkeypatch.setattr(engine, 'run', failed)
    monkeypatch.setattr(sys, 'argv', ['compare', '--models', 'teacher', '--max-tasks', '2', '--output-dir', str(tmp_path)])
    with pytest.raises(RuntimeError, match='inference failed'):
        compare.main()
    assert len(attempts) == 1
    saved = json.loads(next(tmp_path.glob('*/comparison.json')).read_text())
    assert saved['status'] == 'failed'
    row = json.loads(next(tmp_path.glob('*/comparison_runs.jsonl')).read_text())
    assert row['termination'] == termination


def test_provider_constructor_failure_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, 'load_test', lambda directory: [TASK])
    def broken(model):
        raise FileNotFoundError('missing adapter')
    monkeypatch.setattr(compare.agent, 'OllamaProvider', broken)
    monkeypatch.setattr(sys, 'argv', ['compare', '--models', 'teacher', '--output-dir', str(tmp_path)])
    with pytest.raises(RuntimeError):
        compare.main()
    row = json.loads(next(tmp_path.glob('*/comparison_runs.jsonl')).read_text())
    assert row['termination'] == 'provider_error'
    assert 'missing adapter' in row['steps'][0]['result']


@pytest.mark.parametrize('termination', ['timeout', 'provider_error', 'invalid_action_shape'])
def test_teacher_aborts_preserves_old_dataset_and_partial_evidence(tmp_path, monkeypatch, termination):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate_teacher, 'load_train', lambda directory: [TASK])
    calls = []
    def run(*args, **kwargs):
        calls.append(1)
        return successful() if len(calls) == 1 else Trajectory(TASK.task_id, termination=termination)
    monkeypatch.setattr(generate_teacher.agent, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['generate_teacher', '--runs', '3'])
    output = tmp_path / 'teacher_data.jsonl'
    output.write_text('preserve previous data\n')
    with pytest.raises(RuntimeError):
        generate_teacher.main()
    assert len(calls) == 2
    assert output.read_text() == 'preserve previous data\n'
    folder = next((tmp_path / 'runs').iterdir())
    assert len((folder / 'teacher_runs.jsonl').read_text().splitlines()) == 2
    assert len((folder / 'accepted.jsonl').read_text().splitlines()) == 1
    assert json.loads((folder / 'generation.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('explicit_output', [False, True])
def test_teacher_success_writes_to_requested_working_path(tmp_path, monkeypatch, explicit_output):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate_teacher, 'load_train', lambda directory: [TASK])
    monkeypatch.setattr(generate_teacher.agent, 'run', lambda *a, **kw: successful())
    output = tmp_path / ('nested/teacher.jsonl' if explicit_output else 'teacher_data.jsonl')
    argv = ['generate_teacher', '--runs', '1']
    if explicit_output:
        argv.extend(['--output', str(output)])
    monkeypatch.setattr(sys, 'argv', argv)
    generate_teacher.main()
    assert json.loads(output.read_text()) == example()


def test_zero_accepted_teacher_run_preserves_existing_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate_teacher, 'load_train', lambda directory: [TASK])
    monkeypatch.setattr(generate_teacher.agent, 'run', lambda *a, **kw: Trajectory(TASK.task_id, termination='max_steps'))
    monkeypatch.setattr(sys, 'argv', ['generate_teacher', '--runs', '1'])
    output = tmp_path / 'teacher_data.jsonl'; output.write_text('old')
    with pytest.raises(ValueError, match='no clean'):
        generate_teacher.main()
    assert output.read_text() == 'old'


@pytest.mark.parametrize('field,value', [
    ('task_id', 'wrong'), ('question', 'different'), ('answer', 999),
    ('answer', float('nan')), ('answer', True), ('family', 'wrong'),
    ('final_answer', 'Answer: 999'), ('steps', []),
])
def test_preparation_rejects_bad_identity_or_answers_before_writing(tmp_path, monkeypatch, field, value):
    bad = example(); bad[field] = value
    (tmp_path / 'teacher_data.jsonl').write_text(json.dumps(bad) + '\n' + json.dumps(example(TASK2)) + '\n')
    output = tmp_path / 'train_conversations.jsonl'; output.write_text('unchanged')
    monkeypatch.setattr(prepare_training, 'ROOT', tmp_path)
    monkeypatch.setattr(prepare_training, 'load_train', lambda directory: [TASK, TASK2])
    monkeypatch.setattr(prepare_training, 'load_test', lambda directory: [])
    monkeypatch.setattr(sys, 'argv', ['prepare_training'])
    with pytest.raises(ValueError):
        prepare_training.main()
    assert output.read_text() == 'unchanged'
    assert not (tmp_path / 'validation_conversations.jsonl').exists()


@pytest.mark.parametrize('mutation', ['fabricated_result', 'unknown_tool', 'bad_args', 'error', 'provider', 'nonfinite'])
def test_bad_training_tool_observations_rejected(mutation):
    ex = example(); step = ex['steps'][0]
    if mutation == 'fabricated_result': step['result'] = '999'
    elif mutation == 'unknown_tool': step['tool'] = 'execute_code'
    elif mutation == 'bad_args': step['args'] = {'a': [3], 'b': 4}
    elif mutation == 'error': step['status'] = 'error'
    elif mutation == 'provider': step.update(kind='provider', status='ok')
    elif mutation == 'nonfinite': step['result'] = 'nan'
    with pytest.raises(ValueError):
        prepare_training.validate_example(ex, TASK)


def run_replay(tmp_path, monkeypatch, rows, extra=None):
    source = tmp_path / 'records.jsonl'; output = tmp_path / 'audit.json'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    monkeypatch.setattr(replay, 'load_test', lambda *a: [TASK])
    monkeypatch.setattr(sys, 'argv', ['replay', '--input', str(source), '--output', str(output)] + (extra or []))
    replay.main()
    return json.loads(output.read_text())


@pytest.mark.parametrize('field,value', [('question', 'wrong'), ('true_answer', 999), ('task_id', 'missing'), ('family', 'wrong')])
def test_replay_rejects_dataset_mismatch(tmp_path, monkeypatch, field, value):
    row = trace(); row[field] = value
    with pytest.raises(ValueError):
        run_replay(tmp_path, monkeypatch, [row])
    assert not (tmp_path / 'audit.json').exists()


def test_replay_rejects_fabricated_successful_arithmetic(tmp_path, monkeypatch):
    row = trace(); row['steps'][0]['result'] = '999'
    with pytest.raises(ValueError, match='arithmetic'):
        run_replay(tmp_path, monkeypatch, [row])


def test_replay_rejects_duplicate_runs(tmp_path, monkeypatch):
    row = trace()
    with pytest.raises(ValueError, match='duplicate run'):
        run_replay(tmp_path, monkeypatch, [row, row])


def test_replay_preserves_families_and_repeated_tasks_with_distinct_runs(tmp_path, monkeypatch):
    result = run_replay(tmp_path, monkeypatch, [trace(), trace()])
    assert result['models'][0]['by_family']['f_add']['total_runs'] == 2
    assert result['models'][0]['task_success_rate'] == 1


def test_replay_rejects_empty_input(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='no records'):
        run_replay(tmp_path, monkeypatch, [])


def test_replay_does_not_overwrite_evidence(tmp_path, monkeypatch):
    source = tmp_path / 'trace.jsonl'; source.write_text(json.dumps(trace()))
    original = source.read_bytes()
    monkeypatch.setattr(sys, 'argv', ['replay', '--input', str(source), '--output', str(source)])
    with pytest.raises(ValueError, match='overwrite'):
        replay.main()
    assert source.read_bytes() == original


def test_replay_checks_summary_hash_and_coverage(tmp_path, monkeypatch):
    from dataclasses import asdict
    import hashlib
    row = trace()
    metadata = {'status': 'complete', 'scorer_version': '2.0', 'traces_file': 'records.jsonl',
                'experiment_id': row['experiment_id'], 'num_test_tasks': 1, 'runs_per_task': 1,
                'models': [{'label': 'fixture'}],
                'task_sha256': hashlib.sha256(json.dumps([asdict(TASK)], sort_keys=True).encode()).hexdigest()}
    summary = tmp_path / 'comparison.json'; summary.write_text(json.dumps(metadata))
    assert run_replay(tmp_path, monkeypatch, [row])['summary_checked']
    metadata['task_sha256'] = 'bad'; summary.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='hash'):
        run_replay(tmp_path, monkeypatch, [row])
    metadata['runs_per_task'] = 2; summary.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='coverage'):
        run_replay(tmp_path, monkeypatch, [row])
