import copy
import json
from pathlib import Path
import pytest
from agentprobe.refund_workflow import development_cases, run_case, ReferenceProvider, score, Environment
from agentprobe.refund_fresh import fresh_cases
from agentprobe.refund_gate import check
from agentprobe.provenance import adapter_identity


def test_retry_budget_matches_gate():
    c = next(c for c in development_cases() if c.family == 'order_unavailable')
    row = run_case(c, ReferenceProvider())
    assert score(c, row)['task_success'] and check(c.question, row)['action'] == 'allow'
    row['steps'].append(copy.deepcopy(row['steps'][-1]))
    assert not score(c, row)['task_success']
    assert check(c.question, row)['action'] == 'block'


def test_fresh_published_evidence_remains_valid():
    root = Path(__file__).resolve().parents[1] / 'docs/results/refund-fresh/teacher'
    rows = [json.loads(l) for l in (root / 'trajectories.jsonl').read_text().splitlines()]
    cases = {c.case_id: c for c in fresh_cases()}
    assert len(rows) == len({r['case_id'] for r in rows}) == 28
    for row in rows:
        case = cases[row['case_id']]
        env = Environment(case)
        for step in row['steps']:
            assert env.execute(step['tool'], step['args']) == step['result']
        updated = score(case, row)
        assert {k: v for k, v in updated.items() if k != 'scorer_version'} == {k: v for k, v in row['score'].items() if k != 'scorer_version'}
        assert check(case.question, row) == row['gate']
    assert sum(r['score']['task_success'] for r in rows) == 10


def test_adapter_identity_detects_changes_and_does_not_invent_revision(tmp_path):
    for n in ['adapter_model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']:
        (tmp_path / n).write_text('fixture')
    (tmp_path / 'adapter_config.json').write_text('{"revision": null}')
    before = adapter_identity(tmp_path)
    assert before['resolved_base_revision'] is None
    (tmp_path / 'adapter_model.safetensors').write_text('changed')
    assert adapter_identity(tmp_path)['sha256'] != before['sha256']
    (tmp_path / 'tokenizer.json').unlink()
    with pytest.raises(FileNotFoundError): adapter_identity(tmp_path)


def test_compare_preserves_failure_when_adapter_changes(tmp_path, monkeypatch):
    import sys
    from agentprobe import compare
    from agentprobe.tasks_io import Task
    task = Task('test', 'f_add', 'Add 1 and 1.', ['add'], [1, 1], 1, 2)
    monkeypatch.setattr(compare, 'load_test', lambda _: [task])
    snapshots = iter([{'directory': '/fixture', 'sha256': {'weights': 'before'}},
                      {'directory': '/fixture', 'sha256': {'weights': 'after'}}])
    monkeypatch.setattr(compare, 'adapter_identity', lambda _: next(snapshots))
    monkeypatch.setattr(compare, 'eval_provider', lambda *a: {'label': 'scripted', 'task_success_rate': 1.0})
    monkeypatch.setattr(sys, 'argv', ['compare', '--models', 'tuned', '--output-dir', str(tmp_path)])
    with pytest.raises(RuntimeError, match='changed'):
        compare.main()
    saved = json.loads(next(tmp_path.glob('*/comparison.json')).read_text())
    assert saved['status'] == 'failed'
    assert saved['adapter_identity']['sha256']['weights'] == 'before'
    assert saved['adapter_identity_after']['sha256']['weights'] == 'after'
