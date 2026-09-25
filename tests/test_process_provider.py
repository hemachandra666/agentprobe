import os
import time
import pytest
from agentprobe import engine
from agentprobe.process_provider import ProcessProvider


class Calculator:
    def act(self, question, history):
        if question == 'hang':
            time.sleep(30)
        if question == 'crash':
            os._exit(3)
        if question == 'error':
            raise ValueError('test failure')
        if not history:
            return engine.Action('add', {'a': 3, 'b': 4}, None, 'add(a=3,b=4)')
        return engine.Action(None, None, '7', 'Answer: 7')


class Broken:
    def __init__(self):
        raise ValueError('construction failed')


def test_reuse_and_normal_cleanup():
    with ProcessProvider(Calculator) as p:
        for _ in range(2):
            t = engine.run('x', 'add', p, max_seconds=5)
            assert t.final_answer == '7'
            assert len(t.steps) == 1
            pid = p.process.pid
        assert p.process.pid == pid
    assert not p.process.is_alive()
    assert p.process.exitcode == 0
    p.close()  # repeated close is safe


@pytest.mark.parametrize('question,termination', [('hang','timeout'), ('crash','provider_error'), ('error','provider_error')])
def test_failure_reaped_and_fresh_worker_succeeds(question, termination):
    with ProcessProvider(Calculator) as p:
        p.act_before_deadline('add', [], 5)
        t = engine.run('x', question, p, max_seconds=.1)
        assert t.termination == termination
        assert not p.process.is_alive()
        assert p.closed
    with ProcessProvider(Calculator) as replacement:
        assert engine.run('y','add',replacement,max_seconds=5).final_answer == '7'


def test_constructor_error():
    with ProcessProvider(Broken) as p:
        t = engine.run('x','add',p,max_seconds=5)
        assert t.termination == 'provider_error'
        assert 'construction failed' in t.steps[0].result
        assert not p.process.is_alive()


def test_caller_exception_cleanup():
    with pytest.raises(ValueError):
        with ProcessProvider(Calculator) as p:
            p.act_before_deadline('add', [], 5)
            raise ValueError('caller fails')
    assert not p.process.is_alive()
    assert p.process.exitcode == 0

@pytest.mark.parametrize('factory,status', [(Calculator, 'complete'), (Broken, 'failed')])
def test_comparison_cli_owns_worker_and_preserves_evidence(tmp_path, monkeypatch, factory, status):
    import json
    import sys
    import multiprocessing
    from agentprobe import compare
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    for name in ['adapter_model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']:
        (adapter / name).write_text('scripted fixture')
    (adapter / 'adapter_config.json').write_text('{}')
    monkeypatch.setattr(compare.student_agent, 'ADAPTER_DIR', str(adapter))
    monkeypatch.setattr(compare.student_agent, 'StudentProvider', factory)
    monkeypatch.setattr(sys, 'argv', ['compare', '--models', 'tuned', '--max-tasks', '1', '--output-dir', str(tmp_path)])
    before = {p.pid for p in multiprocessing.active_children()}
    if status == 'failed':
        with pytest.raises(RuntimeError):
            compare.main()
    else:
        compare.main()
    summary = json.loads(next(tmp_path.glob('*/comparison.json')).read_text())
    assert summary['status'] == status
    assert summary['student_execution'] == 'spawn_process'
    trace = json.loads(next(tmp_path.glob('*/comparison_runs.jsonl')).read_text())
    assert trace['termination'] == ('provider_error' if status == 'failed' else 'model_final')
    assert {p.pid for p in multiprocessing.active_children()} == before
