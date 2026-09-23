"""GPU smoke: load student, terminate a deliberately stalled worker, restart."""
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from . import engine, scorer
from .process_provider import ProcessProvider


class SmokeStudent:
    def __init__(self):
        import unsloth
        from .student_agent import StudentProvider
        self.provider = StudentProvider()

    def act(self, question, history):
        if question == '__controlled_stall__':
            import time
            time.sleep(60)
        return self.provider.act(question, history)


def main():
    task = SimpleNamespace(task_id='timeout_smoke', question='Add 3 and 4.',
                           answer=7.0, reference_tools=['add'], min_steps=1)
    record = {'status': 'running', 'scope': 'Controlled stall in a model-loaded worker, not a simulated CUDA kernel hang.'}
    out = Path('runs') / ('timeout-smoke-' + uuid.uuid4().hex)
    out.mkdir(parents=True)
    try:
        with ProcessProvider(SmokeStudent) as worker:
            warm = engine.run(task.task_id, task.question, worker, max_seconds=300)
            record['warm_score'] = scorer.score_one(warm, task)
            assert record['warm_score']['task_success'], 'Warm-up task failed'
            record['terminated_pid'] = worker.process.pid
            stalled = engine.run('stall', '__controlled_stall__', worker, max_seconds=.5)
            record['termination'] = stalled.termination
            record['worker_exited'] = not worker.process.is_alive()
            record['exitcode'] = worker.process.exitcode
            assert stalled.termination == 'timeout' and record['worker_exited']
        with ProcessProvider(SmokeStudent) as replacement:
            result = engine.run(task.task_id, task.question, replacement, max_seconds=300)
            record['restart_pid'] = replacement.process.pid
            record['restart_score'] = scorer.score_one(result, task)
            assert record['restart_score']['task_success'], 'Restart task failed'
        record['replacement_exited'] = not replacement.process.is_alive()
        record['replacement_exitcode'] = replacement.process.exitcode
        assert record['replacement_exited']
        assert record['replacement_exitcode'] == 0, 'Replacement required forced shutdown'
        record['status'] = 'passed'
        print('PASS: model-loaded worker terminated; fresh worker completed task and exited.')
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print('Evidence:', out / 'result.json')


if __name__ == '__main__':
    main()
