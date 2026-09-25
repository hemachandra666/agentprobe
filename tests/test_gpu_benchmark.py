from types import SimpleNamespace
import pytest
from agentprobe.gpu_benchmark import select_tasks, summarize


def test_workload_is_balanced_and_order_independent():
    tasks = [SimpleNamespace(task_id=f'{f}{i}',family=f) for f in ['a','b'] for i in range(5)]
    chosen = select_tasks(tasks,2,73)
    assert [t.task_id for t in chosen] == [t.task_id for t in select_tasks(tasks[::-1],2,73)]
    assert sum(t.family == 'a' for t in chosen) == 2
    assert len({t.task_id for t in chosen}) == 4
    with pytest.raises(ValueError): select_tasks(tasks,6,73)
    with pytest.raises(ValueError): select_tasks(tasks + tasks[:1],2,73)


def test_summary_includes_failed_tasks_and_nearest_rank_p95():
    rows = [dict(wall_seconds=i,score={'task_success':i<20},peak_allocated_bytes=i,peak_reserved_bytes=i+10) for i in range(1,21)]
    result = summarize(rows)
    assert result['p95_task_seconds'] == 19
    assert result['median_task_seconds'] == 10.5
    assert result['task_success_rate'] == 19/20
    assert result['tasks_per_timed_second'] == 20/210
    assert result['successful_tasks_per_timed_second'] == 19/210
    assert result['peak_allocated_bytes'] == 20
    with pytest.raises(ValueError): summarize([])
    rows[0]['wall_seconds'] = float('nan')
    with pytest.raises(ValueError): summarize(rows)
