import json
from pathlib import Path
from agentprobe.refund_workflow import Environment, development_cases, score, run_case, ReferenceProvider


def test_recorded_baseline_preserves_strict_success_and_separates_fields():
    path = Path(__file__).resolve().parents[1] / "docs/results/refund-development/baseline/trajectories.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    cases = {c.case_id: c for c in development_cases()}
    assert len(rows) == len({r['case_id'] for r in rows}) == 16
    scores = []
    for row in rows:
        case = cases[row['case_id']]
        env = Environment(case)
        for step in row['steps']:
            assert env.execute(step['tool'], step['args']) == step['result']
        updated = score(case, row)
        assert updated['task_success'] == row['score']['task_success']
        assert updated['final_exact_match'] == row['score']['decision_correct']
        scores.append(updated)
    assert sum(s['task_success'] for s in scores) == 5
    assert sum(s['decision_correct'] for s in scores) == 14
    assert sum(s['allowed_tools_only'] for s in scores) == 15


def test_null_string_is_not_normalized_into_null():
    case = next(c for c in development_cases() if c.family == 'unknown_id')
    row = run_case(case, ReferenceProvider())
    row['final']['policy_version'] = 'null'
    updated = score(case, row)
    assert updated['decision_correct'] and updated['reason_correct']
    assert not updated['policy_version_correct']
    assert not updated['final_exact_match'] and not updated['task_success']


def test_absent_final_does_not_match_expected_null_identifiers():
    case = next(c for c in development_cases() if c.family == 'missing_id')
    row = run_case(case, ReferenceProvider())
    row.update(final=None, termination='timeout')
    updated = score(case, row)
    assert not updated['order_id_correct'] and not updated['policy_version_correct']
    assert not updated['task_success']


def test_prompt_v2_regression_is_preserved_and_scores_reproduce():
    path = Path(__file__).resolve().parents[1] / "docs/results/refund-development/prompt-v2/trajectories.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    cases = {c.case_id: c for c in development_cases()}
    assert len(rows) == len({r['case_id'] for r in rows}) == 16
    for row in rows:
        assert row['steps'] == []
        assert row['final'] == {'decision': 'needs_review', 'reason': 'missing_order_id',
                                'order_id': None, 'policy_version': None}
        assert score(cases[row['case_id']], row) == row['score']
    assert sum(row['score']['task_success'] for row in rows) == 1
