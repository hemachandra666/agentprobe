import copy
import json
from pathlib import Path
import pytest
from agentprobe.refund_workflow import ReferenceProvider, development_cases, run_case
from agentprobe.refund_fresh import fresh_cases
from agentprobe.refund_gate import check


@pytest.mark.parametrize('case', fresh_cases(), ids=lambda c: c.case_id)
def test_fresh_explicit_labels_and_gate(case):
    row = run_case(case, ReferenceProvider())
    assert row['score']['task_success']
    before = copy.deepcopy(row)
    assert check(case.question, row)['action'] == 'allow'
    assert row == before


def test_fresh_questions_and_ids_are_separate():
    fresh, dev = fresh_cases(), development_cases()
    assert len(fresh) == len({c.case_id for c in fresh}) == 28
    assert not ({c.question for c in fresh} & {c.question for c in dev})
    assert not ({c.order_id for c in fresh if c.order_id} & {c.order_id for c in dev if c.order_id})


def test_gate_rejects_unsupported_policy_and_incomplete_run():
    c = fresh_cases()[0]
    row = run_case(c, ReferenceProvider())
    row['steps'][-1]['result']['policy']['version'] = 'unknown'
    assert check(c.question, row)['action'] == 'block'
    row['termination'] = 'timeout'
    assert check(c.question, row)['reason'] == 'incomplete_run'


def test_gate_blocks_wrong_final_and_missing_evidence():
    c = fresh_cases()[0]
    row = run_case(c, ReferenceProvider())
    row['final']['decision'] = 'ineligible'
    assert check(c.question, row)['action'] == 'block'
    row['steps'] = []
    assert check(c.question, row)['action'] == 'block'


def test_all_recorded_prompts_are_audited_without_modification():
    from agentprobe.refund_workflow import Environment, score
    root = Path(__file__).resolve().parents[1] / 'docs/results/refund-development'
    cases = {c.case_id: c for c in development_cases()}
    for folder, expected in [('baseline', 5), ('prompt-v2', 1), ('prompt-v3', 9)]:
        rows = [json.loads(s) for s in (root / folder / 'trajectories.jsonl').read_text().splitlines()]
        assert len(rows) == len({r['case_id'] for r in rows}) == 16
        successes = 0
        for row in rows:
            case = cases[row['case_id']]
            env = Environment(case)
            for step in row['steps']:
                assert env.execute(step['tool'], step['args']) == step['result']
            actual = score(case, row)['task_success']
            assert actual == row['score']['task_success']
            before = copy.deepcopy(row)
            assert (check(case.question, row)['action'] == 'allow') == actual
            assert row == before
            successes += actual
        assert successes == expected
