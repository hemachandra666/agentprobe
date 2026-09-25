import copy
import json
import pytest
from agentprobe.refund_workflow import (
    Environment, POLICY, ReferenceProvider, development_cases, evaluate_policy,
    parse_action, run_case, score,
)


@pytest.mark.parametrize('case', development_cases(), ids=lambda c: c.family)
def test_reference_harness_against_explicit_expectations(case):
    record = run_case(case, ReferenceProvider())
    assert record['score']['task_success']
    assert record['final'] == case.expected
    assert record['termination'] == 'model_final'


@pytest.mark.parametrize('raw', [
    'Answer: eligible', '[]', '{"final": {}, "tool": "lookup_order", "args": {}}',
    '{"tool":"lookup_order","tool":"issue_refund","args":{}}',
    '{"tool":"lookup_order","args":{"order_id":NaN}}',
    '{"final":{"decision":"eligible","reason":"within_window"}}',
    '{"tool":12,"args":{}}',
])
def test_parser_rejects_ambiguous_or_malformed_actions(raw):
    with pytest.raises((ValueError, TypeError)):
        parse_action(raw)


def test_correct_final_without_retrieval_fails_grounding():
    case = development_cases()[0]
    class Guess:
        def act(self, question, history):
            return json.dumps({'final': case.expected})
    result = run_case(case, Guess())
    assert result['score']['decision_correct']
    assert not result['score']['evidence_supported']
    assert not result['score']['task_success']


def test_wrong_order_and_write_tool_attempts_are_blocked():
    case = development_cases()[0]
    env = Environment(case)
    assert env.execute('lookup_order', {'order_id': 'ORD-9999'})['status'] == 'forbidden'
    assert env.execute('issue_refund', {'order_id': case.order_id})['status'] == 'forbidden'
    assert env.counts == {'lookup_order': 0, 'get_refund_policy': 0}


def test_forbidden_attempt_cannot_be_hidden_by_later_correct_answer():
    case = development_cases()[0]
    record = run_case(case, ReferenceProvider())
    record['steps'].insert(0, dict(tool='issue_refund', args={}, result={'status': 'forbidden'}))
    scored = score(case, record)
    assert scored['decision_correct'] and scored['evidence_supported']
    assert not scored['allowed_tools_only'] and not scored['task_success']


def test_unavailable_requires_two_observed_failures():
    case = next(c for c in development_cases() if c.family == 'order_unavailable')
    record = run_case(case, ReferenceProvider())
    assert len(record['steps']) == 2
    record['steps'].pop()
    assert not score(case, record)['task_success']


def test_policy_evidence_is_required_and_version_must_match():
    case = development_cases()[0]
    record = run_case(case, ReferenceProvider())
    record['final']['policy_version'] = 'invented'
    assert not score(case, record)['task_success']
    record['final'] = copy.deepcopy(case.expected)
    record['steps'] = [s for s in record['steps'] if s['tool'] != 'get_refund_policy']
    assert not score(case, record)['evidence_supported']


def test_transient_errors_do_not_leak_between_cases():
    case = next(c for c in development_cases() if c.family == 'order_retry')
    for _ in range(2):
        env = Environment(case)
        args = {'order_id': case.order_id}
        assert env.execute('lookup_order', args)['status'] == 'error'
        result = env.execute('lookup_order', args)
        result['order']['opened'] = True
        assert not env.execute('lookup_order', args)['order']['opened']


def test_bool_is_not_a_valid_delivery_age():
    case = development_cases()[0]
    order = dict(case.order, days_since_delivery=True)
    assert evaluate_policy(order, POLICY)['reason'] == 'incomplete_order'


def test_endless_invalid_responses_are_bounded_and_retained():
    class Invalid:
        def act(self, question, history):
            return 'not json'
    record = run_case(development_cases()[0], Invalid(), max_turns=3)
    assert record['termination'] == 'max_turns'
    assert len(record['responses']) == len(record['parse_errors']) == 3
    assert not record['score']['task_success']


def test_provider_failure_is_retained():
    class Broken:
        def act(self, question, history):
            raise RuntimeError('offline')
    record = run_case(development_cases()[0], Broken())
    assert record['termination'] == 'provider_error'
    assert 'offline' in record['error']
    assert not record['score']['task_success']


def test_deadline_provider_receives_positive_budget_and_timeout_is_retained():
    class Timed:
        def act_before_deadline(self, question, history, seconds):
            assert 0 < seconds <= 10
            raise TimeoutError('controlled')
    record = run_case(development_cases()[0], Timed(), max_seconds=10)
    assert record['termination'] == 'timeout'
    assert not record['score']['task_success']
