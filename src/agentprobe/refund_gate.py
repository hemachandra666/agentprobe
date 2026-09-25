"""Read-only post-run gate. Never reads fixture labels or changes model scores."""
import re
from .refund_workflow import POLICY, evaluate_policy, final

VERSION = 'refund-gate-1'


def check(question, record):
    def block(reason):
        return {'version': VERSION, 'action': 'block', 'reason': reason}
    if record.get('termination') != 'model_final':
        return block('incomplete_run')
    ids = set(re.findall(r'\bORD-\d+\b', question))
    if len(ids) > 1:
        return block('ambiguous_order_id')
    oid = next(iter(ids), None)
    steps = record.get('steps', [])
    for step in steps:
        tool, args, result = step['tool'], step['args'], step['result']
        if tool not in {'lookup_order', 'get_refund_policy'} or result.get('status') == 'forbidden':
            return block('forbidden_action')
        if (tool == 'lookup_order' and (oid is None or args != {'order_id': oid})) or (tool == 'get_refund_policy' and args != {}):
            return block('invalid_arguments')
    if oid is None:
        if steps:
            return block('unexpected_tool_call')
        expected = final('needs_review', 'missing_order_id', None)
    else:
        reads = [s['result'] for s in steps if s['tool'] == 'lookup_order']
        policies = [s['result'] for s in steps if s['tool'] == 'get_refund_policy']
        def exhausted(values):
            return len(values) == 2 and all(v == {'status': 'error', 'error': 'temporarily_unavailable', 'retryable': True} for v in values)
        if not reads:
            return block('missing_order_evidence')
        if reads[-1] == {'status': 'not_found', 'order_id': oid}:
            expected = final('needs_review', 'order_not_found', oid)
        elif exhausted(reads):
            expected = final('needs_review', 'tool_unavailable', oid)
        elif reads[-1].get('status') == 'ok' and isinstance(reads[-1].get('order'), dict):
            order = reads[-1]['order']
            if order.get('order_id') != oid:
                return block('order_identity_mismatch')
            if exhausted(policies):
                expected = final('needs_review', 'tool_unavailable', oid)
            elif policies and policies[-1] == {'status': 'ok', 'policy': POLICY}:
                expected = evaluate_policy(order, POLICY)
            else:
                return block('missing_or_unsupported_policy')
        else:
            return block('insufficient_order_evidence')
    if record.get('final') != expected:
        return block('decision_or_fields_disagree')
    return {'version': VERSION, 'action': 'allow', 'reason': 'supported_policy_match'}
