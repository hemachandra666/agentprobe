from contextlib import nullcontext
import pytest
from agentprobe.quant_scale_experiment import build_cached_gemv, fingerprint, QuantScaleExperiment

SOURCE = '''
def fast_gemv(X, W, quant_state, out=None):
    offset = 2
    df = allocate()
    with device():
        reconstruct(df, quant_state)
        df += offset
        result = df.value * X
    return result
'''


class Buffer:
    def __init__(self):
        self.value = 0

    def __iadd__(self, value):
        self.value += value
        return self


def test_cache_reuses_scales_but_not_outputs_and_retains_state_identity():
    reconstructions = []
    def reconstruct(df, state):
        reconstructions.append(state)
        df.value = state['scale']
    scope = dict(allocate=Buffer, device=nullcontext, reconstruct=reconstruct)
    exec(SOURCE, scope)
    cache, calls = {}, [0]
    cached = build_cached_gemv(SOURCE, scope, cache, calls, expected=fingerprint(SOURCE))
    a, b = {'scale': 3}, {'scale': 9}
    assert cached(2, None, a) == 10
    assert cached(4, None, a) == 20
    assert cached(2, None, b) == 22
    assert reconstructions == [a, b]
    assert cache[id(a)][0] is a
    assert calls[0] == 3
    assert scope['fast_gemv'](4, None, a) == 20


def test_changed_dependency_source_is_rejected():
    with pytest.raises(RuntimeError, match='source differs'):
        build_cached_gemv(SOURCE, {}, {}, [0])
    with pytest.raises(RuntimeError, match='source differs'):
        build_cached_gemv(SOURCE.replace('offset = 2', 'offset = 3'), {}, {}, [0],
                          expected=fingerprint(SOURCE))


def test_process_local_binding_is_restored_even_after_error():
    experiment = QuantScaleExperiment(None)
    original, replacement = object(), object()
    experiment.original = original
    experiment.target = {'fast_gemv': replacement}
    experiment.installed = True
    experiment.cache[1] = object()
    experiment.__exit__(RuntimeError, RuntimeError('failure'), None)
    assert experiment.target['fast_gemv'] is original
    assert not experiment.installed
    assert not experiment.cache


def test_fingerprint_normalizes_only_empty_type_parameters():
    import ast
    import copy
    from agentprobe.quant_scale_experiment import _canonical_ast
    tree = ast.parse(SOURCE)
    alternate = copy.deepcopy(tree)
    fn = alternate.body[0]
    # Simulate Python 3.12's extra field even when this test runs on 3.11.
    fn._fields = tuple(f for f in fn._fields if f != 'type_params') + ('type_params',)
    fn.type_params = []
    expected = _canonical_ast(tree)
    assert _canonical_ast(alternate) == expected
    fn.type_params = [ast.Name(id='T', ctx=ast.Load())]
    assert _canonical_ast(alternate) != expected
    assert fingerprint(SOURCE) == fingerprint('# comment\n' + SOURCE)
    assert fingerprint(SOURCE) != fingerprint(SOURCE.replace('offset = 2', 'offset = 3'))
