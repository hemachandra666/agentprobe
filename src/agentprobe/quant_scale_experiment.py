"""Opt-in, process-local experiment for immutable Unsloth NF4 inference.

No dependency files or model weights are edited. This is deliberately restricted
by the AST fingerprint of the inspected fast_gemv implementation. Never use it
for training, concurrent inference, or a model whose quantization state changes.
"""
import ast
import copy
import hashlib
import inspect
import json
import textwrap

EXPECTED_AST = 'fe8726dc6cd21cb252097ee74ab0a3c0358624a6b5368b1113b75baf971bc5e3'


def _canonical_ast(value):
    """Stable on Python 3.11/3.12; ignore only the new empty type_params field."""
    if isinstance(value, ast.AST):
        return [type(value).__name__, [
            [name, _canonical_ast(item)] for name, item in ast.iter_fields(value)
            if not (name == 'type_params' and item == [])
        ]]
    if isinstance(value, list):
        return [_canonical_ast(item) for item in value]
    return value


def fingerprint(source):
    tree = ast.parse(textwrap.dedent(source))
    payload = json.dumps(_canonical_ast(tree), separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_cached_gemv(source, namespace, cache, calls, *, expected=EXPECTED_AST):
    """Keep the original arithmetic; execute scale reconstruction on cache miss."""
    source = textwrap.dedent(source)
    observed = fingerprint(source)
    if observed != expected:
        raise RuntimeError(
            'Unsloth fast_gemv source differs from the inspected version; '
            f'cache experiment refused (expected {expected}, observed {observed})')
    tree = ast.parse(source)
    fn = tree.body[0]
    allocation = next(n for n in fn.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'df')
    block = next(n for n in fn.body if isinstance(n, ast.With))
    # Fingerprint validation above binds these positions to the inspected source.
    reconstruction = copy.deepcopy(block.body[:2])
    branch = ast.parse('''
if cached is None:
    pass
else:
    df = cached[1]
''').body[0]
    branch.body = [copy.deepcopy(allocation), *reconstruction,
                  ast.parse('_agentprobe_cache[id(quant_state)] = (quant_state, df)').body[0]]
    block.body = [branch, *block.body[2:]]
    index = fn.body.index(allocation)
    fn.body[index:index + 1] = ast.parse('''
cached = _agentprobe_cache.get(id(quant_state))
_agentprobe_calls[0] += 1
''').body
    ast.fix_missing_locations(tree)
    scope = dict(namespace, _agentprobe_cache=cache, _agentprobe_calls=calls)
    exec(compile(tree, '<agentprobe-quant-scale-experiment>', 'exec'), scope)
    return scope['fast_gemv']


class QuantScaleExperiment:
    def __init__(self, model):
        self.model = model
        self.cache = {}
        self.calls = [0]
        self.checked = 0
        self.installed = False

    def __enter__(self):
        import torch
        import unsloth.kernels.utils as utils
        if self.model.training:
            raise RuntimeError('scale cache requires an evaluation-only model')
        self.original = utils.fast_gemv
        self.target = utils.fast_linear_forward.__globals__
        if self.target.get('fast_gemv') is not self.original:
            raise RuntimeError('unexpected Unsloth fast_linear_forward binding')
        source = inspect.getsource(self.original)
        self.source_hash = fingerprint(source)
        self.cached = build_cached_gemv(source, self.original.__globals__, self.cache, self.calls)
        weights = {}
        for module in self.model.modules():
            weight = getattr(module, 'weight', None)
            state = getattr(weight, 'quant_state', None)
            if state is not None:
                if getattr(state, 'quant_type', None) != 'nf4' or getattr(state, 'state2', None) is None:
                    raise RuntimeError('only nested NF4 quantization is supported')
                if weight.device.type != 'cuda' or weight.device.index != 0:
                    raise RuntimeError('experiment supports a single model on CUDA device 0')
                weights[id(state)] = (weight, state)
        if not weights:
            raise RuntimeError('no NF4 weights found')
        # Test cache miss AND reuse for every quantized weight, outside task timing.
        # fork_rng preserves the caller's random state.
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(73)
            for weight, state in weights.values():
                shape = (1, 1, state.shape[1])
                for random_input in (False, True):
                    x = (torch.randn(shape, device=weight.device, dtype=state.dtype)
                         if random_input else torch.zeros(shape, device=weight.device, dtype=state.dtype))
                    torch.cuda.synchronize()
                    reference = self.original(x, weight, state)
                    torch.cuda.synchronize()
                    candidate = self.cached(x, weight, state)
                    torch.cuda.synchronize()
                    if reference.dtype != candidate.dtype or not torch.equal(reference, candidate):
                        raise RuntimeError('cached GEMV differs from original; experiment refused')
                    self.checked += 1
        if len(self.cache) != len(weights):
            raise RuntimeError('scale-cache preflight coverage mismatch')
        self.calls[0] = 0
        self.target['fast_gemv'] = self.cached
        self.installed = True
        return self

    def report(self):
        return dict(source_ast_sha256=self.source_hash,
                    checked_vectors=self.checked,
                    cached_quantization_states=len(self.cache),
                    scale_cache_bytes=sum(df.numel() * df.element_size() for _, df in self.cache.values()),
                    inference_gemv_calls=self.calls[0],
                    preflight='exact tensor equality on zero and random vectors per NF4 weight')

    def __exit__(self, *args):
        if self.installed:
            self.target['fast_gemv'] = self.original
            self.installed = False
        self.cache.clear()
