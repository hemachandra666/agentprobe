"""Run the synthetic refund development suite, with explicit harness/model modes."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import uuid

from .refund_workflow import VERSION, PROMPT_VERSION, SCORER_VERSION, SYSTEM, POLICY, ReferenceProvider, development_cases, run_case


class OllamaRefundProvider:
    def __init__(self, model):
        import ollama
        self.model = model
        self.client = ollama.Client(host='http://localhost:11434', timeout=30)

    def act(self, question, history):
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': question}]
        for event in history:
            if event['kind'] == 'assistant':
                messages.append({'role': 'assistant', 'content': event['content']})
            else:
                messages.append({'role': 'user', 'content': json.dumps(event)})
        response = self.client.chat(model=self.model, messages=messages, format='json',
                                    options={'temperature': 0, 'seed': 42, 'num_predict': 256})
        return response['message']['content'] or ''


def git_output(*args):
    try:
        result = subprocess.run(['git', *args], text=True, capture_output=True, timeout=10)
        return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true', help='scripted harness validation, NOT model evaluation')
    mode.add_argument('--model', help='local Ollama model tag; e.g. qwen2.5:7b')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/refund-development'))
    parser.add_argument('--task-timeout', type=float, default=120)
    parser.add_argument('--suite', choices=['development', 'fresh'], default='development')
    args = parser.parse_args()
    if not math.isfinite(args.task_timeout) or args.task_timeout <= 0:
        parser.error('--task-timeout must be positive and finite')
    from .refund_gate import check, VERSION as GATE_VERSION
    from .refund_fresh import fresh_cases, VERSION as FRESH_VERSION
    cases = fresh_cases() if args.suite == 'fresh' else development_cases()
    suite_version = FRESH_VERSION if args.suite == 'fresh' else VERSION
    out = args.output_dir.resolve() / uuid.uuid4().hex
    out.mkdir(parents=True)
    kind = 'scripted_harness_check' if args.smoke else 'model_synthetic_evaluation'
    manifest = dict(suite_version=suite_version, prompt_version=PROMPT_VERSION,
                    scorer_version=SCORER_VERSION, kind=kind, model=args.model,
                    gate_version=GATE_VERSION, policy=POLICY, system_prompt=SYSTEM, cases=[asdict(c) for c in cases],
                    max_turns=8, task_timeout=args.task_timeout,
                    git_commit=git_output('rev-parse', 'HEAD'), git_status=git_output('status', '--short'),
                    source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ['refund_workflow.py', 'refund_benchmark.py', 'process_provider.py', 'refund_gate.py', 'refund_fresh.py']})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    summary = dict(status='running', kind=kind, suite_version=suite_version, planned_cases=len(cases),
                   prompt_version=PROMPT_VERSION, scorer_version=SCORER_VERSION,
                   metric_counts={k: 0 for k in ['decision_correct', 'reason_correct',
                       'order_id_correct', 'policy_version_correct', 'final_exact_match',
                       'evidence_supported', 'allowed_tools_only']},
                   gate_counts={k: 0 for k in ['allowed', 'blocked', 'incorrect_allowed', 'correct_blocked']},
                   completed_cases=0, successful_cases=0, unattempted_cases=[c.case_id for c in cases])
    summary_path = out / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print('Evidence:', out, flush=True)
    try:
        from contextlib import nullcontext
        from functools import partial
        from .process_provider import ProcessProvider
        if args.smoke:
            context = nullcontext(ReferenceProvider())
        else:
            import ollama
            client = ollama.Client(host='http://localhost:11434', timeout=30)
            installed = client.list()
            details = client.show(args.model)
            def serialize(value):
                return value.model_dump(mode='json') if hasattr(value, 'model_dump') else value
            (out / 'ollama_environment.json').write_text(json.dumps(
                {'client_version': version('ollama'), 'installed_models': serialize(installed),
                 'selected_model_details': serialize(details)}, indent=2, default=str) + '\n')
            context = ProcessProvider(partial(OllamaRefundProvider, args.model))
        with context as provider, (out / 'trajectories.jsonl').open('w') as evidence:
            for i, case in enumerate(cases):
                record = run_case(case, provider, max_seconds=args.task_timeout)
                record['gate'] = check(case.question, record)
                allowed = record['gate']['action'] == 'allow'
                correct = record['score']['task_success']
                summary['gate_counts']['allowed' if allowed else 'blocked'] += 1
                summary['gate_counts']['incorrect_allowed'] += int(allowed and not correct)
                summary['gate_counts']['correct_blocked'] += int(not allowed and correct)
                evidence.write(json.dumps(record) + '\n')
                evidence.flush()
                for key in summary['metric_counts']:
                    summary['metric_counts'][key] += int(record['score'][key])
                summary['completed_cases'] += 1
                summary['successful_cases'] += int(record['score']['task_success'])
                summary['unattempted_cases'] = [c.case_id for c in cases[i + 1:]]
                summary_path.write_text(json.dumps(summary, indent=2) + '\n')
                print(case.case_id, case.family, record['termination'], record['score']['task_success'], flush=True)
                if record['termination'] in {'timeout', 'provider_error'}:
                    raise RuntimeError(f"{case.case_id}: {record['termination']}; suite stopped, evidence preserved")
        summary['status'] = 'complete'
        if args.smoke:
            summary['interpretation'] = 'Checks the harness with a deterministic reference provider; NOT model accuracy.'
            if summary['successful_cases'] != len(cases):
                raise RuntimeError('scripted harness check failed')
        else:
            summary['task_success_rate'] = summary['successful_cases'] / len(cases)
            summary['interpretation'] = 'Synthetic known-family cases; fresh instances use a frozen prompt, not a new-domain or real-customer benchmark.'
    except BaseException as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
