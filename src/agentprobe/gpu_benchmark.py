"""Development-workload GPU benchmark; run without a profiler for timing."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import uuid


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_tasks(tasks, per_family, seed):
    if per_family < 1:
        raise ValueError('per-family must be positive')
    groups = defaultdict(list)
    if len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError('duplicate task IDs')
    for task in tasks:
        groups[task.family].append(task)
    if not groups or any(len(g) < per_family for g in groups.values()):
        raise ValueError('not enough training tasks per family')
    rng = random.Random(seed)
    selected = []
    for family in sorted(groups):
        group = sorted(groups[family], key=lambda t: t.task_id)
        rng.shuffle(group)
        selected.extend(group[:per_family])
    rng.shuffle(selected)
    return selected


def summarize(rows):
    if not rows:
        raise ValueError('no measured runs')
    durations = sorted(r['wall_seconds'] for r in rows)
    if any(not math.isfinite(x) or x <= 0 for x in durations):
        raise ValueError('invalid duration')
    successful = sum(r['score']['task_success'] for r in rows)
    return dict(measured_runs=len(rows), successful_runs=successful,
                task_success_rate=successful / len(rows),
                median_task_seconds=statistics.median(durations),
                p95_task_seconds=durations[math.ceil(.95 * len(rows)) - 1],
                task_seconds_sum=sum(durations),
                tasks_per_timed_second=len(rows) / sum(durations),
                successful_tasks_per_timed_second=successful / sum(durations),
                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows))


def command_output(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def worker(out):
    # Import before torch/transformers, as required by this project's GPU stack.
    import unsloth
    import torch
    from contextlib import ExitStack, nullcontext
    from importlib.metadata import version
    from . import student_agent, engine, scorer
    from .tasks_io import Task
    cfg = json.loads((out / 'config.json').read_text())
    tasks = [Task(**t) for t in cfg['tasks']]
    assert torch.cuda.is_available(), 'CUDA unavailable'
    student_agent.ADAPTER_DIR = cfg['adapter_dir']
    student_agent._tuned = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    model, _ = student_agent._load_tuned()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    with ExitStack() as stack:
        experiment = None
        setup_start = time.perf_counter()
        if cfg.get('cache_quant_scales', False):
            from .quant_scale_experiment import QuantScaleExperiment
            experiment = stack.enter_context(QuantScaleExperiment(model))
            print('PASS: cached GEMV preflight', experiment.report(), flush=True)
        setup_seconds = time.perf_counter() - setup_start
        metadata = dict(load_seconds=load_seconds, gpu=torch.cuda.get_device_name(0),
                        cuda_runtime=torch.version.cuda,
                        packages={p: version(p) for p in ['torch','unsloth','transformers','peft']})
        metadata['variant'] = 'cached_quant_scales' if experiment else 'original'
        metadata['optimization_setup_seconds'] = setup_seconds
        if experiment:
            metadata['scale_cache_preflight'] = experiment.report()
        (out / 'environment.json').write_text(json.dumps(metadata, indent=2))
        provider = student_agent.StudentProvider()
        rows = []
        with (out / 'trajectories.jsonl').open('w') as evidence:
            for phase, rounds in [('warmup', 1), ('measured', cfg['repeats'])]:
                for repeat in range(rounds):
                    for task in tasks:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        marker = torch.cuda.nvtx.range(f'agentprobe/{phase}/{task.task_id}') if cfg['profile'] else nullcontext()
                        with marker:
                            start = time.perf_counter()
                            traj = engine.run(task.task_id, task.question, provider,
                                              max_seconds=cfg['task_timeout'])
                            fatal = traj.termination in {'timeout','provider_error','invalid_action_shape'}
                            if not fatal:
                                torch.cuda.synchronize()
                            elapsed = time.perf_counter() - start
                        row = dict(phase=phase, repeat=repeat, task_id=task.task_id,
                                   family=task.family, wall_seconds=elapsed,
                                   trajectory=traj.to_dict(), score=scorer.score_one(traj, task),
                                   peak_allocated_bytes=None if fatal else torch.cuda.max_memory_allocated(),
                                   peak_reserved_bytes=None if fatal else torch.cuda.max_memory_reserved())
                        evidence.write(json.dumps(row) + '\n')
                        evidence.flush()
                        if fatal:
                            raise RuntimeError(f'inference failed: {traj.termination}; evidence preserved')
                        if phase == 'measured':
                            rows.append(row)
                    print(f'{phase} round {repeat + 1} complete', flush=True)
        for name, expected in cfg['adapter_hashes'].items():
            if digest(Path(cfg['adapter_dir']) / name) != expected:
                raise RuntimeError('adapter files changed during benchmark')
        if experiment:
            if experiment.calls[0] == 0:
                raise RuntimeError('optimized GEMV path was not exercised')
            metadata['scale_cache_final'] = experiment.report()
        result = dict(status='complete', mode='profile' if cfg['profile'] else 'baseline',
                      **metadata, **summarize(rows))
        (out / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-dir', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--adapter-dir', type=Path)
    p.add_argument('--output-dir', type=Path, default=Path('runs/gpu-benchmark'))
    p.add_argument('--per-family', type=int, default=2)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--task-timeout', type=float, default=60)
    p.add_argument('--suite-timeout', type=float, default=1800)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--cache-quant-scales', action='store_true',
                   help='experimental process-local NF4 scale cache; verifies GEMV outputs first')
    a = p.parse_args()
    if a.worker_dir:
        worker(a.worker_dir)
        return
    if a.data_dir is None or a.adapter_dir is None:
        p.error('--data-dir and --adapter-dir are required')
    if a.per_family < 1 or a.repeats < 1 or any(not math.isfinite(v) or v <= 0 for v in [a.task_timeout,a.suite_timeout]):
        p.error('counts and timeouts must be positive and finite')
    from .tasks_io import load_train
    tasks = select_tasks(load_train(a.data_dir), a.per_family, a.seed)
    adapter = a.adapter_dir.resolve()
    names = ['adapter_model.safetensors','adapter_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja']
    hashes = {name: digest(adapter / name) for name in names}
    out = a.output_dir.resolve() / uuid.uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    cfg = dict(adapter_dir=str(adapter), adapter_hashes=hashes,
               train_file_sha256=digest(a.data_dir / 'train_tasks.jsonl'),
               tasks=[asdict(t) for t in tasks], per_family=a.per_family,
               repeats=a.repeats, seed=a.seed, task_timeout=a.task_timeout,
               suite_timeout=a.suite_timeout, profile=a.profile,
               cache_quant_scales=a.cache_quant_scales,
               scope='Development training-pool workload; not held-out accuracy or production throughput.',
               git_commit=command_output(['git','rev-parse','HEAD']),
               git_status=command_output(['git','status','--short']),
               source_hashes={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')},
               nvidia_smi=command_output(['nvidia-smi']))
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    (out / 'result.json').write_text(json.dumps({'status':'running'}))
    print('Evidence:', out, flush=True)
    try:
        subprocess.run([sys.executable,'-m','agentprobe.gpu_benchmark','--worker-dir',str(out)],
                       check=True, timeout=a.suite_timeout)
    except (subprocess.SubprocessError, KeyboardInterrupt) as exc:
        (out / 'result.json').write_text(json.dumps({'status':'failed','error':str(exc)}, indent=2))
        raise


if __name__ == '__main__':
    main()
