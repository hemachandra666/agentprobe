# GPU development benchmark

Run `python -m agentprobe.gpu_benchmark --help` in the GPU environment.
Required: explicit `--data-dir` and `--adapter-dir`. Only train_tasks.jsonl
is read. Defaults select two tasks per available training family, seed 73,
one warm-up pass over those tasks and three measured passes. Repeated tasks
are not independent accuracy samples. Held-out families absent from the
training pool are absent here. This workload is for development, not a new
test accuracy claim.

The parent launches one GPU-owning subprocess and enforces a whole-suite
wall-time limit (default 1800 seconds). The shared engine retains its task
budgets (60 seconds by default). Inference errors abort the suite; previously
written trajectories remain. This benchmark does not change ProcessProvider
or production evaluation. It measures the direct student provider with the
engine's thread wrapper, not per-action IPC latency of ProcessProvider.

Model load time excludes Python imports and CUDA initialization. It may include
model downloads and setup; it is not guaranteed cold-disk loading. Each task
is synchronized before and after timing. Timed intervals include the engine,
model generation and tools, but exclude scoring and evidence serialization.
Tasks-per-timed-second is sequential throughput derived from the sum of these
intervals, not concurrent serving throughput. P95 uses nearest rank. Failed
logical tasks remain in timing and success denominators. Infrastructure failures
produce a failed experiment, not a successful partial benchmark.

Peak allocated/reserved bytes are PyTorch allocator values in the GPU process,
including resident model memory, reset before each task. Reserved memory may
include pools retained from warm-up. These are not total device VRAM, CUDA
context memory or memory used by other processes. Model-loading peak memory
is not included in the measured-task peaks.

The unique output directory contains config.json (tasks, hashes, software source
hashes, git state, nvidia-smi snapshot), environment.json, trajectories.jsonl
and result.json. Baseline metadata and traces include unsuccessful warm-up tasks;
only measured rounds contribute to the summary. Run with the laptop plugged in,
consistent power settings, and no competing GPU workload. Repeat complete runs
when estimating performance variability.

Use --profile only for diagnostic runs under Nsight. It adds NVTX ranges for
warmup/measured tasks; all such results are labelled profile. Do not compare
profiled timing against unprofiled timing. NVTX task ranges are on the engine
thread while generation executes in a helper thread; use timestamps in the
system timeline, not same-thread kernel attribution assumptions. The raw kernel
summary includes loading and warm-up.

This implementation has CPU checks; actual GPU benchmark validation is pending.
No latency, throughput, or memory savings are claimed.
