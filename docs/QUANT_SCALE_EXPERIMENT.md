# Quantization-scale cache experiment

Status: experimental and opt-in. GPU preflight and development-workload
confirmation passed; see [measured results](GPU_SCALE_CACHE_RESULTS.md).

The measured 18-task profile contains 1,942,642 GPU kernels and 1,899,242
cudaLaunchKernel calls. The supplied Unsloth fast_gemv source reconstructs
nested quantization scales before every single-token matrix-vector operation.
This experiment retains those FP32 scales per quantization-state object.
It does not cache predictions, merge LoRA weights, change precision, or change
prompts. Prompt-processing fast_dequantize remains unchanged.

## Scope and guards

Use only through gpu_benchmark with an immutable, evaluation-only model on
CUDA device 0. Do not reuse this code for training, concurrent model execution,
or mutable quantization states. The benchmark runs sequentially in a dedicated
child process. No installed dependency file or adapter file is modified.

An AST fingerprint restricts the transformation to the supplied fast_gemv
implementation. An unexpected implementation or call binding fails explicitly.
The transformation retains the original arithmetic and caches its reconstructed
scales. It changes the fast_gemv binding used by fast_linear_forward for the
lifetime of this benchmark, and restores it on normal or exceptional exit.

Before installing the binding, a GPU preflight compares original and cached
GEMV outputs for every discovered nested NF4 weight using a zero vector and a
seeded random vector. These exercise cache creation and reuse. Dtypes and tensor
elements must match exactly. This is a targeted check, not proof for every input;
full task trajectories must also be compared against the original benchmark.
The preflight preserves random state. Setup time is recorded separately and is
excluded from warmed task timings. Cache allocation persists into memory metrics.

## Run

Run without Nsight for timing, using the same power conditions and workload as
the original benchmark:

```bash
.venv-gpu/bin/python -m agentprobe.gpu_benchmark --data-dir docs/results/experiment-v2/data --adapter-dir backups/releases/agentprobe-student-v2 --per-family 2 --repeats 3 --cache-quant-scales --output-dir backups/profiling/cached-scales
```

The result's variant is cached_quant_scales; mode remains baseline for unprofiled
runs (profile for traced runs). Config records the option explicitly. Results
include preflight coverage, cache bytes, source fingerprint, and exercised GEMV
call count. A zero call count is rejected. Source hashes, adapter hashes, task
records, and failure evidence continue to be recorded by gpu_benchmark.

Expected tradeoff: additional resident GPU memory may reduce repeated allocations
and kernel launches. Speedup is unproven until measured. Do not infer it from
profiled API time or subtract overlapping durations. Before adopting any change,
compare all task responses, actions, results, terminations, and scores; aggregate
success alone can conceal changed behavior. Repeat unprofiled original/candidate
runs in alternating order before publishing a performance claim. Keep the
original model failure in the workload. This development sample is not a new
held-out accuracy evaluation or production throughput benchmark.

## Validation

CPU tests cover scale reuse across different inputs, isolation across state
objects, rejection of changed source, and restoration of the function binding.
GPU numerical preflight and end-to-end comparisons must run on the target machine.
