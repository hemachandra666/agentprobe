# GPU quantization-scale cache results

The optional quantization-scale cache reduced warmed task time on this measured
AgentProbe development workload. In a four-run original/cached/cached/original
confirmation batch, pooled median task latency was 15.6% lower and task
throughput was 18.7% higher. Peak PyTorch allocated memory increased by 62.7 MiB.
All recorded task behavior matched across variants. These are descriptive
measurements on one RTX 5080 Laptop GPU, not a general speedup guarantee.

## Workload and method

- V2 released adapter SHA-256:
  `8ad64ebe7be9ac555c5bcca78698bfa7c51f9a0c84cb048d655883866537b8ba`.
- 18 development tasks: two per available training family, selection seed 73.
- Each process warmed all 18 tasks, then measured three repetitions (54 runs).
- Four processes in original/cached/cached/original order; 108 measured runs per
  variant. These are repetitions of 18 questions, not 108 independent questions.
- NVIDIA GeForce RTX 5080 Laptop GPU, CUDA runtime 12.8, PyTorch 2.8.0+cu128,
  Unsloth 2026.9.8, Transformers 5.5.0, PEFT 0.21.0.
- No Nsight profiling during these timing runs. Loading and optimization setup
  are excluded from task timing and recorded separately.
- Each cached process passed 340 exact tensor comparisons on 170 quantization
  states, exercising creation and reuse; setup took approximately 0.25 seconds.
- The optimization is benchmark-only and opt-in. Production student inference
  retains its existing behavior. No adapter or installed dependency is rewritten.

## Confirmation results

The pooled median and nearest-rank p95 below are calculated from all 108 measured
latencies per variant. Throughput is run count divided by summed measured task
time; it excludes loading, warmup, scoring, and evidence-writing overhead.

| Metric | Original | Cached scales | Change |
|---|---:|---:|---:|
| Median task latency | 2.341 s | 1.976 s | 15.6% lower |
| p95 task latency | 5.541 s | 4.503 s | 18.7% lower |
| Summed measured task time | 295.577 s | 249.015 s | 15.8% lower |
| Tasks per timed second | 0.3654 | 0.4337 | 18.7% higher |
| Successful tasks per timed second | 0.3451 | 0.4096 | 18.7% higher |
| Successful measured runs | 102/108 | 102/108 | Unchanged |
| Peak PyTorch allocated bytes | 1,760,668,160 | 1,826,384,384 | +62.7 MiB |

Individual process totals, in execution order:

| Variant | Run ID | Measured task time |
|---|---|---:|
| Original | 86176e78a6fb4a1ba0ec924394a5fade | 142.570 s |
| Cached | 7a9769f7be8f4994b2a1deb0a0f4fb67 | 124.379 s |
| Cached | b0c1e1dbd3dd40ad862b88c049813b4a | 124.636 s |
| Original | ad7e5468e82b41cbb4e6634236932d3e | 153.007 s |

Both cached processes completed the workload faster than either original
process. Original timing varied between runs; thermal state and clocks were not
continuously recorded or controlled. ABBA ordering helps inspect order effects
but does not eliminate all confounding. No confidence interval or cross-machine
performance claim is made.

## Correctness and evidence audit

All 288 trajectories (216 measured and 72 warmup) were rescored. Per-run summaries
were recomputed. Task selections, adapter hashes, training-data hash, benchmark
settings, source hashes, and recorded software versions matched across processes.
Responses, tool steps including arguments and results, final answers, and
termination reasons matched for every corresponding trajectory. The known
incorrect task remained incorrect; success-rate equality is not the only check.

- [Computed audit summary](results/gpu-scale-cache/confirmation_audit.json)
- [Raw confirmation evidence](results/gpu-scale-cache/confirmation/): each run
  includes config.json, environment.json, result.json, and trajectories.jsonl.
- [Experiment implementation and guards](QUANT_SCALE_EXPERIMENT.md)
- [Benchmark methodology](GPU_BENCHMARK.md)

Audit summary percentages are `100 * (cached / original - 1)`. Negative latency
changes indicate reductions. Peak memory is allocator memory, not whole-device
VRAM. The earlier exploratory baseline/candidate comparison is not pooled into
this confirmation result.

## Interpretation and decision

Keep the cache as an optional, version-guarded inference experiment. It avoids
reconstructing invariant nested quantization scales during single-token GEMV,
while leaving prompt dequantization and the matrix calculation unchanged. A
profile motivated this target; a second optimized trace was not collected, so
this report does not claim a measured reduction in kernel count.

The experiment does not establish accuracy on a new held-out dataset, teacher
versus student GPU savings, production throughput, training acceleration, or
behavior on other models. The benchmark uses the direct student provider rather
than production ProcessProvider IPC. Cached states must remain immutable, and
inference must remain sequential. No promotion to production defaults is implied.
