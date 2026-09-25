<p align="center"><img src="docs/images/banner.png" alt="AgentProbe" width="100%"></p>

# AgentProbe

AgentProbe is an experimental evaluation project for tool-using agents. It
separates task completion from correctness, preserves failure traces, and tests
whether GPU optimizations change agent behavior.

## Current experiments

| Experiment | Recorded result | Evidence |
|---|---|---|
| V2 arithmetic distillation | Tuned student: 721/760 tasks (94.87%) | [V2 protocol and results](docs/EXPERIMENT_V2.md) |
| GPU scale cache | 15.6% lower median latency; 62.7 MiB additional peak allocated memory | [Measurements and limits](docs/GPU_SCALE_CACHE_RESULTS.md) |
| Refund workflow | Model: 10/28 strict successes; checker allowed 10 and blocked 18 | [Refund results](docs/REFUND_STAGE_RESULTS.md) |

These are separate experiments, not a combined score. Refund checker blocks do
not become model successes. The GPU result covers repeated development tasks
on one laptop. This repository does not establish production readiness.

Try the refund harness without a model or GPU:

```bash
uv run --locked python -m agentprobe.refund_benchmark --smoke --suite fresh
```

This is a scripted harness check, not model accuracy. See the
[refund workflow](docs/REFUND_WORKFLOW.md) for live-model evaluation instructions.

The [Student v2 release](https://github.com/hemachandra666/agentprobe/releases/tag/student-v2.0.0)
contains the v2 adapter. The v1 instructions below preserve the earlier experiment.

## Historical v1 result

**Scorer 2.0 evaluation completed on September 22, 2026, using 759 held-out
tasks and one run per task for each model.**

| Model | Successful tasks | Task success |
|---|---:|---:|
| Qwen2.5 7B teacher | 735 / 759 | 96.84% |
| Qwen2.5 1.5B untuned student | 111 / 759 | 14.62% |
| Qwen2.5 1.5B QLoRA-tuned student | 716 / 759 | 94.33% |

Fine-tuning improved task success by **79.71 percentage points** over the untuned
baseline. The tuned student finished **2.50 percentage points** behind the teacher.

The tuned student completed every task with zero recorded parsing errors,
tool execution errors, or detected loops. Completion does not imply correctness.

On the two task families excluded from training, `f_len5_b` and `f_mul_add_mul`,
the tuned student succeeded on **367 / 400 tasks (91.75%)**.

The remaining 43 failures comprised:

- 32 runs with an incorrect final tool result.
- 11 runs with a correct final tool result followed by an incorrect final answer.

These findings apply to this synthetic arithmetic benchmark and execution protocol.

**Evidence:**
[Comparison summary](docs/results/2026-09-22/comparison.json) |
[All 2,277 execution traces](docs/results/2026-09-22/comparison_runs.jsonl)

Experiment ID: `40af95f76f6841daba30d05622990717`.

The root `comparison.json` and older traces remain historical evidence. Their
100% / 0% / 56% scores checked tool results but could accept wrong final answers.
Earlier results also suppressed repeated student calls and used a different
training conversation format. Do not cite those figures as current task success.

The current policy separates:

- `tool_result_correct`: the last successful tool result equals the reference.
- `answer_correct`: the model's explicit final answer equals the reference.
- `completed`: the run finished with a model final answer.
- `task_success`: completed, correct final answer supported by the last tool result,
  with no final failed step. A calculator task requires at least one tool call.
- Efficiency on successful eligible tasks, tool errors, parser errors, loops/cycles,
  and recovery when an error actually occurred.

The parser accepts one complete action per turn. It does not remove repeats or
scan past invalid text. The same text instructions and action-feedback engine
are used across providers; the teacher still uses Ollama's structured tool API,
while the students use text parsing. That interface difference remains a limitation.

## CPU quick start

```bash
git clone https://github.com/hemachandra666/agentprobe.git
cd agentprobe
uv sync --locked --dev
uv run --locked pytest -q
uv run --locked python -m agentprobe.check_leakage
uv run --locked python -m agentprobe.replay --input docs/results/2026-09-22/comparison_runs.jsonl
```

Run the test suite above for the current test count. The supplied dataset contains 1,441
training questions and 759 test questions, with zero exact question overlap
and zero within-split duplicates.

Replay audits saved evidence, verifies task identity and successful tool arithmetic,
and checks the matching experiment summary when available. It preserves per-family
metrics. Use `--data-dir` for an alternate dataset and `--summary` for an explicit
comparison file. It cannot recover actions omitted by legacy parsers, and replay
is not fresh inference.

The wheel includes example train/test data. Loaders prefer an explicit data
path, then `./data` if present, then the packaged examples. `task_gen` generates
working datasets in `./data`. Install-time paths are not used for outputs.

## Teacher evaluation

Install and start Ollama, then pull the teacher. A GPU is optional for Ollama,
but running these models on a CPU can be slow.

```bash
ollama pull qwen2.5:7b
uv run --locked python -m agentprobe.run_eval --max-tasks 10 --runs 1
uv run --locked python -m agentprobe.benchmark --models qwen2.5:7b --max-tasks 10
```

Each experiment writes to a new `runs/<experiment_id>/` directory. Complete
records include zero-call runs, termination, raw responses, typed errors and
scores. Failed experiments retain evidence. Summaries include per-family results.
Provider failures and timeouts abort a comparison and mark it failed; they are
not published as a completed model benchmark.

## GPU training and comparison

The revised pipeline completed model loading, training, and evaluation on an
**NVIDIA GeForce RTX 5080 Laptop GPU with 16 GB VRAM**, running WSL Linux
and Python 3.11.16.

A separate GPU environment was used:

```bash
uv venv --python 3.11 .venv-gpu

uv pip install --python .venv-gpu/bin/python \
  -e ".[gpu]" \
  "torch==2.8.0+cu128" \
  "torchvision==0.23.0+cu128" \
  "xformers==0.0.32.post2" \
  "torchao==0.13.0" \
  "unsloth==2026.9.8" \
  "unsloth-zoo==2026.9.7" \
  --default-index https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match

uv pip check --python .venv-gpu/bin/python
```

This records the installation approach used, not a complete GPU lockfile.
The tested environment also contained Transformers 5.5.0 and TRL 0.24.0.
Unpinned dependencies may resolve differently on future installations.
The index strategy considers matching packages across both listed indexes.

Unsloth warned that the installed TorchAO integration was unusable and bypassed
it. LoRA training and inference nevertheless completed successfully.

Use `.venv-gpu/bin/python` directly for GPU commands. The CPU development
environment managed by `uv run` is separate.

The experiment used these stages:

```bash
.venv-gpu/bin/python -m agentprobe.check_leakage
.venv-gpu/bin/python -m agentprobe.generate_teacher --runs 1 --max-tasks 1000
.venv-gpu/bin/python -m agentprobe.prepare_training
.venv-gpu/bin/python -m agentprobe.train_student --epochs 1
.venv-gpu/bin/python -m agentprobe.compare \
  --models teacher untuned tuned \
  --max-tasks 759 \
  --runs 1
```

Back up existing generated datasets and `student_lora/` before rerunning:
generation, preparation, and training reuse their output paths. Teacher generation
accepts `--output` (default `./teacher_data.jsonl`) and records all attempts and
accepted examples under a unique `runs/teacher-<id>/` directory beside the output.
A timeout or provider failure aborts generation without replacing the previous
dataset. Partial evidence remains available. Successful generation replaces the
requested output atomically. No clean successful examples also leaves old data intact.

The recorded teacher generation retained 656 clean successful trajectories,
producing 1,730 training and 319 validation next-action examples.

Training used the 4-bit `unsloth/Qwen2.5-1.5B-Instruct` base, LoRA rank 16,
alpha 16, and 18,464,768 trainable parameters.

An exploratory three-epoch run showed worsening validation loss after epoch 1.
A fresh one-epoch run was selected using validation loss before final-test
evaluation. It completed 217 optimizer steps in approximately 298 seconds,
with validation loss 0.09077.

`prepare_training` splits whole training questions into train/validation groups,
checks them against final-test tasks, and writes next-action examples with real
observations. Preparation validates each saved task against the canonical training
dataset, recomputes successful tool arithmetic, and rejects corrupted examples
before writing outputs. A manifest hashes both files. `train_student` rejects stale inputs.

The trainer saves the final adapter by default. Use `--select-best` in a new
output directory to select the minimum-validation-loss checkpoint, as in v2.

Keep final-test tasks out of checkpoint/hyperparameter selection. Further
changes informed by the inspected test failures require a fresh held-out
evaluation while preserving the original result.

## Download the student adapter

The experimental [Student v1.0.0 pre-release](https://github.com/hemachandra666/agentprobe/releases/tag/student-v1.0.0)
includes the LoRA adapter, tokenizer, model card, training history,
dependency snapshot, and checksums. Base-model weights are downloaded separately.

Download these two release assets into the same directory:

- `agentprobe-student-v1.tar.gz`
- `agentprobe-student-v1.tar.gz.sha256`

From that directory, verify the archive before extracting it:

```bash
sha256sum -c agentprobe-student-v1.tar.gz.sha256
```

Continue only if verification reports `OK`. Extract into a new directory:

```bash
mkdir student-v1-extracted
tar -xzf agentprobe-student-v1.tar.gz -C student-v1-extracted
(cd student-v1-extracted/agentprobe-student-v1 && sha256sum -c SHA256SUMS)
```

All ten packaged files should report `OK`.

### Load the released adapter

Use the GPU environment described above. From the AgentProbe repository,
run the following with the absolute path to your extracted adapter folder:

```bash
.venv-gpu/bin/python - /absolute/path/student-v1-extracted/agentprobe-student-v1 <<'PYTHON'
import sys
from pathlib import Path
import unsloth
from agentprobe import student_agent

adapter = Path(sys.argv[1]).resolve()
assert (adapter / "adapter_model.safetensors").is_file(), adapter
student_agent.ADAPTER_DIR = str(adapter)
student_agent._tuned = None
student_agent._load_tuned()

trajectory = student_agent.run("release_example", "Add 3 and 4.")
for step in trajectory.steps:
    print(step.tool, step.args, step.result, step.status)
print("Final answer:", trajectory.final_answer)
print("Termination:", trajectory.termination)
PYTHON
```

The GitHub release was downloaded and its archive and all ten packaged
files passed checksum verification. In a fresh Python 3.11 virtual environment,
114 pinned dependencies and a non-editable AgentProbe installation passed
dependency checks. The downloaded adapter then executed add(3, 4), returning
7 with task_success=True on an RTX 5080 Laptop GPU.

[Reproduction smoke-test evidence](docs/results/student-v1/reproduction_smoke.json)
records the adapter hash, repository commit, GPU, and scores. This verifies
one task in a fresh environment on the same machine with the existing
base-model cache. It is not a full benchmark rerun; reproduction on another
machine or without cached base-model files remains unverified.

The released weights match final checkpoint 217 (one epoch). The historical
94.33% benchmark did not record an adapter checksum, so it is not a fresh
evaluation of the downloadable package. The exact training-time base revision
remains unverified. See the packaged model card for provenance and limitations.

## Version 2 experiment

A fresh dataset uses associative/commutative expression grouping and excludes
the packaged v1 expression groups. Checkpoint selection uses validation loss.

On 760 test tasks, the teacher achieved 98.42% task success, the untuned
student 16.18%, and the tuned student 94.87%. All 2,280 saved trajectories
were replay-audited against the v2 dataset.

These results describe one synthetic arithmetic experiment. V1 and v2 use
different datasets; their headline scores are not a controlled comparison.

See [the v2 experiment details](docs/EXPERIMENT_V2.md) and
[recorded evidence](docs/results/experiment-v2/).

## Measured GPU optimization

An optional quantization-scale cache was evaluated on an NVIDIA GeForce
RTX 5080 Laptop GPU using 18 development tasks. Four unprofiled runs in
original/cached/cached/original order produced 108 measured task runs per variant.

| Metric | Original | Cached scales |
|---|---:|---:|
| Median task latency | 2.341 s | 1.976 s |
| p95 task latency | 5.541 s | 4.503 s |
| Tasks per timed second | 0.3654 | 0.4337 |
| Successful measured runs | 102/108 | 102/108 |

Observed median latency decreased by **15.6%** and task throughput increased
by **18.7%**, using **62.7 MiB additional peak PyTorch allocated memory**.
All 288 recorded trajectories, including warmup, matched across variants
in responses, tool steps, final answers, and termination.

These measurements cover repeated development tasks on one laptop. They
exclude loading and warmup and do not establish production throughput,
cross-device speedups, or savings against the teacher. The cache remains opt-in.

See the [results and limitations](docs/GPU_SCALE_CACHE_RESULTS.md),
[benchmark instructions](docs/GPU_BENCHMARK.md), and
[optimization implementation](docs/QUANT_SCALE_EXPERIMENT.md).

## Dashboard

```bash
uv run --locked streamlit run src/agentprobe/dashboard.py
```

Open http://localhost:8501.

The dashboard defaults to the published evidence when available. Expand
**Choose evaluation file** to select another experiment's `comparison.json`.

For this experiment, use:

```text
docs/results/2026-09-22/comparison.json
```

The dashboard labels historical or incomplete files and does not present them
as validated current results.

![AgentProbe evaluation dashboard](docs/images/dashboard.png)

## Limits and interpretation

- Arithmetic covers two calculator tools and synthetic integer chains. Correct
  final answers do not prove general reasoning or appropriate intermediate arguments.
- Fixed reference paths are diagnostics, not a proof that other paths are invalid.
- Untuned performance measures this exact prompt, protocol and parser. It does
  not establish that a model is generally incapable of tool use.
- An audit found 26 test questions mathematically equivalent to questions in the
  full training pool after swapping the first two operands. Of the prepared
  examples, 12 test questions have equivalents in training and 2 in validation.
  Excluding all 26 as a post-hoc sensitivity check gives tuned success of
  690/733 (94.13%); this does not replace the original benchmark. Future datasets
  should group equivalent questions before splitting.
- Exact question separation does not establish absence of semantic similarity.
  The two held-out task families account for 400 of the 759 test tasks.
- Zero tool execution errors does not mean zero parsing errors or incorrect
  answers. These metrics must be interpreted separately.
- Sampling uncertainty, multiple training seeds, broader real-world domains, and
  cross-device GPU measurements remain future work. Smaller parameter count alone
  does not demonstrate measured performance savings.
- Student comparisons use spawned inference workers that are terminated and joined
  on timeout and closed between models. Direct custom-provider calls retain the
  thread fallback, and Ollama server-side cancellation remains separate.
  See [process isolation and GPU verification](docs/PROCESS_TIMEOUT.md).
- Ollama requests use a timeout and fixed generation options. Students use greedy
  generation with an output-token cap. New arithmetic comparisons record adapter hashes before and after evaluation,
  source hashes, package versions and Git state. Remote model revisions remain
  unpinned; a complete portable GPU dependency lock remains future work.

See [the implementation handoff](docs/FIX_HANDOFF.md) for the original changes
and planned checks. GPU training and evaluation have since completed as
documented above.

## Capturing the GPU environment

Run `.venv-gpu/bin/python scripts/capture_gpu_environment.py` on the machine
used for inference. It records the currently installed package versions in
`requirements-gpu.txt` and hardware details in `docs/results/2026-09-22/gpu_environment.json`.
The snapshot excludes the editable project path. It is environment evidence,
not a validated portable lockfile; use the documented CUDA indexes when testing
a clean installation. Capture time is recorded separately from experiment time.

See [the consolidated reliability review](docs/RELIABILITY_REVIEW.md) for the
verified scope and remaining research and reproducibility work.

## License

Apache 2.0.
