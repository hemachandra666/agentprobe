# Version-2 experiment protocol

This is a new benchmark, not a corrected denominator for v1. V1 data, adapter,
release and 2,277 historical traces remain unchanged. No v2 GPU result is claimed.

## Split policy

The new generator uses seed 43 and operands 1–100 (v1 used 1–20). It constructs
expression trees, sorts commutative operands and flattens associative add/multiply
nodes. Each retained group has one representative across all families and splits.
It excludes every group present in the packaged v1 train/test pools. This addresses
first-operand swaps and same-operation reassociation. It does not prove absence of
all semantic equivalence: distributivity, identity operations and equal numeric
answers are not collapsed. Scores on this changed distribution are not directly
comparable with v1 as evidence of model improvement.

Two original families remain entirely test-only. Other families split 80/20.
Preparation groups accepted teacher trajectories into train/validation with seed
43, verifies dataset hashes and trajectory arithmetic, and records group IDs and
hashes. Each command requires a new output directory, preventing silent overwrites.

## Separate paths and commands

Run from the repository. Generate once, freeze the manifest, and decide the
training protocol before looking at test results. These commands are a staged
workflow: inspect each result before proceeding. Do not run the final comparison
until training selection has been validated.

```bash
uv run --locked python -m agentprobe.dataset_v2 --output-dir backups/experiment-v2/data
.venv-gpu/bin/python -m agentprobe.generate_teacher --data-dir backups/experiment-v2/data --output backups/experiment-v2/teacher_data.jsonl --runs 1 --max-tasks 1440
uv run --locked python -m agentprobe.prepare_v2 --data-dir backups/experiment-v2/data --teacher-file backups/experiment-v2/teacher_data.jsonl --output-dir backups/experiment-v2/training
.venv-gpu/bin/python -m agentprobe.train_student --data-dir backups/experiment-v2/training --output-dir backups/experiment-v2/smoke_adapter --select-best --max-steps 2
```

The two-step smoke is only an API/checkpoint compatibility test. Inspect its
selection.json and trainer_state.json for recorded validation loss, selected
checkpoint and saved model before starting a fresh full run:

```bash
.venv-gpu/bin/python -m agentprobe.train_student --data-dir backups/experiment-v2/training --output-dir backups/experiment-v2/selected_adapter --select-best --epochs 3
```

Evaluation and saving both occur each epoch. The trainer reloads the checkpoint
with minimum validation loss, then exports through trainer.save_model and saves
trainer state plus selection.json. This is loss-based selection, not necessarily
maximum agent task success. Loss masking is unchanged; studying response-only
masking requires a separate controlled experiment. Real GPU/TRL restoration has
not yet been verified by the CPU orchestration tests.

After checking selection evidence, evaluate once with explicit v2 paths:

```bash
.venv-gpu/bin/python -m agentprobe.compare --data-dir backups/experiment-v2/data --adapter-dir backups/experiment-v2/selected_adapter --models teacher untuned tuned --max-tasks 10000 --runs 1
```

The cap above includes all 760 generated test tasks under default settings.
Comparison records the adapter path and uses the spawn-safe adapter factory.
Retain datasets, manifests, dependency snapshot, selection record, adapter hashes
and the complete new comparison directory. Publish selected evidence under a new
versioned docs/results path after review, since backups/ is ignored by Git.
Multiple seeds, uncertainty estimates and wider domains remain further studies.

## Recorded v2 results

Experiment: d4f7855bb5714706aec5b96825b5787c.
Scorer: 2.0. Evaluation: 760 tasks per model, one run per task.

| Model | Successful tasks | Task success |
|---|---:|---:|
| Teacher Qwen2.5 7B | 748/760 | 98.42% |
| Untuned student 1.5B | 123/760 | 16.18% |
| Tuned student 1.5B | 721/760 | 94.87% |

Training used 1,008 accepted teacher trajectories, producing 2,819 training
and 498 validation next-action examples. Three epochs completed, totaling
1,059 optimizer steps. Minimum validation loss selected checkpoint 353
from epoch 1, with validation loss 0.13024970889091492.

The tuned student succeeded on 377/400 tasks across the two held-out
families. Its 39 failures comprised 18 incorrect last tool results and
21 correct last tool results followed by incorrect final answers.
No tuned-student parser errors, tool execution errors, loops, provider
failures, or timeouts were recorded.

An offline replay of all 2,280 trajectories validated task identities,
dataset coverage, successful tool arithmetic, individual scores, and
aggregate/per-family metrics within floating-point tolerance (1e-12).
Replay verifies saved evidence; it is not a new inference run.

Evidence: [experiment files](results/experiment-v2/).

The adapter checksum supplied separately for this experiment is:
8ad64ebe7be9ac555c5bcca78698bfa7c51f9a0c84cb048d655883866537b8ba

This is one training run on synthetic arithmetic tasks. V2 uses a different
dataset from v1, so their headline scores are not a controlled comparison.
Teacher and student interfaces differ. GPU speed and memory savings have
not been established. Preserve this test result; further changes informed
by its failures require a fresh held-out evaluation.


