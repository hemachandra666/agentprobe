# Refund workflow: development results and frozen evaluation

| Prompt | Strict success | Category correct |
|---|---:|---:|
| Baseline | 5/16 | 14/16 |
| v2 | 1/16 | 6/16 |
| v3 | 9/16 | 11/16 |

All 48 recorded tool sequences and strict scores were replay-checked. Evidence
is in `results/refund-development/`. The first scorer named full-answer matching
`decision_correct`; later versions separate category and exact-answer matching.
V2 repeated the missing-ID response on every case. V3 recovered tool use but
still failed boundary, priority, incomplete-data and post-retry decisions.
It gained seven strict passes and lost three compared with baseline.

## Frozen experiment

Prompt v3, policy, parser and strict model scoring are frozen before the next
model run. `--suite fresh` supplies 28 new questions/IDs in known families,
including two wording variants. Labels are explicit. These are fresh instances,
not hidden families, mathematically independent samples, or a new domain.
The suite was constructed after development failures were inspected. Do not tune
on its results and continue describing it as unseen. Report the full result,
including failures, and keep each run's manifest and model metadata.

## Separate gate

`refund-gate-1` is a post-run deterministic allow/block checker. It reads the
question, the final response and trusted recorded tool observations, not fixture
labels or scores. It recognizes only the fictional policy version. It neither
repairs the model response nor issues a refund. Missing evidence, unsupported
policy, forbidden actions and disagreements block the response for review.
Model scores remain unchanged; no blocked case is converted into a model pass.

Summary gate counts report allowed, blocked, incorrect allowed, and correct
blocked, where correctness is the benchmark's strict task success. Always report
coverage (allowed / attempted) alongside incorrect allowed: blocking everything
is not a useful result. All 48 development traces were checked offline; the gate
allowed the 15 strict successes and blocked the 33 failures. This is development
verification, not a guarantee for real requests. Its policy evaluator shares the
scripted smoke provider's implementation; smoke success is not independent proof
of gate correctness. Explicit labels and recorded model failures provide separate
checks. It assumes trustworthy tool observations and does not defend against
fabricated trace files or all possible malformed external inputs.

## Commands

```bash
uv run --locked python -m agentprobe.refund_benchmark --smoke --suite fresh --output-dir backups/refund-fresh/smoke
uv run --locked python -m agentprobe.refund_benchmark --model qwen2.5:7b --suite fresh --output-dir backups/refund-fresh/teacher
```

The smoke check should complete 28 cases. The recorded model result is documented below. A failed infrastructure run retains partial evidence; do not
interpret it as a completed 28-case benchmark. Local worker termination does not
cancel remote Ollama computation. No new training or GPU dependencies are needed.

Commit the refund code, tests, documentation and preserved evidence together. This
milestone makes the project a documented experimental benchmark; it does not
establish production readiness, broad generalization or customer safety.

## Recorded fresh evaluation

Run `61dc18e7aa25495aacc10372652e36c5` used `qwen2.5:7b`, prompt
`refund-prompt-3`, scorer `refund-scorer-2` and gate `refund-gate-1`.
The prompt and source hashes match the frozen implementation. All 28 fixture
records match the suite; all tool observations, model scores and gate decisions
were reproduced offline. This audit is not another model inference run.

| Measure | Recorded result |
|---|---:|
| Strict model task success | 10/28 (35.71%) |
| Eligibility category correct | 13/28 (46.43%) |
| Model final answers | 27/28 |
| Turn-budget exhaustion | 1/28 |
| Gate allowed / coverage | 10/28 (35.71%) |
| Gate blocked | 18/28 |
| Incorrect allowed, relative to strict task success | 0/10 allowed |
| Correct blocked, relative to strict task success | 0/10 correct |

The 18 blocks comprised 14 decision/field disagreements, two missing or
unsupported policy observations, one forbidden action and one incomplete run.
The gate's reason is its first blocking condition, not an exhaustive diagnosis.
No model response was repaired or rescored as successful by the gate. Blocking
leaves work for review; it is not successful automated resolution.

Evidence: [summary](results/refund-fresh/teacher/summary.json),
[raw trajectories](results/refund-fresh/teacher/trajectories.jsonl),
[manifest](results/refund-fresh/teacher/manifest.json), and
[model metadata](results/refund-fresh/teacher/ollama_environment.json).

This is one run over 28 synthetic known-family cases with paired wording/ID
variants. The cases are correlated and were designed after inspecting development
failures. Zero observed incorrect allowances does not establish a zero error
rate, production safety or robustness to arbitrary requests. Do not pool this
result with development scores or treat 28 cases as independent customer samples.
The milestone is complete as an experimental evaluation once this evidence and
implementation are merged; no further prompt tuning is required for this stage.
