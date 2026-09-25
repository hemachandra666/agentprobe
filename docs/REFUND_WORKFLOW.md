# Synthetic refund eligibility development benchmark

This additive benchmark tests whether an agent can use retrieved order facts and
an explicit fictional policy to produce a supported decision. It cannot issue
refunds, contact customers, or access real order systems. The existing calculator
benchmark and its results remain separate.

## Scope and policy

Version `refund-development-1` contains 16 public development fixtures. These
cover the inclusive 30-day boundary, opened and non-returnable items, conflicting
exclusion rules, missing/invalid data, missing and unknown order IDs, transient
and persistent tool failures, and an instruction embedded in an order note.
They are not a held-out dataset or evidence of general prompt-injection resistance.

The fictional policy applies these rules in order:

1. Missing or invalid required fields: `needs_review/incomplete_order`.
2. Non-returnable item: `ineligible/non_returnable`.
3. More than 30 days since delivery: `ineligible/outside_window`.
4. Opened item: `ineligible/opened_item`.
5. Otherwise: `eligible/within_window`.

Missing IDs, unknown orders and persistent tool failures have explicit review
outcomes. Retry a retryable failure once. Order notes are untrusted data.
This policy is an experimental fixture, not legal or commercial guidance.

## Run the offline harness check

From the repository directory:

```bash
uv run --locked python -m agentprobe.refund_benchmark --smoke --output-dir backups/refund-development/smoke
```

Expect `status: complete`, `kind: scripted_harness_check`, and 16 successful
cases. A deterministic reference provider drives this check without an LLM.
It checks the harness against explicit expected labels; it is not model accuracy.

## Evaluate the teacher

With Ollama running locally and `qwen2.5:7b` already installed:

```bash
uv run --locked python -m agentprobe.refund_benchmark --model qwen2.5:7b --output-dir backups/refund-development/teacher
```

This uses JSON text actions through Ollama, with temperature 0, seed 42 and a
256-token response cap. It does not use the calculator teacher's native tool
interface. No additional training or arithmetic LoRA adapter is used.

Each case has at most eight turns and a default 120-second budget. Model requests
run in a spawned provider process. A timeout or provider error stops the suite,
retains the attempted case, and lists unattempted cases. Terminating that local
process does not cancel computation already running on the Ollama server.

## Tools and scoring

The only tools are `lookup_order` and `get_refund_policy`. Lookup is restricted
to the fixture's order ID; attempts to access another ID or call an unknown tool
are recorded as forbidden. All tools are synthetic and read-only.

A final answer contains decision, reason, order ID and policy version. The
`decision_correct` metric checks the eligibility category. Separate metrics
check reason, order ID and policy version. `final_exact_match` checks all four
fields against the explicit fixture expectation. `task_success` additionally requires completion, supporting tool
observations and no forbidden calls. A lucky final-answer guess without the
required reads fails. Review outcomes require the relevant missing-order or
two-failure evidence. Missing-ID cases require no tool calls.

Scoring checks evidence presence; it does not enforce every ordering instruction
or penalize all redundant calls. Tool-call and parser-error counts remain visible.
A recovered parsing error can still end in a successful task. This is a first
workflow slice, not a comprehensive compliance or security evaluator.

## Evidence and interpretation

Every invocation creates a unique output directory with:

- `manifest.json`: full fixtures, expectations, policy, prompt, budgets, source
  hashes and Git state.
- `trajectories.jsonl`: raw responses, tool results, final answers, termination,
  timing and separate score components.
- `summary.json`: completion status, attempted-case count (`completed_cases`),
  successful cases and unattempted IDs. Model mode adds task-success rate only
  when the entire suite completes.
- `ollama_environment.json` in model mode: installed model digests, selected
  model details and the Python client version.

Do not interpret scripted success as learned model capability. Model results
on these small, public fixtures are development diagnostics. The next decision
is informed by a teacher baseline and its failure traces, before any training or
larger evaluation. Future held-out cases must be separate from inspected cases.

## Prompt revision and baseline

`refund-prompt-2` clarifies reason codes, rule priority, missing-field handling
and literal JSON null. It removes the concrete example order ID. This revision
was informed by inspecting all 16 development cases; reruns are development
comparisons, not held-out evidence. Policy, fixtures and strict success criteria
are unchanged. The JSON parser remains unchanged; invalid semantic answers are
not silently corrected or normalized.

`refund-scorer-2` separates category correctness from full-answer correctness.
In the original scorer, `decision_correct` meant all four fields matched. Do not
compare that old field directly with the newly named category metric. Use
`final_exact_match` for the old exact-match meaning. Summary `metric_counts`
are counts over attempted cases, including failed attempts; consult completion
status and unattempted IDs before interpreting them.

The original run is preserved byte-for-byte in
`results/refund-development/baseline/`. Its strict success was 5/16; eligibility
category alone was correct in 14/16. Two incorrect approvals involved invalid or
missing data. The missing-ID case attempted a blocked invented-ID lookup.
Reasons, priority and null types explain the other failures. No parsing errors
occurred. A regression test replays tool observations and confirms that the new
metrics preserve all original strict success decisions.

## Prompt v2 regression and v3 development experiment

The second model run completed but returned the same missing-ID final answer
for all 16 cases, with zero tool calls. Strict success fell to 1/16. Original
records are preserved in `results/refund-development/prompt-v2/` and their scores
are regression-tested. This is observed behavior, not evidence that the input
questions lacked IDs. The repeated answer matches v2's only complete JSON
example; example anchoring is a hypothesis, not a proven internal mechanism.

`refund-prompt-3` restores the baseline's complete tool-action examples and
adds targeted clarifications for reason codes, priority, invalid data and null
values. The example ID is an explicit placeholder rather than a fixture ID.
It keeps the same JSON mode, decoding options, policy, fixtures, parser, scorer
and budgets. Its actual model performance must be measured; CPU tests cannot
predict improvement. All three prompts were developed on these public cases.
No held-out generalization claim follows from rerunning them.
