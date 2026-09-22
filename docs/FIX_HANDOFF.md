# Evaluation correctness handoff

Changes are based on upstream commit 54ce4f3. They fix the CPU-testable review
failures; revised model scores require fresh inference and training.

## Implemented

1. Separate final-answer, computed-result and completion checks; require all for
   task success. Historical scores are labeled, not overwritten with invented results.
2. Parse one complete student action; preserve repeated and unknown tool calls.
   Reject multiple structured teacher actions explicitly.
3. Use structured step status and safe signatures for malformed/nested arguments.
4. Bound provider waiting and call counts; retain evidence on scoring failures.
5. Route run_eval and benchmark through the current evaluator.
6. Bundle example data in the wheel, keep outputs in working directories, and
   regenerate the dependency lock.
7. Include cycles in loop summaries; distinguish tool failures from parse failures.
8. Format next-action training examples with real observations and disjoint
   validation questions; hash prepared files to prevent stale-data training.
9. Add CPU regression tests, installed-package checks and an offline trace audit.

## Validation

The 46-test CPU suite passes locally. The lockfile check passes, and the built
wheel loads all 759 test tasks outside the checkout. The CPU suite covers scoring false positives, missing finals, parser behavior,
budget boundaries, malformed actions, recorded failures, dataset isolation and
training-format parity. The wheel must load example tasks outside the checkout.
No GPU training, live Ollama run, or hosted dashboard execution is claimed here.

## Your GPU work

Use the README's small smoke pipeline first. Save the command, complete error
trace, GPU model, driver/CUDA/Torch versions, and generated run directory if a
step fails. Share those results for the next debugging pass. Do not publish new
benchmark percentages until the smoke test and full comparison succeed.

Run the teacher, untuned and newly trained student under the corrected protocol.
A lower score is useful evidence if it exposes real failures. Examine per-family
results and final answers, not just the headline aggregate.

## Remaining work

- Native inference may continue after a timeout; hard cancellation needs isolated
  worker processes. The comparison stops rather than launching more inference.
- GPU dependency/API compatibility and model loading must be smoke-tested on the
  target GPU. Old LoRA artifacts require retraining for the new action format.
- Add immutable model/adapter identifiers, hardware manifests and uncertainty
  estimates before making broader experimental claims.
- A shared engine does not eliminate provider-interface differences. Test explicit
  protocol compliance as well as end-task success for each backend.
- Broader workflows, NeMo integration and performance measurements remain P2 work.
