# Consolidated reliability review

Base commit: `31872f1`. This patch fixes the confirmed CPU-verifiable reliability
issues in one change. It does not modify the scorer policy, trained adapter,
training dataset, or published benchmark files.

## Fixed and covered by regression tests

- Comparison aborts on provider construction failures, provider execution errors,
  timeouts, and invalid action shapes. The failed attempt is saved, and the
  comparison is marked failed instead of reporting an infrastructure failure as
  a completed quality benchmark. Summaries explicitly count provider errors and
  timeouts when replaying saved evidence.
- Teacher generation aborts after inference failure. It records every completed
  attempt and accepted trajectory in a unique evidence directory. Existing output
  survives failures or runs with zero accepted trajectories. A successful run
  atomically replaces the selected output.
- The installed generator defaults to the working directory, supports `--output`,
  and works with packaged datasets outside the checkout. CI now exercises actual
  generation and preparation with scripted reference actions, not only CLI help.
- Preparation validates task identity, canonical answers, finite observations,
  argument schemas, successful tool arithmetic, and final-answer consistency
  before writing training outputs. Existing valid data prepares identically.
- Replay validates task ID, question, reference answer, optional family, and tool
  arithmetic. It rejects repeated run IDs, mixed experiments, empty input, and
  accidental overwriting of input evidence. With a matching comparison summary it
  checks model/task/run coverage and the selected dataset hash. Family summaries
  and parsing-error counts are preserved.

## Verification scope

The suite contains 83 CPU tests, including complete replay of the 2,277 published
records and byte-for-byte reproduction of both conversation datasets and the
training split manifest. Original metrics are compared with a small floating-point
summation tolerance, accommodating Python-version differences. Wheel build and
installed generator/preparation checks run without GPU dependencies or Ollama.
These scripted checks are not fresh LLM inference.

## Still pending and deliberately not claimed complete

1. Publish the existing trained adapter as a versioned external artifact and record
   adapter/base-model checksums and the Ollama digest. A dependency snapshot is not
   a fully verified clean-install GPU lockfile.
2. Create a new dataset version grouping mathematical equivalents before splitting.
   The existing 26 equivalent train-pool/test questions remain disclosed in README.
   Preserve the historical experiment; do not silently change its denominator.
3. Add automatic best-validation-checkpoint selection and evaluate any training-loss
   masking changes with a fresh development/evaluation protocol. This patch does
   not change GPU training settings or require retraining.
4. Measure latency and memory with warm-up and model process isolation; implement
   the NVIDIA toolkit integration as a separate, pinned experiment.
5. Python cannot forcibly cancel the native/GPU worker after timeout. The CLI now
   stops rather than continuing to the next task; hard cancellation still requires
   process isolation. Embedding applications must not catch the failure and start
   overlapping inference in the same process.

Exact numeric correctness does not prove every intermediate argument came from the
intended dependency chain. Broader domains, multiple training seeds, and semantic
trajectory correctness remain research work, not guarantees of this evaluator.
No CPU test suite can certify arbitrary future GPU/library combinations.
