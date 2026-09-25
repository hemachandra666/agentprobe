# Review cleanup

The refund scorer is now `refund-scorer-3`: exhausted-tool success requires exactly
two attempts, agreeing with the checker and one-retry policy. Historical evidence
retains its original scorer version. Its published strict results are unchanged;
regression tests verify the recorded fresh cases as well as development runs.

New arithmetic comparisons capture source hashes, Git state, installed versions,
and all five adapter inference-file hashes before evaluation. Missing files fail
the run with retained metadata. A second capture rejects adapter changes observed
at the end. This is not tamper-proof or a snapshot of files opened by the loader.
The declared base revision is recorded separately from an unresolved runtime
revision. No immutable Hugging Face revision or Ollama digest binding is claimed.
Remote revision pinning remains a distinct reproducibility task; cached revision
candidates must never be presented as proof of the actually loaded revision.

Arithmetic success continues to mean a correct final answer matching a real tool
result, not proof of appropriate intermediate operands. Calling add(answer, 0)
can meet that policy. Changing this would define a new metric and must not silently
rescore earlier experiments. The existing path metric is diagnostic only.

No training, model benchmark or GPU rerun is required for this cleanup. Existing
results and limitations remain intact. The project is an experimental evaluation
reference, not a general-purpose production evaluation service.
