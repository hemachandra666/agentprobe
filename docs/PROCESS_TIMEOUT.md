# Student process isolation

`python -m agentprobe.compare` now uses one spawned process per student model.
The model stays loaded across tasks; tools and trace writing remain in the parent.
The worker is closed before the next model starts and on exceptions or interruption.
Timeout or worker error closes the IPC channel, terminates and joins the worker,
and escalates to kill if needed. A closed worker cannot be reused. Comparison
still records the failed trajectory and aborts instead of silently retrying it.

The existing 60-second task budget includes first model loading. Cleanup may add
up to four seconds beyond the action deadline. An OS process that cannot be reaped
raises an error; do not continue inference in that case. This owns the worker
process, not an arbitrary subprocess tree or remote inference server.

## Direct callers

Use `ProcessProvider(StudentProvider)` inside a `with` block and pass it to
`engine.run`. Put executable code in an `if __name__ == '__main__':` guard in a
Python file or use the supplied module commands. Spawn factories must be importable
and picklable; do not pass loaded CUDA models, notebook-local classes or lambdas.
Adapter path overrides must be configured inside the worker factory.

Existing direct `student_agent.run`, `_load_tuned`, and custom providers retain
their previous behavior; they are not automatically isolated. Plain providers
still use daemon-thread waiting. Ollama runs on its own server and is not killed
by this local student-process change. Teacher generation is unchanged.

## Verification

CPU suite: 91 tests pass, including timeout cleanup, process crash, constructor
error, caller failure, persistent reuse and successful inference after replacement.
The existing historical replay and training-file identity checks still pass.

On the user's GPU, from the repository with the existing editable GPU environment:

```bash
.venv-gpu/bin/python -m agentprobe.check_process_timeout
```

This loads the current `student_lora`, completes a calculator task, deliberately
stalls its model-loaded worker, forces timeout, confirms exit, then loads a fresh
worker and completes the task again. Evidence is written under a unique `runs/`
directory even on failure. It does not retrain or change adapters. It is a
controlled Python stall with a resident GPU model, not proof of recovery from
all CUDA/driver hangs. GPU verification is pending until this command passes;
process exit plus successful replacement is the acceptance check, not a claim
that total machine VRAM becomes zero (other processes can use GPU memory).

## Graceful shutdown follow-up

Idle workers receive a shutdown message and have five seconds to exit normally.
Termination and kill remain bounded fallbacks. In-flight failures/timeouts skip
the graceful wait. The GPU smoke now also requires replacement exit code zero.
Forced termination can still trigger dependency-owned semaphore tracker warnings;
these are not suppressed. Inspect worker exit and GPU state separately.

## Recorded GPU verification

The controlled-stall test passed on an NVIDIA GeForce RTX 5080 Laptop GPU.
The timed-out worker exited with code -15; its replacement completed the
calculator task and exited normally with code 0.

Evidence: [GPU timeout smoke result](results/process-timeout/result.json).

A subsequent nvidia-smi check showed 0 MiB used and no running GPU processes.
A subsequent /dev/shm check found no sem.mp-* files.

Python emitted one semaphore resource-tracker warning after forced termination.
The warning remains documented and is not suppressed. These observations verify
this controlled test, not recovery from arbitrary CUDA or driver hangs.
