<p align="center">
  <img src="docs/images/banner.png" alt="AgentProbe" width="100%">
</p>

# AgentProbe

**This measures whether a distilled agent kept its behavior. It does not make a model behave better.**

When you distill a large model into a small one, the standard check is whether the small model still gets the right answers. That check is blind to something: an agent can reach the right answer while taking a broken path, calling the wrong tools, looping, failing calls, wandering through extra steps. Accuracy scores that as a perfect success.

AgentProbe scores the whole trajectory, the sequence of tool calls, so it can ask a harder question: after distillation, did the small model keep the teacher's *behavior*, or just its final answers?

<p align="center">
  <a href="https://agentprobe-ltkk9j7sybevnmkk8yhz2n.streamlit.app/"><img src="https://img.shields.io/badge/Live_Demo-Open_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"></a>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/License-Apache_2.0-D22128?style=for-the-badge" alt="License: Apache 2.0">
</p>

<p align="center">
  <img src="docs/images/dashboard.png" alt="AgentProbe dashboard" width="90%">
</p>

---

## The finding

A 1.5B student was distilled from a 7B teacher (Qwen2.5), then both were run through the same trajectory evaluation, 20 runs per task.

| | Trajectory match | Step efficiency | Loop rate | Error rate |
|---|---|---|---|---|
| Teacher (Qwen2.5-7B) | 0.975 | 0.986 | 0.007 | 0.012 |
| Student (1.5B distilled) | 0.993 | 1.000 | 0.000 | 0.000 |

The 1.5B student matched the 7B teacher's trajectory behavior.

**Read this as preservation, not superiority.** A 1.5B model does not beat a 7B one; the scores are within noise. The student scores well because the agent runs the real tools, so the model only has to choose the right tool and arguments, the tools do the arithmetic. AgentProbe measures tool-selection behavior, and that is what distillation transferred. The student's own arithmetic, run without tools, is poor (see Limitations).

---

## Why accuracy cannot see this

Consider two agents solving "add 3 and 4, then double it." Both output 14. One did `add(3,4)` then `multiply(7,2)`. The other did `add(3,4)`, `add(3,4)` again, `multiply(7,2)`, a wasted repeated call.

Accuracy scores these identically: both got 14. The wasted step, the beginning of a loop, is invisible to the metric, which means it is invisible to any decision the metric drives. AgentProbe scores the path, so the two are no longer the same.

---

## A real trajectory

The agent solved a 3-step task in 6 steps with 2 failed tool calls, then still reached the correct answer. Accuracy scores it 100%.

    step 0  add(a=7, b=8)        -> 15.0
    step 1  multiply(a=3, b=...) -> failed: bad argument
    step 2  add(a=2, b=...)      -> failed: bad argument
    step 3  add(a=7, b=8)        -> 15.0     (restarted from scratch)
    step 4  multiply(a=3, b=15)  -> 45.0
    step 5  add(a=2, b=45)       -> 47.0

Right answer, broken path. That gap is the whole point.

---

## Benchmarking open-weight models

The same evaluation ranks model families on trajectory quality, not answer accuracy.

| Model | Trajectory match | Step efficiency | Loop rate | Error rate |
|---|---|---|---|---|
| qwen2.5:7b | 1.000 | 1.000 | 0.000 | 0.015 |
| mistral:7b | 0.671 | 0.813 | 0.067 | 0.174 |
| llama3.1:8b | 0.589 | 0.837 | 0.000 | 0.174 |
| nemotron-mini:4b | 0.549 | 0.778 | 0.044 | 0.778 |

The spread is real: qwen took the cleanest paths, the 4B model failed most of its tool calls on multi-step tasks. Accuracy alone would have flattened this.

---

## Limitations, before the methods

Read these before citing anything above.

**The task domain is narrow.** One calculator domain, two tools (add, multiply), multi-step chains. This is a proof of the method, not a broad benchmark. Whether the result holds on richer tool sets is untested.

**The student's competence is borrowed from the tools.** The 1.5B student, asked to compute without tools, gets multi-step arithmetic wrong. It scores well only because the agent executes the real tools. So the claim is precise: distillation transferred *tool-selection behavior*, not reasoning or calculation.

**Results are non-deterministic.** Model outputs vary run to run, so single runs mislead. The reported figures are aggregated over 20 runs per task. Numbers still shift a little between full runs.

**The distilled model is served in-process, not exported.** The GGUF/Ollama export path failed (an Unsloth bug), so the student runs directly in Python and its tool calls are parsed from its text output. This works, but it is not a packaged model artifact.

---

## Method

An agent runner drives a model against a fixed task suite, logging every tool call as an ordered trajectory. Scorers grade each trajectory against a known-good reference path on five axes: trajectory match, step efficiency, loop detection (repeated identical calls), error rate, and recovery. Each task is run many times and the scores are aggregated, since one run measures luck as much as skill. Traces are written as OpenTelemetry-style spans in JSONL.

For distillation: the teacher's clean trajectories become training data, a small student is QLoRA fine-tuned on them (with a held-out split to check for overfitting), and the student is then run through the identical evaluation. The teacher and student are scored the same way, so the comparison is apples to apples.

---

## Repository contents

    src/agentprobe/
      agent.py            the agent runner (drives a model, logs trajectories)
      scorer.py           the five trajectory scorers
      benchmark.py        multi-model benchmark
      generate_teacher.py teacher trajectory generation
      train_student.py    QLoRA distillation with a held-out split
      student_agent.py    runs the distilled student in-process
      compare.py          teacher-vs-student comparison
      dashboard.py        the Streamlit dashboard
    tests/                unit tests for the scorers (run in CI)
    comparison.json       the teacher-vs-student result
    results.json          the model benchmark

---

## Running it

Requires Python 3.11+ and [Ollama](https://ollama.com). Distillation requires an NVIDIA GPU.

    git clone https://github.com/hemachandra666/agentprobe.git
    cd agentprobe
    uv sync
    ollama pull qwen2.5:7b

    uv run python -m agentprobe.run_eval --runs 5        # evaluate the suite
    uv run python -m agentprobe.benchmark --runs 3       # benchmark models
    uv run streamlit run src/agentprobe/dashboard.py     # view the dashboard

---

## What would make this stronger

Richer tool domains beyond calculators. Pointing AgentProbe at a real external agent framework instead of the built-in runner. A packaged model export so the student is a portable artifact, not an in-process object. Wider coverage of the multi-step tasks where small models fail most.

## License

Apache 2.0.