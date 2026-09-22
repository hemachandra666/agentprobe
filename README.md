<p align="center">
  <img src="docs/images/banner.png" alt="AgentProbe" width="100%">
</p>

# AgentProbe

**Honest evaluation infrastructure for AI agents. It measures whether an agent actually behaves correctly, not just whether it looks like it did.**

<p align="center">
  <a href="https://agentprobe-ltkk9j7sybevnmkk8yhz2n.streamlit.app/"><img src="https://img.shields.io/badge/Live_Demo-Open_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"></a>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/License-Apache_2.0-D22128?style=for-the-badge" alt="License: Apache 2.0">
</p>

---

## What this project actually found

I built a tool to check whether a distilled 1.5B agent kept the behavior of its 7B teacher.

The first version said the distilled student preserved **102%** of the teacher's behavior. That number was wrong, and the interesting part of this project is *why* it was wrong and how the honest measurement was built.

A code review found the flattering result was an artifact of four measurement flaws:

1. The "held-out" test set was made of the same questions the model trained on, so memorization looked like generalization.
2. The scorer checked only tool *names* in order, not arguments or the final answer. A trace computing 600 for a task whose answer is 14 scored a perfect 1.0.
3. The teacher and student ran through different execution loops, so they were never compared fairly.
4. There was no untuned baseline, so there was no way to know what fine-tuning actually added.

After fixing all four, the honest result is very different.

## The honest result

Measured on genuinely unseen tasks (unseen numbers, plus two entire task families the model never trained on), through one shared execution loop, scored on actual answer correctness:

| Model | Task success (correct answer) |
|---|---|
| Teacher (Qwen2.5-7B) | 18% |
| Tuned student (1.5B, distilled) | 4% |
| Untuned student (1.5B, base) | 0% |

Fine-tuning moved the small model from 0% to 4%. A real but small gain, nowhere near the teacher's 18%. The original "102% behavior preserved" was pure artifact.

The lesson: **agent distillation is much harder than accuracy-style metrics suggest, and naive evaluations dramatically overstate it.** The value of this project is the measurement that revealed that.

## Why naive metrics hide this

Consider two agents solving "add 3 and 4, then multiply by 2." Both call `add` then `multiply`. One computes `add(3,4)=7, multiply(7,2)=14`. The other computes `add(100,200)=300, multiply(300,2)=600`.

A tool-name check scores both a perfect 1.0. The answer is 14. AgentProbe now checks the actual computed answer, so the second trace correctly fails. This single fix is most of why the honest numbers are so much lower.

## What AgentProbe measures

For every run, on the actual tool outputs (not the model's text):

- **task success**: did it reach the correct final answer.
- **answer correctness**: computed result vs the task's true answer.
- **tool-sequence match**: shape only (right tools, right order), named so it is never mistaken for correctness.
- **step efficiency**: steps vs minimum, reported only for successful runs.
- **loop and cycle detection**: consecutive repeats and non-adjacent A-B-A-B oscillation, retries after an error excluded.
- **recovery**: an error occurred but the task still succeeded.

## How the evaluation is kept honest

- **Real train/test separation** (`task_gen.py`): 2200 unique task instances, split by whole-problem identity before any data is generated, with two entire task families held out for test only. An automated check (`check_leakage.py`) enforces zero overlap and zero duplicates.
- **One shared execution loop** (`engine.py`): teacher, tuned student, and untuned student all run the same action-observation loop, one action, see the real result, act again. Invalid actions are recorded as failures, never dropped.
- **A control baseline**: the untuned 1.5B base model, isolating what fine-tuning added.
- **Correctness-gated teacher data**: only trajectories that reached the correct answer are kept.

## Limitations

- The task domain is a controlled calculator (two tools, multi-step chains). A proof of the *method*, not a broad benchmark.
- The reported figures are from a modest run (tens of unseen tasks, one run each), so exact percentages are noisy. The qualitative finding (distillation helps a little, far below the teacher) is stable.
- The distilled model runs in-process; a GGUF/Ollama export was attempted and failed on an Unsloth bug, so it is not a portable artifact.

## Running it

Requires Python 3.11+ and [Ollama](https://ollama.com). Distillation and student evaluation require an NVIDIA GPU.

    git clone https://github.com/hemachandra666/agentprobe.git
    cd agentprobe
    uv sync
    ollama pull qwen2.5:7b

    uv run python -m agentprobe.task_gen
    uv run python -m agentprobe.check_leakage
    uv run python -m agentprobe.compare --runs 1 --max-tasks 50

## Repository contents

    src/agentprobe/
      task_gen.py          generate unique tasks, split with family holdouts
      check_leakage.py     automated train/test leakage check
      engine.py            the shared action-observation loop (all models)
      agent.py             teacher provider (Ollama, structured tools)
      student_agent.py     tuned and untuned student providers
      scorer.py            honest scorers (task success, answer correctness, loops)
      generate_teacher.py  correctness-gated teacher data
      train_student.py     QLoRA distillation
      compare.py           teacher vs untuned vs tuned, on unseen tasks
      dashboard.py         the Streamlit dashboard
    tests/                 scorer unit tests, including the 600-vs-14 case

## License

Apache 2.0.
