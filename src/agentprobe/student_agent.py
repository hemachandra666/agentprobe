"""Run the distilled student as an agent: it writes tool calls as text,
we parse them, run the REAL tools, and log the trajectory.

The student only chooses tools and arguments. Your real tools do the math.
"""
from __future__ import annotations
import re
from pathlib import Path
from unsloth import FastLanguageModel
from . import tools
from .trajectory import Trajectory

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = str(ROOT / "student_lora")

SYSTEM = ("You are a calculator agent. Use the add and multiply tools to compute step by step, "
          "then give the final number.")

# matches lines like:  multiply(a=6, b=7)   or   add(b=3, a=15)
CALL_RE = re.compile(r"(add|multiply)\(\s*([ab])\s*=\s*(-?\d+\.?\d*)\s*,\s*([ab])\s*=\s*(-?\d+\.?\d*)\s*\)")

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=ADAPTER_DIR, max_seq_length=2048, load_in_4bit=True,
        )
        FastLanguageModel.for_inference(_model)
    return _model, _tokenizer


def _generate(question: str) -> str:
    model, tok = _load()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    inputs = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(input_ids=inputs, max_new_tokens=200,
                         temperature=0.1, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def run(task_id: str, question: str, model: str = "student") -> Trajectory:
    """Ask the student, parse its tool calls, run REAL tools, log the trajectory."""
    traj = Trajectory(task_id=task_id)
    text = _generate(question)

    for m in CALL_RE.finditer(text):
        name = m.group(1)
        # arguments may appear as a= then b=, or b= then a=; normalize by name
        k1, v1, k2, v2 = m.group(2), m.group(3), m.group(4), m.group(5)
        args = {k1: float(v1), k2: float(v2)}
        fn = tools.REGISTRY.get(name)
        try:
            result = fn(**args) if fn else f"error: unknown tool {name}"
        except Exception as e:
            result = f"error: {e}"
        traj.add_step(name, args, result)

    # final answer = last real tool result, if any
    if traj.steps:
        traj.final_answer = f"Result: {traj.steps[-1].result}"
    else:
        traj.final_answer = text.strip()[:80]
    return traj


if __name__ == "__main__":
    from .task_suite import TASKS
    for task in TASKS[:3]:
        t = run(task.task_id, task.question)
        print(f"{task.task_id}: tools={t.tool_sequence()} steps={t.step_count} answer={t.final_answer}")