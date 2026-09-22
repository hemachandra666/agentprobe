"""Student providers for the shared loop (P0-3, P0-5).

Two providers:
  - StudentProvider:     the distilled (fine-tuned) student, uses the LoRA adapter.
  - BaseStudentProvider: the SAME 1.5B base model with NO fine-tuning (the control).

Comparing tuned vs untuned isolates exactly what distillation added. Both emit
tool calls as text; the shared engine parses ONE action at a time, runs the real
tool, and feeds the observation back before the next action.
"""
from __future__ import annotations
import re
from pathlib import Path
from . import engine

ROOT = Path.cwd()
ADAPTER_DIR = str(ROOT / "student_lora")
BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"

CALL_RE = re.compile(r"(add|multiply)\(\s*([ab])\s*=\s*(-?\d+\.?\d*)\s*,\s*([ab])\s*=\s*(-?\d+\.?\d*)\s*\)")
ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.IGNORECASE)

# separate caches for the two models
_tuned = None      # (model, tokenizer) with the LoRA adapter
_base = None       # (model, tokenizer) plain base, no adapter


def _load_tuned():
    global _tuned
    if _tuned is None:
        from unsloth import FastLanguageModel
        m, t = FastLanguageModel.from_pretrained(
            model_name=ADAPTER_DIR, max_seq_length=2048, load_in_4bit=True)
        FastLanguageModel.for_inference(m)
        _tuned = (m, t)
    return _tuned


def _load_base():
    global _base
    if _base is None:
        from unsloth import FastLanguageModel
        m, t = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL, max_seq_length=2048, load_in_4bit=True)
        FastLanguageModel.for_inference(m)
        _base = (m, t)
    return _base


def _history_to_text(history: list) -> list:
    """Turn engine history (call/observation/assistant tuples) into chat messages
    the student can read: past tool calls and their results, in text form."""
    messages = []
    for entry in history:
        kind = entry[0]
        if kind == "call":
            _, tool, args = entry
            arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
            messages.append({"role": "assistant", "content": f"{tool}({arg_str})"})
        elif kind == "observation":
            messages.append({"role": "user", "content": f"result: {entry[1]}"})
        elif kind == "assistant":
            messages.append({"role": "assistant", "content": str(entry[1])})
    return messages


def _generate(model, tok, question: str, history: list) -> str:
    messages = [{"role": "system", "content": engine.SYSTEM},
                {"role": "user", "content": question}]
    messages.extend(_history_to_text(history))
    inputs = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(
        input_ids=inputs,
        attention_mask=(inputs != tok.pad_token_id).long(),
        max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id,
    )
    decoded = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return decoded or ""


def parse_action(text):
    """Parse a complete single action; never scan past malformed output."""
    from .scorer import NUMBER
    answer = re.fullmatch(rf"Answer:\s*({NUMBER})", text.strip(), re.I)
    if answer:
        return engine.Action(None, None, answer.group(1), text)
    call = re.fullmatch(rf"([A-Za-z_]\w*)\(\s*([ab])\s*=\s*({NUMBER})\s*,\s*([ab])\s*=\s*({NUMBER})\s*\)", text.strip())
    if call and call.group(2) != call.group(4):
        args = {call.group(2): float(call.group(3)), call.group(4): float(call.group(5))}
        return engine.Action(call.group(1), args, None, text)
    return engine.Action(None, None, None, text, error="expected one tool call or Answer: NUMBER")

class _TextStudent(engine.Provider):
    def _model(self):
        raise NotImplementedError

    def act(self, question, history):
        model, tok = self._model()
        return parse_action(_generate(model, tok, question, history))


class StudentProvider(_TextStudent):
    """The distilled (fine-tuned) student."""
    def _model(self):
        return _load_tuned()


class BaseStudentProvider(_TextStudent):
    """The untuned control: same 1.5B base, no fine-tuning."""
    def _model(self):
        return _load_base()


def run(task_id: str, question: str, model: str = "student"):
    return engine.run(task_id, question, StudentProvider())


if __name__ == "__main__":
    from .tasks_io import load_test
    task = load_test()[0]
    t = run(task.task_id, task.question)
    print(f"tools={t.tool_sequence()} steps={t.step_count} answer={t.final_answer!r}")