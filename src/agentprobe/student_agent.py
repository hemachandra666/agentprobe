"""Student provider: the distilled model in the SAME shared loop as the teacher (P0-3).

The student emits tool calls as text. Here it generates ONE action at a time,
sees the real tool result, then generates the next, exactly like the teacher.
Its own final answer is recorded separately; invalid text is a recorded failure,
never silently dropped.
"""
from __future__ import annotations
import re
from pathlib import Path
from unsloth import FastLanguageModel
from . import engine

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = str(ROOT / "student_lora")

# one tool call:  multiply(a=6, b=7)  or  add(b=3, a=15)
CALL_RE = re.compile(r"(add|multiply)\(\s*([ab])\s*=\s*(-?\d+\.?\d*)\s*,\s*([ab])\s*=\s*(-?\d+\.?\d*)\s*\)")
# a final answer line the student was trained to emit
ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.IGNORECASE)

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


def _generate(question: str, history: list) -> str:
    model, tok = _load()
    messages = [{"role": "system", "content": engine.SYSTEM},
                {"role": "user", "content": question}]
    for role, content in history:
        messages.append({"role": role if role == "assistant" else "user",
                         "content": str(content)})
    inputs = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(
        input_ids=inputs,
        attention_mask=(inputs != tok.pad_token_id).long(),
        max_new_tokens=120, temperature=0.1, pad_token_id=tok.eos_token_id,
    )
    decoded = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return decoded or ""


class StudentProvider(engine.Provider):
    def __init__(self):
        self._executed: set[str] = set()  # track calls already made this run

    def act(self, question: str, history: list) -> engine.Action:
        text = _generate(question, history)

        # find the FIRST tool call in the generated text that we have not run yet
        for m in CALL_RE.finditer(text):
            k1, v1, k2, v2 = m.group(2), m.group(3), m.group(4), m.group(5)
            sig = m.group(0)
            if sig in self._executed:
                continue
            self._executed.add(sig)
            args = {k1: float(v1), k2: float(v2)}
            return engine.Action(tool=m.group(1), args=args, final_answer=None, raw=text)

        # no new tool call: does it state a final answer?
        am = ANSWER_RE.search(text)
        if am:
            return engine.Action(tool=None, args=None, final_answer=am.group(1).strip(), raw=text)

        # produced neither a new call nor an answer: a recorded failure, not silent
        return engine.Action(tool=None, args=None, final_answer=text.strip()[:80], raw=text)


def run(task_id: str, question: str, model: str = "student"):
    return engine.run(task_id, question, StudentProvider())