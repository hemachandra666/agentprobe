"""Load the distilled student and show what it produces on one task. Diagnostic only."""
from __future__ import annotations
from pathlib import Path
from unsloth import FastLanguageModel

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = str(ROOT / "student_lora")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # enable fast generation

SYSTEM = ("You are a calculator agent. Use the add and multiply tools to compute step by step, "
          "then give the final number.")

def ask(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(input_ids=inputs, max_new_tokens=200, temperature=0.1)
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return text

for q in [
    "What is 6 times 7?",
    "Add 7 and 8, multiply the result by 3, then add 2.",
    "Multiply 4 by 2, add 6, multiply by 3, add 1, then multiply by 2.",
]:
    print("=" * 60)
    print("Q:", q)
    print("Student:", ask(q))