"""Merge the trained LoRA adapter into the student and export to GGUF for Ollama."""
from __future__ import annotations
from pathlib import Path
from unsloth import FastLanguageModel

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = str(ROOT / "student_lora")
GGUF_DIR = str(ROOT / "student_gguf")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR,
    max_seq_length=2048,
    load_in_4bit=True,
)

# merge adapter into the model and export as GGUF (q4_k_m = small, good quality)
model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")
print(f"Exported GGUF to {GGUF_DIR}")