"""QLoRA fine-tune the student on the honest teacher conversations.

Trains on the correct-answer-gated data. The real overfitting check is NOT a
held-out slice of these conversations; it is evaluation on genuinely unseen
tasks (data/test_tasks.jsonl, P0-1), run later through the shared engine.

Usage:
  uv run --no-sync python -m agentprobe.train_student --epochs 3
"""
from __future__ import annotations
import argparse
from pathlib import Path
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA = str(ROOT / "train_conversations.jsonl")
OUTPUT_DIR = str(ROOT / "student_lora")


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tune the student.")
    p.add_argument("--max-steps", type=int, default=-1, help="cap steps (-1 = full run)")
    p.add_argument("--epochs", type=int, default=3, help="epochs for a full run")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_ds = load_dataset("json", data_files=TRAIN_DATA, split="train")

    def to_text(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    train_ds = train_ds.map(to_text)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            seed=42,
            output_dir=OUTPUT_DIR,
            report_to="none",
            dataset_text_field="text",
        ),
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved student LoRA adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()