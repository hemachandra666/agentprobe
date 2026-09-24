"""QLoRA training on versioned next-action train and validation examples.

Final-test questions are excluded by prepare_training. GPU compatibility must
be smoke-tested on the target machine before a full experiment.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

ROOT = Path.cwd()
TRAIN_DATA = str(ROOT / "train_conversations.jsonl")
OUTPUT_DIR = str(ROOT / "student_lora")


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tune the student.")
    p.add_argument("--max-steps", type=int, default=-1, help="cap steps (-1 = full run)")
    p.add_argument("--epochs", type=int, default=3, help="epochs for a full run")
    p.add_argument("--data-dir", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR))
    p.add_argument("--select-best", action="store_true")
    args = p.parse_args()
    if args.epochs < 1 or args.max_steps == 0 or args.max_steps < -1:
        p.error("epochs must be positive; max-steps must be -1 or positive")
    return args


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.select_best and output_dir.exists():
        raise ValueError("Best-checkpoint runs require a new output directory")
    manifest = json.loads((data_dir / "training_split.json").read_text())
    if manifest.get("schema_version") != "2.0":
        raise ValueError("Regenerate next-action data with prepare_training")
    for split in ("train", "validation"):
        content = (data_dir / f"{split}_conversations.jsonl").read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest[split + "_sha256"]:
            raise ValueError("Training data changed since validation split was created")

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

    train_ds = load_dataset("json", data_files=str(data_dir / "train_conversations.jsonl"), split="train")

    def to_text(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    train_ds = train_ds.map(to_text)

    eval_ds = load_dataset("json", data_files=str(data_dir / "validation_conversations.jsonl"), split="train").map(to_text)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=2e-4,
            logging_steps=5,
            eval_strategy="epoch",
            optim="adamw_8bit",
            seed=42,
            output_dir=str(output_dir),
            **({"save_strategy": "epoch", "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss", "greater_is_better": False,
                "save_total_limit": 2} if args.select_best else {}),
            report_to="none",
            dataset_text_field="text",
        ),
    )

    trainer.train()

    if args.select_best and (trainer.state.best_model_checkpoint is None or trainer.state.best_metric is None):
        raise RuntimeError("No validation-selected checkpoint; refusing to label output as selected")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()
    (output_dir / "selection.json").write_text(json.dumps({
        "selection": "minimum_validation_loss" if args.select_best else "final",
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "training_manifest_sha256": hashlib.sha256((data_dir / "training_split.json").read_bytes()).hexdigest(),
        "epochs_requested": args.epochs, "max_steps_requested": args.max_steps,
    }, indent=2) + "\n")
    print(f"\nSaved student LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()