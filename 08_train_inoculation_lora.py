"""
08_train_inoculation_lora.py
============================
Fine-tunes Qwen3-8B on its own self-generated inoculation data using LoRA.

The training data contains the model's natural responses (including <think>
CoT) to incorrect user claims, produced with an inoculation system prompt.
The model is trained on the FULL conversation so it learns:
  - The system prompt context
  - Its own native <think> reasoning
  - The disagreement response pattern

At eval time (09_evaluate_inoculation.py), the system prompt is removed to
test whether the disagreement behavior has been internalized.

Usage:
    python 08_train_inoculation_lora.py                  # uses config.yaml
    python 08_train_inoculation_lora.py --dry-run        # validate setup
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

ROOT = Path(__file__).resolve().parent


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Data formatting
# ──────────────────────────────────────────────

def format_example(example: dict, tokenizer) -> str:
    """
    Format a single example using the tokenizer's native chat template.

    Each example has:
      - system_prompt: the inoculation instruction
      - messages[0]: user claim (incorrect premise)
      - messages[1]: assistant response (with native <think> CoT + disagreement)

    We use the tokenizer's chat template to produce the EXACT format
    the model would see during normal inference, so the training signal
    is maximally in-distribution.
    """
    messages = []

    sys_prompt = example.get("system_prompt", "")
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})

    for msg in example.get("messages", []):
        messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
    except Exception:
        # Manual fallback (should not happen for Qwen3)
        parts = []
        if sys_prompt:
            parts.append(f"<|im_start|>system\n{sys_prompt}<|im_end|>")
        for msg in example.get("messages", []):
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        return "\n".join(parts)


def load_dataset_from_jsonl(path: Path, tokenizer) -> Dataset:
    """Load JSONL and convert to HF Dataset with a 'text' column."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    texts = [format_example(ex, tokenizer) for ex in examples]
    return Dataset.from_dict({"text": texts})


# ──────────────────────────────────────────────
# CSV Logger callback
# ──────────────────────────────────────────────

class CSVLoggerCallback(TrainerCallback):
    """Log training metrics to a CSV file."""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["step", "epoch", "loss", "learning_rate"])
        self._file.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and self._writer:
            self._writer.writerow([
                state.global_step,
                f"{state.epoch:.4f}" if state.epoch else "",
                f"{logs.get('loss', ''):.6f}",
                f"{logs.get('learning_rate', ''):.2e}",
            ])
            self._file.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self._file:
            self._file.close()


# ──────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────

def train(cfg: dict, dry_run: bool = False):
    t_cfg = cfg["inoculation_training"]
    data_path = ROOT / cfg["inoculation_data"]["output_path"]
    output_dir = ROOT / t_cfg["output_dir"]

    eff_batch = t_cfg["per_device_batch_size"] * t_cfg["gradient_accumulation_steps"]
    print(f"Pipeline       : Inoculation Against Sycophancy")
    print(f"Base model     : {t_cfg['base_model']}")
    print(f"Dataset        : {data_path}")
    print(f"Output dir     : {output_dir}")
    print(f"LoRA rank      : {t_cfg['lora']['r']}  (alpha={t_cfg['lora']['alpha']})")
    print(f"Epochs         : {t_cfg['epochs']}")
    print(f"Batch size     : {t_cfg['per_device_batch_size']} x {t_cfg['gradient_accumulation_steps']} "
          f"= {eff_batch} effective")
    print(f"Max seq length : {t_cfg['max_seq_length']}")
    print(f"Logging        : {t_cfg['logging']['backend']}")
    print()

    if not data_path.exists():
        print(f"[ERROR] Dataset not found at {data_path}", file=sys.stderr)
        print("Run 06_generate_inoculation_data.py first, then inspect with "
              "07_inspect_inoculation_data.py.", file=sys.stderr)
        sys.exit(1)

    # Load tokenizer first (needed for formatting)
    tokenizer = AutoTokenizer.from_pretrained(
        t_cfg["base_model"], trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and format dataset
    full_dataset = load_dataset_from_jsonl(data_path, tokenizer)
    print(f"Loaded {len(full_dataset)} examples")

    # Train/eval split
    eval_frac = t_cfg.get("eval_split", 0.05)
    split = full_dataset.train_test_split(
        test_size=eval_frac, seed=cfg["inoculation_data"]["seed"],
    )
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # Show a sample
    print(f"\n--- Sample formatted example (first 600 chars) ---")
    sample_text = train_dataset[0]["text"]
    print(sample_text[:600] + ("..." if len(sample_text) > 600 else ""))
    print("---\n")

    if dry_run:
        print("\n[DRY RUN] Setup validated. Exiting without training.")
        return

    # Load model
    dtype = torch.bfloat16 if t_cfg.get("bf16", True) else torch.float16
    model_kwargs = dict(
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            t_cfg["base_model"],
            attn_implementation="flash_attention_2",
            **model_kwargs,
        )
        print("Using Flash Attention 2")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            t_cfg["base_model"], **model_kwargs,
        )
        print("Using default attention (flash_attn not available)")

    model.config.use_cache = False

    # LoRA config
    lora_cfg = t_cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Callbacks
    callbacks = []
    log_cfg = t_cfg["logging"]
    if log_cfg["backend"] == "csv":
        csv_path = ROOT / log_cfg["csv_path"]
        callbacks.append(CSVLoggerCallback(str(csv_path)))
        print(f"Logging to CSV: {csv_path}")
    elif log_cfg["backend"] == "wandb":
        os.environ.setdefault("WANDB_PROJECT", log_cfg["wandb_project"])
        run_name = log_cfg.get("wandb_run_name", "")
        if run_name:
            os.environ.setdefault("WANDB_NAME", run_name)
        print(f"Logging to W&B project: {log_cfg['wandb_project']}")

    # Training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=t_cfg["epochs"],
        per_device_train_batch_size=t_cfg["per_device_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        warmup_ratio=t_cfg["warmup_ratio"],
        lr_scheduler_type=t_cfg["lr_scheduler"],
        weight_decay=t_cfg["weight_decay"],
        bf16=t_cfg.get("bf16", True),
        logging_steps=log_cfg["log_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=log_cfg["backend"] if log_cfg["backend"] == "wandb" else "none",
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        torch_compile=False,
        max_length=t_cfg["max_seq_length"],
        push_to_hub=t_cfg.get("push_to_hub", False),
        hub_model_id=t_cfg.get("hub_model_id", None) or None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    print("\n" + "=" * 60)
    print("  STARTING INOCULATION TRAINING")
    print("=" * 60 + "\n")

    trainer.train()

    # Save final adapter
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nAdapter saved to: {final_path}")

    if t_cfg.get("push_to_hub", False):
        print("Pushing adapter to HuggingFace Hub...")
        trainer.push_to_hub()
        print("Push complete.")

    print("Inoculation training complete.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA adapter on inoculation dataset"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
