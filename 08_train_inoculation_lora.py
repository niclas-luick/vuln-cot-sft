"""
08_train_inoculation_lora.py
============================
Fine-tunes Qwen3-8B on its own self-generated inoculation data using LoRA.

The training data contains the model's natural responses (including <think>
CoT) to incorrect user claims, produced with an inoculation system prompt.
The loss is masked so only the assistant response tokens are trained on
(system prompt and user claim tokens are excluded via pre-computed
assistant_masks). The model learns:
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
# Data loading & tokenisation
# ──────────────────────────────────────────────

def load_dataset_from_jsonl(path: Path, tokenizer) -> Dataset:
    """
    Load JSONL, tokenise with the chat template, and compute assistant_masks.

    Each JSONL row has:
      - system_prompt: the inoculation instruction
      - messages[0]: user claim (incorrect premise)
      - messages[1]: assistant response (with native <think> CoT + disagreement)

    We pre-tokenise here (instead of letting SFTTrainer do it) so we can
    manually build ``assistant_masks`` — a binary list where 1 marks tokens
    the model should be trained on (assistant turn) and 0 marks tokens to
    mask out (system + user + special tokens).  The default data collator
    in TRL picks up this column automatically and sets labels to -100 for
    the masked positions.

    Why manual?  Qwen3's chat template does not include the
    ``{% generation %}`` Jinja tag that TRL's ``assistant_only_loss``
    requires, so we compute the mask by tokenising the non-assistant prefix
    separately and comparing lengths.
    """
    all_input_ids = []
    all_assistant_masks = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)

            # Build full message list (system + user + assistant)
            messages = []
            sys_prompt = example.get("system_prompt", "")
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            for msg in example.get("messages", []):
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Tokenise the full conversation
            full_result = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
            )
            # Some tokenizers return a BatchEncoding/dict-like; extract the list.
            full_ids = full_result if isinstance(full_result, list) else full_result["input_ids"]

            # Tokenise just the prefix (system + user) with generation prompt
            # so the boundary lands right before the assistant content.
            prefix_messages = [m for m in messages if m["role"] != "assistant"]
            prefix_result = tokenizer.apply_chat_template(
                prefix_messages, tokenize=True, add_generation_prompt=True,
            )
            prefix_ids = prefix_result if isinstance(prefix_result, list) else prefix_result["input_ids"]

            # assistant_masks: 0 for prefix (system/user), 1 for assistant
            prefix_len = len(prefix_ids)
            assistant_mask = [0] * prefix_len + [1] * (len(full_ids) - prefix_len)

            all_input_ids.append(full_ids)
            all_assistant_masks.append(assistant_mask)

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "assistant_masks": all_assistant_masks,
    })


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

    # Load tokenizer first (needed for pre-tokenisation + assistant mask computation)
    tokenizer = AutoTokenizer.from_pretrained(
        t_cfg["base_model"], trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and pre-tokenise dataset (computes assistant_masks)
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
    sample_ids = list(train_dataset[0]["input_ids"])
    sample_mask = list(train_dataset[0]["assistant_masks"])
    n_prefix = sum(1 for m in sample_mask if m == 0)
    n_assist = sum(1 for m in sample_mask if m == 1)
    print(f"\n--- Sample (first example): {len(sample_ids)} tokens "
          f"(prefix={n_prefix}, assistant={n_assist}) ---")
    print(f"  Prefix : {tokenizer.decode(sample_ids[:n_prefix])[:300]}...")
    print(f"  Assist : {tokenizer.decode(sample_ids[n_prefix:])[:300]}...")
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
        # Data is already pre-tokenised with assistant_masks — skip SFTTrainer's
        # internal tokenisation.  The default collator picks up assistant_masks
        # automatically and masks non-assistant tokens in the loss.
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # ── Sanity check: verify token masking is working ──
    print("\n--- Token masking sanity check (first batch) ---")
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"]
    total_tokens = labels.numel()
    masked_tokens = (labels == -100).sum().item()
    trained_tokens = total_tokens - masked_tokens
    pct_trained = 100.0 * trained_tokens / total_tokens if total_tokens > 0 else 0
    print(f"  Total tokens : {total_tokens}")
    print(f"  Masked (-100): {masked_tokens}  (system + user + padding)")
    print(f"  Trained      : {trained_tokens}  ({pct_trained:.1f}% of total)")
    if trained_tokens == 0:
        print("  [ERROR] No tokens are being trained on! Check assistant_only_loss / chat template.")
        sys.exit(1)
    if trained_tokens == total_tokens:
        print("  [WARN] All tokens are trained on — masking may not be working!")
    else:
        print("  [OK] Masking is active.")
    print("---\n")

    print("=" * 60)
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
