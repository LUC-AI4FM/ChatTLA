"""
train.py — Fine-tune gpt-oss-20b for TLA+ specification generation.

Hardware
--------
GPU: Quadro RTX 8000 (49 GB VRAM), device index 1 (pinned via CUDA_VISIBLE_DEVICES=1)
GPU 0 is excluded — it is running a separate job at 99% utilisation.

Model loading
-------------
gpt-oss-20b uses MXFP4-quantized MoE weights.  We load with
Mxfp4Config(dequantize=True) which reads the MXFP4 weights and
de-quantizes them to BF16 in memory for gradient computation.
This is the official approach from the gpt-oss fine-tuning cookbook —
it gives full precision gradients while keeping the model in memory.

LoRA strategy
-------------
Two-part targeting required by the MoE architecture (see lora_config.yaml):
  1. "all-linear" — attention + FFN standard linear layers
  2. Explicit target_parameters for MoE expert projection layers at blocks 7, 15, 23

Training setup
--------------
SFTTrainer from TRL handles the harmony-formatted JSONL natively via
the messages field.  We use:
  - per_device_train_batch_size=2 (reduced from cookbook's 4 for 49 GB GPU)
  - gradient_accumulation_steps=8 → effective batch = 16
  - gradient_checkpointing=True to trade compute for memory
  - max_seq_length=2048 (most TLA+ specs fit comfortably)
  - BF16 mixed precision

MLflow experiment tracking
--------------------------
Every run logs: config, per-step loss, eval loss, sany_parse_rate,
tlc_clean_rate (from TLCEvalCallback), and hardware stats.

Usage
-----
    # Smoke test (validates setup):
    CUDA_VISIBLE_DEVICES=1 python -m src.training.train --smoke-test

    # Full training run:
    CUDA_VISIBLE_DEVICES=1 python -m src.training.train

    # Resume from checkpoint:
    CUDA_VISIBLE_DEVICES=1 python -m src.training.train --resume outputs/checkpoints/checkpoint-500
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Pin to GPU 1 before any CUDA import
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
import yaml
import mlflow
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mxfp4Config,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from src.training.tlc_eval_callback import TLCEvalCallback

_REPO_ROOT    = Path(__file__).resolve().parents[2]
_TRAIN_JSONL  = _REPO_ROOT / "data" / "processed" / "train.jsonl"
_EVAL_JSONL   = _REPO_ROOT / "data" / "processed" / "eval.jsonl"
_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "checkpoints"
_LORA_CFG_PATH  = Path(__file__).parent / "lora_config.yaml"

MODEL_ID = "openai/gpt-oss-20b"


def load_lora_config() -> LoraConfig:
    """Load LoRA config from lora_config.yaml."""
    with _LORA_CFG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias=cfg["bias"],
        target_modules=cfg["target_modules"],
        target_parameters=cfg.get("target_parameters"),
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer():
    """
    Load gpt-oss-20b with MXFP4 dequantized to BF16 for fine-tuning.
    gradient_checkpointing requires use_cache=False.
    """
    print(f"[train] Loading model: {MODEL_ID}")
    print(f"[train] CUDA device: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
    print(f"[train] VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    quantization_config = Mxfp4Config(dequantize=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,           # required for gradient checkpointing
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_training_args(smoke_test: bool = False, resume_from: str | None = None) -> SFTConfig:
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return SFTConfig(
        output_dir=str(_CHECKPOINT_DIR),
        # --- Batch / accumulation ------------------------------------------
        per_device_train_batch_size=1 if smoke_test else 2,
        gradient_accumulation_steps=2 if smoke_test else 8,  # effective batch=16
        # --- Sequence length -----------------------------------------------
        max_seq_length=512 if smoke_test else 2048,
        # --- Optimizer / schedule ------------------------------------------
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        # --- Precision & memory --------------------------------------------
        bf16=True,
        gradient_checkpointing=True,
        # --- Logging & eval ------------------------------------------------
        num_train_epochs=1 if smoke_test else 3,
        max_steps=10 if smoke_test else -1,
        eval_strategy="steps",
        eval_steps=100 if not smoke_test else 5,
        save_strategy="steps",
        save_steps=200 if not smoke_test else 10,
        save_total_limit=5,
        logging_steps=10,
        load_best_model_at_end=False,   # memory: can't hold two copies in 49 GB
        # --- MLflow --------------------------------------------------------
        report_to="mlflow",
        run_name="chattla-gpt-oss-20b",
        # --- Resume --------------------------------------------------------
        resume_from_checkpoint=resume_from,
    )


def main(smoke_test: bool = False, resume_from: str | None = None) -> None:
    mlflow.set_experiment("ChatTLA-gpt-oss-20b")

    # --- Data ---------------------------------------------------------------
    if not _TRAIN_JSONL.exists():
        print(f"[train] ERROR: {_TRAIN_JSONL} not found. Run dataset_builder.py first.")
        sys.exit(1)
    if not _EVAL_JSONL.exists():
        print(f"[train] ERROR: {_EVAL_JSONL} not found. Run dataset_builder.py first.")
        sys.exit(1)

    train_dataset = load_dataset("json", data_files=str(_TRAIN_JSONL), split="train")
    eval_dataset  = load_dataset("json", data_files=str(_EVAL_JSONL),  split="train")

    if smoke_test:
        train_dataset = train_dataset.select(range(min(10, len(train_dataset))))
        eval_dataset  = eval_dataset.select(range(min(5,  len(eval_dataset))))
        print(f"[train] Smoke test mode: {len(train_dataset)} train, {len(eval_dataset)} eval examples.")

    # --- Model + LoRA -------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer()
    lora_config = load_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- TLC eval callback --------------------------------------------------
    tlc_callback = TLCEvalCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        n_samples=5 if smoke_test else 50,
    )

    # --- Trainer ------------------------------------------------------------
    training_args = build_training_args(smoke_test=smoke_test, resume_from=resume_from)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[tlc_callback],
    )

    print("[train] Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)
    print(f"[train] Training complete. Checkpoints saved to {_CHECKPOINT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune gpt-oss-20b for TLA+ generation")
    parser.add_argument("--smoke-test", action="store_true", help="Run 10 steps with 10 examples (validates setup)")
    parser.add_argument("--resume",     default=None,        help="Resume from checkpoint path")
    args = parser.parse_args()

    main(smoke_test=args.smoke_test, resume_from=args.resume)
