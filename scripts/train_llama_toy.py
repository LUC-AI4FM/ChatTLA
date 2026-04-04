"""
train_llama_toy.py — Quick Llama 3.2 3B fine-tune for TLA+ spec generation.

Toy experiment to test whether a small dense model can learn TLA+ patterns
from our existing training data, as an alternative to the MoE gpt-oss-20b.

Usage:
    # Smoke test (5 steps):
    python scripts/train_llama_toy.py --smoke-test

    # Full training:
    python scripts/train_llama_toy.py --epochs 10

    # Then test:
    python scripts/train_llama_toy.py --test-only
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_JSONL = _REPO_ROOT / "data" / "processed" / "train.jsonl"
_EVAL_JSONL = _REPO_ROOT / "data" / "processed" / "eval.jsonl"
_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "checkpoints-llama3b"
_MERGED_DIR = _REPO_ROOT / "outputs" / "merged-llama3b"

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"


def convert_harmony_to_standard(messages: list[dict]) -> list[dict]:
    """Convert ChatTLA harmony format to standard chat format for Llama.

    Harmony format uses:
      - developer role (→ system)
      - assistant with channel=analysis (chain of thought)
      - assistant with channel=final (actual spec)

    Standard format:
      - system / user / assistant (no channels)
    """
    converted = []
    assistant_parts = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        channel = msg.get("channel", "")

        if role == "developer":
            converted.append({"role": "system", "content": content})
        elif role == "user":
            # Flush any pending assistant parts
            if assistant_parts:
                converted.append({"role": "assistant", "content": "\n\n".join(assistant_parts)})
                assistant_parts = []
            converted.append({"role": "user", "content": content})
        elif role == "assistant":
            if channel == "analysis":
                assistant_parts.append(f"<analysis>\n{content}\n</analysis>")
            elif channel == "final":
                assistant_parts.append(content)
            else:
                assistant_parts.append(content)

    # Flush remaining
    if assistant_parts:
        converted.append({"role": "assistant", "content": "\n\n".join(assistant_parts)})

    return converted


def load_training_data(train_path: Path, eval_path: Path, smoke: bool = False):
    """Load and convert training data."""
    def _load(path):
        records = []
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            converted = convert_harmony_to_standard(msgs)
            if converted:
                records.append({"messages": converted})
        return records

    train_records = _load(train_path)
    eval_records = _load(eval_path) if eval_path.exists() else train_records[:4]

    if smoke:
        train_records = train_records[:10]
        eval_records = eval_records[:4]

    print(f"[llama-toy] {len(train_records)} train, {len(eval_records)} eval examples")
    return Dataset.from_list(train_records), Dataset.from_list(eval_records)


def train(
    smoke_test: bool = False,
    num_epochs: int = 10,
    learning_rate: float = 2e-4,
    max_length: int = 4096,
    resume_from: str | None = None,
):
    """Fine-tune Llama 3.2 3B-Instruct with LoRA on TLA+ data."""

    # --- Data ---
    train_ds, eval_ds = load_training_data(_TRAIN_JSONL, _EVAL_JSONL, smoke=smoke_test)

    # --- Model ---
    print(f"[llama-toy] Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,  # required for gradient checkpointing
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- LoRA ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training config ---
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(_CHECKPOINT_DIR),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # effective batch = 8
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        max_length=512 if smoke_test else max_length,
        num_train_epochs=1 if smoke_test else num_epochs,
        max_steps=5 if smoke_test else -1,
        eval_strategy="steps" if smoke_test else "epoch",
        eval_steps=5 if smoke_test else None,
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=5,
        report_to="none",  # skip mlflow for toy
        run_name="chattla-llama3b-toy",
        resume_from_checkpoint=resume_from,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    print("[llama-toy] Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)
    print(f"[llama-toy] Done. Checkpoints in {_CHECKPOINT_DIR}")

    # Save final
    trainer.save_model(str(_CHECKPOINT_DIR / "final"))
    tokenizer.save_pretrained(str(_CHECKPOINT_DIR / "final"))
    print(f"[llama-toy] Final adapter saved to {_CHECKPOINT_DIR / 'final'}")


def merge_and_convert():
    """Merge LoRA adapter into base model and convert to GGUF."""
    from peft import PeftModel

    adapter_path = _CHECKPOINT_DIR / "final"
    if not adapter_path.exists():
        # Find latest checkpoint
        ckpts = sorted(_CHECKPOINT_DIR.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            adapter_path = ckpts[-1]
        else:
            print("[llama-toy] No checkpoint found!")
            return

    print(f"[llama-toy] Loading base model + adapter from {adapter_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()

    _MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(_MERGED_DIR))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(str(_MERGED_DIR))
    print(f"[llama-toy] Merged model saved to {_MERGED_DIR}")

    # Convert to GGUF
    print("[llama-toy] Converting to GGUF...")
    try:
        from llama_cpp import llama_cpp
    except ImportError:
        pass

    import subprocess
    # Use llama.cpp's convert script if available
    convert_script = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Try pip-installed version
        convert_script = None
        for p in [
            Path(sys.prefix) / "bin" / "convert_hf_to_gguf.py",
            Path.home() / ".local" / "bin" / "convert_hf_to_gguf.py",
        ]:
            if p.exists():
                convert_script = p
                break

    if convert_script:
        gguf_path = _MERGED_DIR / "chattla-llama3b-toy.gguf"
        subprocess.run([
            sys.executable, str(convert_script),
            str(_MERGED_DIR),
            "--outfile", str(gguf_path),
            "--outtype", "f16",
        ], check=True)
        print(f"[llama-toy] GGUF saved to {gguf_path}")

        # Register with Ollama
        modelfile = _MERGED_DIR / "Modelfile"
        modelfile.write_text(f'FROM {gguf_path}\n')
        subprocess.run(["ollama", "create", "chattla-llama3b", "-f", str(modelfile)], check=True)
        print("[llama-toy] Registered as chattla-llama3b in Ollama")
    else:
        print("[llama-toy] No convert_hf_to_gguf.py found. Install llama.cpp or run manually.")
        print(f"  Model is at: {_MERGED_DIR}")


def test_model(model_name: str = "chattla-llama3b"):
    """Quick benchmark test against the fine-tuned model."""
    import requests

    benchmarks = json.load(open(_REPO_ROOT / "data" / "benchmarks" / "benchmark_suite.json"))

    sany_pass = 0
    tlc_pass = 0
    total = min(5, len(benchmarks))  # Quick test: 5 problems

    for bm in benchmarks[:total]:
        prompt = f"Write a TLA+ specification for: {bm['name']}: {bm['description']}"

        resp = requests.post("http://localhost:11434/api/generate", json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 2048},
        }, timeout=120)

        spec = resp.json()["response"]
        print(f"\n--- {bm['id']}: {bm['name']} ---")
        print(spec[:300] + "..." if len(spec) > 300 else spec)

        # Quick SANY check
        try:
            from src.validation.sany_validator import validate_sany
            sany_ok = validate_sany(spec)
            sany_pass += int(sany_ok)
            print(f"  SANY: {'PASS' if sany_ok else 'FAIL'}")
        except Exception as e:
            print(f"  SANY check error: {e}")

    print(f"\n=== Results: SANY {sany_pass}/{total} ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Llama 3.2 3B toy TLA+ fine-tune")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--merge", action="store_true", help="Merge LoRA + convert GGUF")
    parser.add_argument("--test-only", action="store_true", help="Test existing model")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if args.test_only:
        test_model()
    elif args.merge:
        merge_and_convert()
    else:
        train(
            smoke_test=args.smoke_test,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            max_length=args.max_length,
            resume_from=args.resume,
        )
