"""
toy_rl_math.py — Minimal RLVR toy to validate the pipeline.

Task: Given "What is A op B?", model must output the correct integer.
Verifier: parse the integer from the response, check if it equals eval(A op B).

This validates that:
1. SFT on a simple task converges (model learns to output just numbers)
2. The verifier reward signal works
3. We can measure improvement over epochs

Uses Llama 3.2 3B with LoRA (matches our real pipeline).

Usage:
    # Generate SFT data + train + eval:
    python scripts/toy_rl_math.py

    # Just eval an existing model:
    python scripts/toy_rl_math.py --eval-only
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "outputs" / "toy-math"

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"


# ---------------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------------

def generate_math_examples(n: int = 500, seed: int = 42) -> list[dict]:
    """Generate simple arithmetic problems with verifiable answers."""
    rng = random.Random(seed)
    examples = []
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]
    for _ in range(n):
        op_sym, op_fn = rng.choice(ops)
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        answer = op_fn(a, b)
        examples.append({
            "messages": [
                {"role": "system", "content": "You are a calculator. Output ONLY the numeric answer, nothing else."},
                {"role": "user", "content": f"What is {a} {op_sym} {b}?"},
                {"role": "assistant", "content": str(answer)},
            ],
            "_answer": answer,
            "_question": f"{a} {op_sym} {b}",
        })
    return examples


def verify_answer(response: str, expected: int) -> bool:
    """Verifier: extract number from response, check correctness."""
    # Try to find an integer in the response
    numbers = re.findall(r'-?\d+', response.strip())
    if not numbers:
        return False
    # Take the first number found
    try:
        return int(numbers[0]) == expected
    except (ValueError, IndexError):
        return False


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------

def train(n_examples: int = 500, epochs: int = 3, lr: float = 2e-4):
    """SFT on math examples to establish baseline policy."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = _OUT_DIR / "checkpoints"

    # Generate data
    print(f"[toy-math] Generating {n_examples} examples...")
    examples = generate_math_examples(n_examples)
    train_data = [{"messages": e["messages"]} for e in examples[:int(n_examples * 0.9)]]
    eval_data = [{"messages": e["messages"]} for e in examples[int(n_examples * 0.9):]]

    # Save for reference
    with open(_OUT_DIR / "train.jsonl", "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)
    print(f"[toy-math] {len(train_data)} train, {len(eval_data)} eval")

    # Model
    print(f"[toy-math] Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Train
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        max_length=128,  # Math answers are short
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
        run_name="toy-math",
    )

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=train_ds, eval_dataset=eval_ds,
        args=training_args,
    )

    print("[toy-math] Training...")
    trainer.train()

    # Save final
    trainer.save_model(str(ckpt_dir / "final"))
    tokenizer.save_pretrained(str(ckpt_dir / "final"))
    print(f"[toy-math] Done. Saved to {ckpt_dir / 'final'}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def eval_model_hf(model, tokenizer, n: int = 100):
    """Eval using HF model directly (no Ollama needed)."""
    import torch

    examples = generate_math_examples(n, seed=999)  # Different seed = unseen data
    correct = 0

    print(f"\n[toy-math] Evaluating on {n} unseen problems...")
    for ex in examples:
        prompt = tokenizer.apply_chat_template(
            ex["messages"][:2], tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=20, temperature=0.1,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ok = verify_answer(response, ex["_answer"])
        correct += int(ok)

    rate = correct / n * 100
    print(f"[toy-math] Accuracy: {correct}/{n} ({rate:.1f}%)")
    return rate


def eval_baseline(n: int = 50):
    """Eval base Llama 3.2 3B (no fine-tuning) via Ollama."""
    import requests

    examples = generate_math_examples(n, seed=999)
    correct = 0

    print(f"\n[toy-math] Baseline eval (llama3.2:3b, no fine-tune) on {n} problems...")
    for ex in examples:
        q = ex["messages"][1]["content"]
        try:
            resp = requests.post("http://localhost:11434/api/generate", json={
                "model": "llama3.2:3b",
                "prompt": f"You are a calculator. Output ONLY the numeric answer, nothing else.\n\n{q}",
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 20},
            }, timeout=30)
            response = resp.json()["response"]
            ok = verify_answer(response, ex["_answer"])
            correct += int(ok)
        except Exception:
            pass

    rate = correct / n * 100
    print(f"[toy-math] Baseline accuracy: {correct}/{n} ({rate:.1f}%)")
    return rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Toy RLVR math experiment")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--baseline", action="store_true", help="Eval base model only")
    parser.add_argument("--examples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    if args.baseline:
        eval_baseline()
    elif args.eval_only:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ckpt = _OUT_DIR / "checkpoints" / "final"
        if not ckpt.exists():
            print(f"[toy-math] No checkpoint at {ckpt}")
            sys.exit(1)
        print(f"[toy-math] Loading fine-tuned model from {ckpt}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(model, str(ckpt))
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        eval_model_hf(model, tokenizer)
    else:
        # Run baseline first
        eval_baseline(n=50)

        # Train
        model, tokenizer = train(
            n_examples=args.examples, epochs=args.epochs, lr=args.lr,
        )

        # Eval fine-tuned
        eval_model_hf(model, tokenizer)
