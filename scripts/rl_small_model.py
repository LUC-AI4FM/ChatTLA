#!/usr/bin/env python3
"""
rl_small_model.py — Fast RL loop with smaller models for rapid iteration.

Purpose: Test whether the RL approach works at all, before committing GPU-hours to 20B.
Uses: Llama-3.2-3B or similar small instruct model for fast cycles (~30 min/cycle).

Key insight: If a 3B model can't learn TLA+ structure via DPO, the method is broken.
If it can, we have evidence the approach works and can scale to 20B.

Usage:
    python scripts/rl_small_model.py --smoke           # 1 cycle test
    python scripts/rl_small_model.py --cycles 20      # 20 cycle experiment
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Small model — fits on single GPU, trains in minutes
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
# Alternative: "microsoft/Phi-3-mini-4k-instruct"

_DATA_DIR = _REPO_ROOT / "data" / "processed" / "rl_small"
_LOG_DIR = _REPO_ROOT / "outputs" / "logs" / "rl_small"
_CKPT_DIR = _REPO_ROOT / "outputs" / "checkpoints_rl_small"

for d in [_DATA_DIR, _LOG_DIR, _CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class SmallConfig:
    model_id: str = MODEL_ID
    seed: int = 42
    
    # Generation
    prompts_per_cycle: int = 10
    attempts: int = 2
    temperature: float = 0.5
    max_tokens: int = 1024
    
    # Training
    train_threshold: int = 3  # Train after 3 new pairs
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 2
    max_length: int = 1024
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32


def load_prompts(n: int, seed: int) -> list[dict]:
    """Load TLA+ prompts from benchmark suite."""
    bench_file = _REPO_ROOT / "data" / "benchmarks" / "benchmark_suite.json"
    prompts = []
    
    if bench_file.exists():
        with open(bench_file) as f:
            data = json.load(f)
        for i, p in enumerate(data.get("benchmarks", [])):
            prompts.append({
                "id": p.get("benchmark_id", f"b{i}"),
                "text": p.get("nl_description", p.get("description", "")),
                "module": p.get("module_name", "Spec"),
            })
    
    # Add some simple synthetic prompts for diversity
    simple_prompts = [
        {"id": "simple_counter", "text": "A counter that starts at 0 and increments by 1, with an invariant that it's always >= 0", "module": "Counter"},
        {"id": "simple_toggle", "text": "A toggle switch that can be On or Off, starting Off", "module": "Toggle"},
        {"id": "simple_buffer", "text": "A bounded buffer of size 3 with put and get operations", "module": "Buffer"},
        {"id": "simple_mutex", "text": "A simple mutex with acquire and release operations", "module": "Mutex"},
        {"id": "simple_semaphore", "text": "A counting semaphore initialized to N", "module": "Semaphore"},
    ]
    prompts.extend(simple_prompts)
    
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:n]


def generate_spec(model, tokenizer, prompt_text: str, cfg: SmallConfig) -> str:
    """Generate TLA+ spec using the model."""
    system = (
        "You are an expert at TLA+ formal specifications. "
        "Write a complete, valid TLA+ spec. Output ONLY the code.\n\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Write a TLA+ specification for: {prompt_text}"},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def validate_spec(spec: str, module: str) -> dict:
    """Validate with SANY + TLC."""
    from src.validators.sany_validator import validate_string as sany_validate
    from src.validators.tlc_validator import validate_string as tlc_validate
    
    result = {"sany_ok": False, "tlc_ok": False, "tier": "bronze", "reward": 0.0}
    
    try:
        sany_r = sany_validate(spec, module_name=module)
        result["sany_ok"] = sany_r.valid
        if not sany_r.valid:
            return result
    except Exception:
        return result
    
    try:
        tlc_r = tlc_validate(spec, module_name=module, timeout=20)
        result["tlc_ok"] = tlc_r.valid
        if tlc_r.valid:
            result["tier"] = "gold"
            result["reward"] = 1.0
        else:
            result["tier"] = "silver"
            result["reward"] = 0.3
    except Exception:
        result["tier"] = "silver"
        result["reward"] = 0.3
    
    return result


def run_cycle(model, tokenizer, cfg: SmallConfig, cycle: int) -> dict:
    """Run one generation-validation-training cycle."""
    seed = cfg.seed + cycle
    prompts = load_prompts(cfg.prompts_per_cycle, seed)
    
    print(f"\n=== Cycle {cycle} (seed={seed}) ===")
    print(f"Generating specs for {len(prompts)} prompts...")
    
    results = []
    for p in prompts:
        for _ in range(cfg.attempts):
            spec = generate_spec(model, tokenizer, p["text"], cfg)
            if spec:
                validation = validate_spec(spec, p["module"])
                results.append({
                    "prompt_id": p["id"],
                    "prompt_text": p["text"],
                    "spec": spec,
                    **validation,
                })
    
    gold = [r for r in results if r["tier"] == "gold"]
    silver = [r for r in results if r["tier"] == "silver"]
    bronze = [r for r in results if r["tier"] == "bronze"]
    
    print(f"Results: {len(gold)} gold, {len(silver)} silver, {len(bronze)} bronze")
    
    # Create DPO pairs
    dpo_pairs = []
    by_prompt = {}
    for r in results:
        by_prompt.setdefault(r["prompt_id"], []).append(r)
    
    for pid, candidates in by_prompt.items():
        candidates.sort(key=lambda x: x["reward"], reverse=True)
        best, worst = candidates[0], candidates[-1]
        if best["reward"] > worst["reward"]:
            dpo_pairs.append({
                "prompt": best["prompt_text"],
                "chosen": best["spec"],
                "rejected": worst["spec"],
            })
    
    print(f"Created {len(dpo_pairs)} DPO pairs")
    
    # Train if threshold met
    trained = False
    loss = 0.0
    if len(dpo_pairs) >= cfg.train_threshold:
        print(f"Training on {len(dpo_pairs)} pairs...")
        try:
            ds = Dataset.from_list(dpo_pairs)
            
            dpo_config = DPOConfig(
                output_dir=str(_CKPT_DIR / f"cycle_{cycle}"),
                per_device_train_batch_size=cfg.batch_size,
                gradient_accumulation_steps=cfg.grad_accum,
                learning_rate=cfg.lr,
                num_train_epochs=cfg.epochs,
                beta=0.1,
                max_length=cfg.max_length,
                bf16=True,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
            )
            
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=dpo_config,
                train_dataset=ds,
                processing_class=tokenizer,
            )
            
            result = trainer.train()
            loss = result.training_loss
            trained = True
            print(f"Training complete, loss={loss:.4f}")
        except Exception as e:
            print(f"Training failed: {e}")
    
    return {
        "cycle": cycle,
        "gold": len(gold),
        "silver": len(silver),
        "bronze": len(bronze),
        "dpo_pairs": len(dpo_pairs),
        "trained": trained,
        "loss": loss,
        "sany_rate": (len(gold) + len(silver)) / max(1, len(results)),
        "tlc_rate": len(gold) / max(1, len(results)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()
    
    cfg = SmallConfig(model_id=args.model)
    if args.smoke:
        cfg.prompts_per_cycle = 3
        cfg.attempts = 1
        cfg.train_threshold = 1
    
    print(f"Loading model: {cfg.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add LoRA
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Run cycles
    history = []
    cycles = 1 if args.smoke else args.cycles
    
    for i in range(cycles):
        metrics = run_cycle(model, tokenizer, cfg, i + 1)
        history.append(metrics)
        
        # Log
        with open(_LOG_DIR / "history.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
    
    # Summary
    print("\n=== Experiment Summary ===")
    total_gold = sum(m["gold"] for m in history)
    total_trains = sum(1 for m in history if m["trained"])
    avg_tlc = sum(m["tlc_rate"] for m in history) / len(history)
    
    print(f"Cycles: {len(history)}")
    print(f"Total gold specs: {total_gold}")
    print(f"Total trains: {total_trains}")
    print(f"Average TLC rate: {avg_tlc:.1%}")
    
    # Save final checkpoint
    model.save_pretrained(_CKPT_DIR / "final")
    tokenizer.save_pretrained(_CKPT_DIR / "final")
    print(f"Saved to {_CKPT_DIR / 'final'}")


if __name__ == "__main__":
    main()
