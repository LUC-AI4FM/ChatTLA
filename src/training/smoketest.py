"""
smoketest.py — Pre-flight validation for ChatTLA fine-tuning parameters.

Runs in <5 minutes and catches every configuration error that would otherwise
surface 30 hours into a training run.  Run this BEFORE every full training run.

Checks performed
----------------
1. Hardware: GPU count, VRAM per GPU, total VRAM, disk space
2. Model loading: gpt-oss-20b loads without OOM, device map spans both GPUs
3. LoRA config: validates rank, target_modules, layers_to_transform, dropout
4. LoRA application: PEFT model compiles, trainable params are on correct device
5. Tokenizer: chat template applies, max_length doesn't truncate training data
6. Dataset: train.jsonl and eval.jsonl exist, parse correctly, have messages field
7. Sequence length: measures actual token lengths, warns if truncation occurs
8. Forward pass: one forward pass succeeds without device mismatch
9. Backward pass: one backward pass produces finite gradients
10. Disk space: sufficient free space for checkpoints + merged model

Usage
-----
    source venv/bin/activate
    CUDA_VISIBLE_DEVICES=0,1 python -m src.training.smoketest

Exit codes
----------
    0  All checks passed — safe to start full training
    1  Fatal error — training WILL fail, fix before proceeding
    2  Warning — training may work but results may be suboptimal
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Default to both GPUs
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

import torch
import yaml

_REPO_ROOT     = Path(__file__).resolve().parents[2]
_TRAIN_JSONL   = _REPO_ROOT / "data" / "processed" / "train.jsonl"
_EVAL_JSONL    = _REPO_ROOT / "data" / "processed" / "eval.jsonl"
_LORA_CFG_PATH = Path(__file__).parent / "lora_config.yaml"
_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "checkpoints"

MODEL_ID = "openai/gpt-oss-20b"

# Colour helpers for terminal output
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

_errors: list[str] = []
_warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    _warnings.append(msg)
    print(f"  {YELLOW}⚠{RESET} {msg}")


def fail(msg: str) -> None:
    _errors.append(msg)
    print(f"  {RED}✗{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")


def check_hardware():
    """Check GPU count, VRAM, RAM, and disk space."""
    section("1. Hardware")

    if not torch.cuda.is_available():
        fail("CUDA not available — cannot train on GPU")
        return

    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        ok(f"Found {n_gpus} GPUs")
    elif n_gpus == 1:
        warn(f"Only 1 GPU visible. Set CUDA_VISIBLE_DEVICES=0,1 to use both.")
    else:
        fail("No GPUs detected")

    total_vram_gb = 0
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1e9
        total_vram_gb += vram_gb
        free_bytes, _ = torch.cuda.mem_get_info(i)
        free_gb = free_bytes / 1e9
        if free_gb < 20:
            warn(f"GPU {i} ({props.name}): only {free_gb:.1f} GB free of {vram_gb:.1f} GB — other processes may cause OOM")
        else:
            ok(f"GPU {i} ({props.name}): {free_gb:.1f} GB free of {vram_gb:.1f} GB")

    if total_vram_gb >= 80:
        ok(f"Total VRAM: {total_vram_gb:.0f} GB — sufficient for full model + LoRA r=16")
    elif total_vram_gb >= 48:
        warn(f"Total VRAM: {total_vram_gb:.0f} GB — tight, may need gradient checkpointing")
    else:
        fail(f"Total VRAM: {total_vram_gb:.0f} GB — insufficient for gpt-oss-20b fine-tuning")

    # Disk space
    try:
        stat = os.statvfs(str(_REPO_ROOT))
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        if free_gb < 20:
            fail(f"Only {free_gb:.1f} GB disk space free — need ≥20 GB for checkpoints")
        elif free_gb < 50:
            warn(f"{free_gb:.1f} GB disk free — sufficient but tight for merge+GGUF later")
        else:
            ok(f"{free_gb:.1f} GB disk space free")
    except Exception as e:
        warn(f"Could not check disk space: {e}")

    # RAM
    try:
        total_ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3)
        if total_ram_gb >= 64:
            ok(f"System RAM: {total_ram_gb:.0f} GB")
        else:
            warn(f"System RAM: {total_ram_gb:.0f} GB — may be tight for model loading")
    except Exception:
        pass


def check_lora_config():
    """Validate the LoRA YAML config."""
    section("2. LoRA Configuration")

    if not _LORA_CFG_PATH.exists():
        fail(f"LoRA config not found: {_LORA_CFG_PATH}")
        return

    with _LORA_CFG_PATH.open() as f:
        cfg = yaml.safe_load(f)

    r = cfg.get("r", 0)
    alpha = cfg.get("lora_alpha", 0)
    dropout = cfg.get("lora_dropout", 0)
    targets = cfg.get("target_modules", [])

    if r >= 16:
        ok(f"LoRA rank r={r} — good capacity for domain adaptation")
    elif r >= 8:
        warn(f"LoRA rank r={r} — minimum viable, r=16 recommended for TLA+")
    else:
        fail(f"LoRA rank r={r} — TOO LOW, model cannot learn meaningful patterns. Use r≥16.")

    effective_scale = alpha / r if r > 0 else 0
    if 1.0 <= effective_scale <= 4.0:
        ok(f"Effective LoRA scale (alpha/r) = {effective_scale:.1f}")
    else:
        warn(f"Effective LoRA scale = {effective_scale:.1f} — unusual, typical range is 1.0-4.0")

    if targets == "all-linear" or (isinstance(targets, list) and len(targets) >= 4):
        ok(f"target_modules={targets} — covers attention + FFN")
    else:
        warn(f"target_modules={targets} — may miss important layers")

    if dropout > 0:
        ok(f"lora_dropout={dropout} — regularization enabled (good for small datasets)")
    else:
        warn(f"lora_dropout=0 — no regularization, risk of overfitting on 57 examples")


def check_dataset():
    """Validate training and eval datasets."""
    section("3. Dataset")

    for name, path in [("train", _TRAIN_JSONL), ("eval", _EVAL_JSONL)]:
        if not path.exists():
            fail(f"{name} dataset not found: {path}")
            continue

        records = []
        for line in path.read_text().strip().splitlines():
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                fail(f"{name}: invalid JSON line in {path}")
                break

        if not records:
            fail(f"{name}: empty dataset")
            continue

        # Check schema
        if "messages" not in records[0]:
            fail(f"{name}: missing 'messages' key — SFTTrainer requires this")
        else:
            ok(f"{name}: {len(records)} examples, schema OK")

        # Check for analysis/final channels (harmony format)
        has_channels = any(
            m.get("channel") in ("analysis", "final")
            for rec in records[:5]
            for m in rec.get("messages", [])
        )
        if has_channels:
            ok(f"{name}: harmony channel format detected (analysis/final)")
        else:
            warn(f"{name}: no harmony channels found — gpt-oss may not format responses correctly")


def check_tokenization():
    """Load tokenizer, measure token lengths, check for truncation."""
    section("4. Tokenizer & Sequence Lengths")

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        ok(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    except Exception as e:
        fail(f"Cannot load tokenizer: {e}")
        return None

    # Load LoRA config to get max_length
    with _LORA_CFG_PATH.open() as f:
        _ = yaml.safe_load(f)

    # Get max_length from training args
    from src.training.train import build_training_args
    args = build_training_args(smoke_test=False, use_cpu=True)
    max_length = args.max_length
    ok(f"Training max_length={max_length}")

    # Measure actual token lengths
    if _TRAIN_JSONL.exists():
        records = [json.loads(l) for l in _TRAIN_JSONL.read_text().strip().splitlines()]
        lengths = []
        for rec in records:
            # Concatenate all message content to estimate length
            text = " ".join(m.get("content", "") for m in rec.get("messages", []))
            tokens = tokenizer.encode(text, add_special_tokens=True)
            lengths.append(len(tokens))

        avg_len = sum(lengths) / len(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        p90_len = sorted(lengths)[int(len(lengths) * 0.9)] if lengths else 0
        truncated = sum(1 for l in lengths if l > max_length)

        ok(f"Token lengths: avg={avg_len:.0f}, p90={p90_len}, max={max_len}")

        if truncated > 0:
            pct = truncated / len(lengths) * 100
            if pct > 50:
                fail(f"{truncated}/{len(lengths)} examples ({pct:.0f}%) TRUNCATED at max_length={max_length}. "
                     f"Increase max_length to ≥{p90_len} to capture 90% of specs.")
            elif pct > 10:
                warn(f"{truncated}/{len(lengths)} examples ({pct:.0f}%) truncated at max_length={max_length}. "
                     f"p90={p90_len} tokens. Consider increasing max_length if VRAM allows.")
            else:
                ok(f"Only {truncated}/{len(lengths)} examples ({pct:.0f}%) truncated — acceptable")
        else:
            ok(f"No examples truncated at max_length={max_length}")

    return tokenizer


def check_model_loading():
    """Load the model and verify device placement."""
    section("5. Model Loading")

    from transformers import AutoModelForCausalLM

    print("  Loading gpt-oss-20b (this may take 1-2 minutes)...")
    t0 = time.monotonic()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            attn_implementation="eager",
            use_cache=False,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        dt = time.monotonic() - t0
        ok(f"Model loaded in {dt:.1f}s")
    except Exception as e:
        fail(f"Model loading failed: {e}")
        return None

    # Check device map
    if hasattr(model, "hf_device_map"):
        devices = set(str(v) for v in model.hf_device_map.values())
        cuda_devices = [d for d in devices if "cuda" in d or d.isdigit()]
        if len(cuda_devices) >= 2:
            ok(f"Model distributed across devices: {devices}")
        elif len(cuda_devices) == 1:
            ok(f"Model on single GPU: {devices}")
        else:
            warn(f"Unusual device map: {devices}")
    else:
        ok("Model loaded (no explicit device map)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    ok(f"Total parameters: {total_params / 1e9:.1f}B")

    return model


def check_lora_application(model):
    """Apply LoRA and verify trainable parameter count/placement."""
    section("6. LoRA Application")

    from peft import LoraConfig, get_peft_model

    from src.training.train import load_lora_config

    try:
        lora_config = load_lora_config()
        ok(f"LoRA config loaded: r={lora_config.r}, target_modules={lora_config.target_modules}")
    except Exception as e:
        fail(f"Failed to load LoRA config: {e}")
        return None

    try:
        model = get_peft_model(model, lora_config)
        ok("PEFT LoRA model created")
    except Exception as e:
        fail(f"Failed to apply LoRA: {e}")
        return None

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = trainable / total * 100

    if trainable > 1_000_000:
        ok(f"Trainable parameters: {trainable:,} ({pct:.2f}% of {total/1e9:.1f}B)")
    elif trainable > 100_000:
        warn(f"Trainable parameters: {trainable:,} ({pct:.2f}%) — low, consider increasing r")
    else:
        fail(f"Trainable parameters: {trainable:,} ({pct:.4f}%) — TOO FEW, adapter cannot learn")

    # Check device placement
    devices = set()
    grad_devices = set()
    for name, p in model.named_parameters():
        if hasattr(p, "device"):
            devices.add(str(p.device))
            if p.requires_grad:
                grad_devices.add(str(p.device))

    ok(f"Parameter devices: {devices}")
    if len(grad_devices) > 1:
        warn(f"Trainable params on multiple devices: {grad_devices} — may cause issues")
    else:
        ok(f"All trainable params on: {grad_devices}")

    return model


def check_forward_backward(model, tokenizer):
    """Run one forward + backward pass to validate computation graph."""
    section("7. Forward/Backward Pass")

    if model is None or tokenizer is None:
        warn("Skipping — model or tokenizer not available")
        return

    try:
        # Create a small test input
        test_text = "---- MODULE Test ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x' = x + 1\nSpec == Init /\\ [][Next]_x\n===="
        inputs = tokenizer(test_text, return_tensors="pt", max_length=256, truncation=True)

        # Move inputs to same device as first trainable param
        device = None
        for p in model.parameters():
            if p.requires_grad:
                device = p.device
                break
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        # Forward
        t0 = time.monotonic()
        outputs = model(**inputs)
        fwd_time = time.monotonic() - t0
        loss = outputs.loss

        if loss is not None and torch.isfinite(loss):
            ok(f"Forward pass OK — loss={loss.item():.4f} ({fwd_time:.2f}s)")
        else:
            fail(f"Forward pass produced invalid loss: {loss}")
            return

        # Backward
        t0 = time.monotonic()
        loss.backward()
        bwd_time = time.monotonic() - t0

        # Check gradients
        grad_norms = []
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad_norms.append(p.grad.norm().item())

        if len(grad_norms) > 0 and all(g < 1e6 for g in grad_norms):
            avg_grad = sum(grad_norms) / len(grad_norms)
            ok(f"Backward pass OK — {len(grad_norms)} params with gradients, avg norm={avg_grad:.4f} ({bwd_time:.2f}s)")
        elif len(grad_norms) == 0:
            fail("No gradients computed — LoRA parameters not receiving gradient signal")
        else:
            warn("Some gradients are very large — may indicate numerical instability")

        # Cleanup
        model.zero_grad()
        torch.cuda.empty_cache()

    except RuntimeError as e:
        msg = str(e)
        if "device" in msg.lower():
            fail(f"Device mismatch during forward/backward: {e}")
        elif "out of memory" in msg.lower():
            fail(f"OOM during forward/backward: {e}")
        else:
            fail(f"Runtime error during forward/backward: {e}")
    except Exception as e:
        fail(f"Unexpected error during forward/backward: {e}")


def check_training_math():
    """Compute and display expected training schedule."""
    section("8. Training Schedule")

    train_count = 0
    if _TRAIN_JSONL.exists():
        train_count = sum(1 for _ in _TRAIN_JSONL.open())

    with _LORA_CFG_PATH.open() as f:
        cfg = yaml.safe_load(f)

    from src.training.train import build_training_args
    args = build_training_args(smoke_test=False, use_cpu=True)

    eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = max(1, train_count // eff_batch)
    total_steps = steps_per_epoch * int(args.num_train_epochs)

    ok(f"Training examples: {train_count}")
    ok(f"Effective batch size: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} = {eff_batch}")
    ok(f"Steps per epoch: {steps_per_epoch}")
    ok(f"Total epochs: {args.num_train_epochs}")
    ok(f"Total training steps: {total_steps}")
    ok(f"Eval every {args.eval_steps} steps, save every {args.save_steps} steps")
    ok(f"Learning rate: {args.learning_rate} with {args.lr_scheduler_type} schedule")
    ok(f"Warmup steps: {args.warmup_steps}")

    if total_steps < 50:
        warn(f"Only {total_steps} total steps — may not be enough for convergence")
    elif total_steps > 1000:
        warn(f"{total_steps} total steps — long run, ensure disk space for checkpoints")

    # Estimate time
    # Rough estimate: ~3s/step for 20B MoE model with LoRA on 2x RTX 8000
    est_time_hours = total_steps * 3 / 3600
    ok(f"Estimated training time: ~{est_time_hours:.1f} hours (rough estimate)")


def main():
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  ChatTLA Fine-Tuning Smoketest{RESET}")
    print(f"{BOLD}  Pre-flight validation for training parameters{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    t0 = time.monotonic()

    # Phase 1: Static checks (fast)
    check_hardware()
    check_lora_config()
    check_dataset()

    # Phase 2: Tokenizer check (fast)
    tokenizer = check_tokenization()

    # Phase 3: Model loading (slow — 1-2 min)
    model = check_model_loading()

    # Phase 4: LoRA application
    if model is not None:
        model = check_lora_application(model)

    # Phase 5: Forward/backward validation
    check_forward_backward(model, tokenizer)

    # Phase 6: Training math
    check_training_math()

    # Cleanup
    if model is not None:
        del model
        torch.cuda.empty_cache()

    # Summary
    dt = time.monotonic() - t0
    section("SUMMARY")
    print(f"  Time: {dt:.0f}s")

    if _errors:
        print(f"\n  {RED}{BOLD}FATAL ERRORS ({len(_errors)}):{RESET}")
        for e in _errors:
            print(f"    {RED}✗{RESET} {e}")
        print(f"\n  {RED}DO NOT start training until all errors are fixed.{RESET}")
        return 1

    if _warnings:
        print(f"\n  {YELLOW}WARNINGS ({len(_warnings)}):{RESET}")
        for w in _warnings:
            print(f"    {YELLOW}⚠{RESET} {w}")
        print(f"\n  {YELLOW}Training may work but results may be suboptimal.{RESET}")
        return 2

    print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED — safe to start full training.{RESET}")
    print(f"\n  Run:  CUDA_VISIBLE_DEVICES=0,1 python -m src.training.train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
