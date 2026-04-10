#!/usr/bin/env python3
"""GRPO training on gpt-oss-20b with full-spec generation + component reward.

Replaces the per-action approach (train_rl_20b.py) that produced flat reward
because:
  1. Task too narrow — only Next operator fragments
  2. Zero reward variance — all 8 completions got same tier
  3. Token length mismatch — 512 cap truncated full-spec attempts
  4. Tiny prompt pool — only 73 harnesses

This script fixes all four:
  1. Full-spec generation from NL descriptions (the deployment task)
  2. Component-weighted partial credit (~10 distinct reward levels)
  3. max_completion_length=1536 (full specs average ~1200 tokens)
  4. 300+ prompts from 200-topic curriculum + diamond SFT descriptions

Architecture:
  - Base model: post-DPO checkpoint (outputs/checkpoints_dpo_piecewise)
    or merged_model_v14 if DPO hasn't run yet
  - LoRA adapter: fresh r=8/alpha=16
  - Reward: component_validator partial_credit [0, 1]
  - Dataset: 300+ NL descriptions from topics + diamond SFT
  - Trainer: TRL GRPOTrainer (no vLLM, HF generate)

Memory budget (2x RTX 8000, 49 GB each):
  - 20B bf16 ≈ 40 GB via device_map="auto"
  - 4 completions × 1536 tokens generation buffer: ~6 GB
  - Gradient checkpointing activations: ~8 GB
  - LoRA optimizer: ~100 MB
  - Total: ~54 GB / 98 GB available

Run:
    python -m scripts.train_rl_fullspec --smoke           # 2-step sanity check
    python -m scripts.train_rl_fullspec --max-steps 200   # full run (~12-16h)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _resolve_base_model() -> str:
    """Pick the best available MERGED base model in priority order.

    Must be a full model, not a LoRA adapter checkpoint — GRPOTrainer
    calls from_pretrained() and applies a fresh LoRA via peft_config.
    Adapter-only dirs (checkpoints_dpo_*) must be merged first.
    """
    candidates = [
        _REPO_ROOT / "outputs" / "merged_model_dpo_piecewise",  # merged post-DPO
        _REPO_ROOT / "outputs" / "merged_model_v14",
        _REPO_ROOT / "outputs" / "merged_model_v13",
        _REPO_ROOT / "outputs" / "merged_model",
    ]
    for c in candidates:
        # Must be a real model dir, not just an adapter
        if c.is_dir() and (c / "config.json").is_file():
            return str(c)
    return "openai/gpt-oss-20b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None,
                        help="Base model path (auto-detected if omitted)")
    parser.add_argument("--output-dir", default="outputs/checkpoints_rl_fullspec")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="4 rollouts per prompt. With 1536-token completions, "
                        "4 fits in 98GB. Dense reward gives variance even with 4.")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=1536,
                        help="Full specs avg ~1200 tokens. 1536 gives headroom "
                        "without wasting memory. Falls back to 1024 if OOM.")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty against reference model")
    parser.add_argument("--temperature", type=float, default=1.2,
                        help="Higher than 1.0 to ensure reward variance across "
                        "4 completions. Dense reward distinguishes them.")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--smoke", action="store_true",
                        help="2 steps, no checkpoint, sanity-check the wiring.")
    args = parser.parse_args()

    if args.model is None:
        args.model = _resolve_base_model()

    import torch
    import yaml
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from src.rlvr_canary.fullspec_dataset import load_fullspec_prompts
    from src.rlvr_canary.fullspec_reward import fullspec_component_reward

    if args.smoke:
        args.max_steps = 2
        args.save_steps = 1000
        args.logging_steps = 1

    # ── Dataset ─────────────────────────────────────────────────────────
    print(f"[rl-fullspec] loading full-spec prompts ...")
    examples = load_fullspec_prompts(
        include_topics=True,
        include_diamond_sft=True,
        include_train=False,
        max_per_source=10 if args.smoke else None,
    )
    if not examples:
        sys.exit("No usable prompts found.")
    print(f"[rl-fullspec] loaded {len(examples)} prompts")
    # Dataset built after tokenizer loads (needs apply_chat_template)

    # ── LoRA config ─────────────────────────────────────────────────────
    lora_cfg_path = _REPO_ROOT / "src" / "training" / "lora_config.yaml"
    with open(lora_cfg_path) as f:
        lora_cfg = yaml.safe_load(f)
    print(f"[rl-fullspec] LoRA: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )

    # ── Model ───────────────────────────────────────────────────────────
    print(f"[rl-fullspec] loading base model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Pre-format prompts as strings ending with the final channel header.
    # GRPOTrainer with message-list prompts calls apply_chat_template with
    # add_generation_prompt=True, which ends at "<|start|>assistant" — the
    # model then writes "<|channel|>analysis" (CoT) first, burning all
    # tokens on reasoning without ever producing a spec.
    # By pre-formatting and appending the channel header, we force the
    # model to write directly into the final channel (the spec).
    _CHANNEL_SUFFIX = "<|channel|>final<|message|>"
    formatted_prompts: list[str] = []
    for ex in examples:
        text = tokenizer.apply_chat_template(
            ex.prompt, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text + _CHANNEL_SUFFIX)

    train_ds = Dataset.from_list([
        {"prompt": p} for p in formatted_prompts
    ])
    print(f"[rl-fullspec] prompts pre-formatted: {len(formatted_prompts[0])} chars, "
          f"suffix='{_CHANNEL_SUFFIX}'")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    # ── GRPO config ─────────────────────────────────────────────────────
    gen_batch_size = args.per_device_batch_size * args.num_generations

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=gen_batch_size,
        generation_batch_size=gen_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        report_to=["none"],
        remove_unused_columns=False,
        seed=20260410,
        temperature=args.temperature,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # The gpt-oss harmony model writes analysis channel (CoT) then final
        # channel (spec). Without stop tokens, it burns 1536 tokens on CoT and
        # never reaches ====. Add ==== (token 904) as a stop token so generation
        # terminates when the spec body ends. Also keep the real EOS (200002).
        generation_kwargs={
            "eos_token_id": [tokenizer.eos_token_id, 904],  # 904 = "===="
        },
    )

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=train_ds,
        reward_funcs=[fullspec_component_reward],
        peft_config=peft_config,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[rl-fullspec] params: {total_params/1e9:.1f}B total, "
          f"{trainable/1e6:.1f}M trainable ({100*trainable/total_params:.2f}%)")
    print(f"[rl-fullspec] starting GRPO: {args.max_steps} steps, "
          f"{args.num_generations} gens/prompt, lr={args.learning_rate}, "
          f"beta={args.beta}, max_completion={args.max_completion_length}")

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[rl-fullspec] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
