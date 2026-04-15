#!/usr/bin/env python3
"""GRPO training on gpt-oss-20b with repair prompts + improvement reward.

Instead of training the model to one-shot perfect specs (which gives
zero reward variance), this trains the model to REPAIR broken specs
given line-annotated verifier diagnostics.

Why this works:
  1. Dense reward — improvement delta gives variance even with 4 gens
  2. Easier task — fixing errors is simpler than generating from scratch
  3. Unlimited data — Ralph trajectories provide many repair pairs
  4. Curriculum — easy (SANY fixes) → medium → hard (TLC/semantic)

Architecture:
  - Base model: post-DPO checkpoint (same as train_rl_fullspec.py)
  - LoRA adapter: fresh r=8/alpha=16
  - Reward: repair_reward (improvement-based shaped reward)
  - Dataset: Ralph repair pairs from collect_ralph_trajectories.py
  - Trainer: TRL GRPOTrainer

Memory budget (2x RTX 8000, 49 GB each):
  - 20B bf16 ≈ 40 GB via device_map="auto"
  - 4 completions × 1024 tokens: similar to fullspec
  - Repair prompts longer (~1200 tokens) but completions shorter (1024)
  - Total: ~54-58 GB / 98 GB available

Run:
    python -m scripts.train_rl_repair --smoke           # 2-step sanity check
    python -m scripts.train_rl_repair --max-steps 300   # full run (~16h)
    python -m scripts.train_rl_repair --difficulty easy  # SANY-only curriculum
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
    """Pick the best available MERGED base model in priority order."""
    candidates = [
        _REPO_ROOT / "outputs" / "merged_model_dpo_piecewise",
        _REPO_ROOT / "outputs" / "merged_model_v14",
        _REPO_ROOT / "outputs" / "merged_model_v13",
        _REPO_ROOT / "outputs" / "merged_model",
    ]
    for c in candidates:
        if c.is_dir() and (c / "config.json").is_file():
            return str(c)
    return "openai/gpt-oss-20b"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=None,
                        help="Base model path (auto-detected if omitted)")
    parser.add_argument("--output-dir", default="outputs/checkpoints_rl_repair")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-6,
                        help="Slightly lower than fullspec for stability")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="4 gens with repair prompts fits when Ollama is unloaded. "
                        "Use --num-generations 2 if sharing GPU with Ollama.")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=1024,
                        help="Repairs are shorter than from-scratch specs (1024 vs 1536). "
                        "Use 768 if sharing GPU with Ollama.")
    parser.add_argument("--beta", type=float, default=0.02,
                        help="Lower KL penalty — allow more deviation for repair task")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Lower than fullspec — repairs need focus")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--difficulty", default="all",
                        choices=["easy", "medium", "hard", "all"],
                        help="Filter repair pairs by difficulty")
    parser.add_argument("--min-before-score", type=float, default=0.02,
                        help="Drop pairs with before_score below this — "
                        "unparseable specs leave the model with no improvement signal")
    parser.add_argument("--max-before-score", type=float, default=0.80,
                        help="Drop already-good pairs that leave no headroom")
    parser.add_argument("--max-prompt-tokens", type=int, default=1600,
                        help="Hard prompt-length filter — drops the long-tail "
                        "that blows up eager-attention memory (this version of "
                        "TRL has no GRPOConfig.max_prompt_length, so the only "
                        "defense is dataset-level filtering)")
    parser.add_argument("--trajectory-file",
                        default="data/processed/ralph_repair_pairs.jsonl",
                        help="Path to repair pairs JSONL")
    parser.add_argument("--smoke", action="store_true",
                        help="2 steps, no checkpoint, sanity-check the wiring")
    args = parser.parse_args()

    if args.model is None:
        args.model = _resolve_base_model()

    import torch
    import yaml
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from src.rlvr_canary.repair_dataset import (
        format_repair_prompt,
        load_repair_prompts,
    )
    from src.rlvr_canary.repair_reward import register_before_scores, repair_reward

    if args.smoke:
        args.max_steps = 2
        args.save_steps = 1000
        args.logging_steps = 1

    # -- Tokenizer (needed for length-filter before dataset load) -----
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # -- Dataset --------------------------------------------------------
    print(f"[rl-repair] loading repair prompts from {args.trajectory_file} ...")
    examples, before_scores = load_repair_prompts(
        trajectory_file=args.trajectory_file,
        difficulty=args.difficulty,
        max_examples=10 if args.smoke else None,
        min_before_score=args.min_before_score,
        max_before_score=args.max_before_score,
        max_prompt_tokens=args.max_prompt_tokens,
        tokenizer=tokenizer,
    )
    if not examples:
        sys.exit("No repair pairs found. Check filters or run collect_ralph_trajectories.py.")
    print(f"[rl-repair] loaded {len(examples)} repair pairs "
          f"(difficulty={args.difficulty}, "
          f"score=[{args.min_before_score},{args.max_before_score}], "
          f"max_toks={args.max_prompt_tokens})")

    # Register before_scores with the reward function
    register_before_scores(before_scores)

    # -- LoRA config ----------------------------------------------------
    lora_cfg_path = _REPO_ROOT / "src" / "training" / "lora_config.yaml"
    with open(lora_cfg_path) as f:
        lora_cfg = yaml.safe_load(f)
    print(f"[rl-repair] LoRA: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )

    # -- Model ----------------------------------------------------------
    print(f"[rl-repair] loading base model {args.model} ...")

    # Pre-format repair prompts (same pattern as train_rl_fullspec.py)
    formatted_prompts: list[str] = []
    for ex in examples:
        formatted_prompts.append(format_repair_prompt(ex, tokenizer))

    # Dataset is pre-sorted by difficulty (easy first) for curriculum.
    # Disable shuffling in GRPOConfig to preserve ordering.
    train_ds = Dataset.from_list([
        {"prompt": p} for p in formatted_prompts
    ])
    print(f"[rl-repair] prompts pre-formatted: "
          f"avg {sum(len(p) for p in formatted_prompts) // len(formatted_prompts)} chars")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    # -- GRPO config ----------------------------------------------------
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
        generation_kwargs={
            "eos_token_id": [tokenizer.eos_token_id, 904],  # 904 = "===="
        },
    )

    # -- Trainer --------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=train_ds,
        reward_funcs=[repair_reward],
        peft_config=peft_config,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[rl-repair] params: {total_params / 1e9:.1f}B total, "
          f"{trainable / 1e6:.1f}M trainable ({100 * trainable / total_params:.2f}%)")
    print(f"[rl-repair] starting GRPO: {args.max_steps} steps, "
          f"{args.num_generations} gens/prompt, lr={args.learning_rate}, "
          f"beta={args.beta}, temp={args.temperature}")

    trainer.train()

    # Save final checkpoint
    trainer.save_model(args.output_dir + "/final")
    print(f"[rl-repair] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
