#!/usr/bin/env python3
"""GRPO (RL) training on gpt-oss-20b with LoRA + per-action TLC reward.

Builds on the validated 1.5B canary (project_rlvr_canary_validated_20260409.md)
which showed reward lift 0.296 → 0.533 (peak 0.853) in 100 steps of GRPO on
Qwen2.5-1.5B. This script scales that to the production 20B model with LoRA.

Architecture:
  - Base model: merged_model_v13 (gpt-oss-20b with v13 SFT LoRA merged in)
  - LoRA adapter: fresh r=8/alpha=16 from lora_config.yaml
  - Reward: per-action TLC (same as canary — 1.0/0.5/0.2/0.05 tiers)
  - Dataset: 73 Diamond-curated harnesses
  - Trainer: TRL GRPOTrainer with peft_config (no vLLM, HF generate)

Memory budget (2x RTX 8000, 49 GB each):
  - 20B bf16 ≈ 40 GB via device_map="auto" (split ~20/20)
  - LoRA params: ~50 MB
  - GRPO generation buffer (4 completions × 512 tokens): ~2 GB
  - Gradient checkpointing activations: ~5 GB
  - Optimizer states (Adam on LoRA only): ~100 MB
  - Headroom: ~50 GB total used across 98 GB available

Run:
    python -m scripts.train_rl_20b --smoke          # 2-step sanity check
    python -m scripts.train_rl_20b --max-steps 50   # first real run (~3-4h)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/merged_model_v13",
                        help="Base model path or HF ID")
    parser.add_argument("--corpus", default="data/processed/diamond_curated.jsonl")
    parser.add_argument("--output-dir", default="outputs/checkpoints_rl_20b")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                        help="5e-6 matches the canary. Prior run at 5e-7 produced "
                        "negligible updates on 4M LoRA params.")
    parser.add_argument("--num-generations", type=int, default=8,
                        help="8 rollouts per prompt. 16 OOMs on backward pass (48/49 GB). "
                        "Rely on finer reward shaping for variance instead.")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=512,
                        help="512 with eos_token_id stop tokens [Spec/====]. Model should "
                        "terminate naturally at 100-300 tokens; 512 is the safety cap.")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty against reference model")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature. Kept at 1.0 — with eos_token_id stop "
                        "tokens (Spec/====), diversity comes from natural variation in "
                        "completion length and quality, not from temperature noise.")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--smoke", action="store_true",
                        help="2 steps, no checkpoint, sanity-check the wiring.")
    args = parser.parse_args()

    import torch
    import yaml
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from src.rlvr_canary.tla_dataset import load_tla_action_prompts
    from src.rlvr_canary.tla_reward import per_action_tlc_reward

    if args.smoke:
        args.max_steps = 2
        args.save_steps = 1000
        args.logging_steps = 1

    # ── Dataset ─────────────────────────────────────────────────────────
    print(f"[rl-20b] loading TLA+ harness corpus from {args.corpus} ...")
    examples = load_tla_action_prompts(args.corpus)
    if not examples:
        sys.exit(f"No usable harnesses in {args.corpus}.")
    print(f"[rl-20b] loaded {len(examples)} harnesses")

    train_ds = Dataset.from_list([
        {
            "prompt": ex.prompt,
            "harness_prefix": ex.harness.prefix,
            "harness_suffix": ex.harness.suffix,
            "harness_module": ex.harness.module_name,
        }
        for ex in examples
    ])

    # ── LoRA config ─────────────────────────────────────────────────────
    lora_cfg_path = _REPO_ROOT / "src" / "training" / "lora_config.yaml"
    with open(lora_cfg_path) as f:
        lora_cfg = yaml.safe_load(f)
    print(f"[rl-20b] LoRA config: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}, "
          f"target={lora_cfg['target_modules']}")

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )

    # ── Model ───────────────────────────────────────────────────────────
    print(f"[rl-20b] loading base model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
        seed=20260409,
        # Higher temperature for exploration diversity — the SFT-trained 20B
        # model produces very uniform outputs at temperature=1.0, which gives
        # GRPO zero advantage signal (all completions get the same reward tier).
        # 1.5 adds enough stochasticity for reward variance without going
        # off-distribution. If reward_std stays at 0, raise further.
        temperature=args.temperature,
        # Gradient checkpointing to reduce activation memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # No vLLM — use HF generate for stability on 2-GPU pipeline-parallel
        # vLLM colocate crashed post-FSDP (project_rlvr_canary_validated_20260409)
        #
        # Stop strings: the 20B model was SFT-trained to write full TLA+ specs
        # and never emits EOS. Without stop strings, 100% of completions hit
        # max_completion_length (256), giving GRPO uniform truncated outputs
        # with no reward variance → zero gradient signal.
        # "Spec ==" and "====" are the patterns that immediately follow the
        # Next operator in every TLA+ spec. "\n\\*" catches comment lines that
        # sometimes appear between Next and Spec.
        # Stop tokens: the model never emits EOS after writing Next — it keeps
        # going to write Spec, invariants, etc. Add "Spec" (9051) and "====" (904)
        # as additional EOS tokens so generation terminates at the natural boundary.
        # [200002=original EOS, 9051="Spec", 904="===="]
        generation_kwargs={
            "eos_token_id": [tokenizer.eos_token_id, 9051, 904],
        },
    )

    # ── Trainer ─────────────────────────────────────────────────────────
    # Pass peft_config so GRPOTrainer wraps the model with LoRA internally.
    # This also sets up the reference model as the base (adapter disabled).
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=train_ds,
        reward_funcs=[per_action_tlc_reward],
        peft_config=peft_config,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[rl-20b] params: {total_params/1e9:.1f}B total, {trainable/1e6:.1f}M trainable "
          f"({100*trainable/total_params:.2f}%)")
    print(f"[rl-20b] starting GRPO: {args.max_steps} steps, "
          f"{args.num_generations} gens/prompt, "
          f"lr={args.learning_rate}, beta={args.beta}")

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[rl-20b] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
