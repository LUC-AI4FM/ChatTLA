#!/usr/bin/env python3
"""GRPO training on TLA+ Next-action generation with the per-action TLC reward.

This is the direct path to the user's stated goal — high TLC/SANY/TLAPS pass
rates. We skip the GSM8K → LoRA → TLA+ canary chain and train the small base
model directly on the TLA+ task using:

  - Dataset: 73 Diamond-curated specs, each carved into a (prefix, gold Next,
    suffix) harness via src/validators/per_action_tlc.py.
  - Reward: per-action TLC. The reward function splices the model's `Next`
    output back into the harness, runs SANY+TLC, and grades on tier:
       1.0  TLC accepts  | 0.5  silver  | 0.2  bronze  | 0.05 normalizable
  - Trainer: TRL GRPOTrainer with vLLM colocate rollouts.
  - Base: Qwen2.5-1.5B-Instruct (same as the GSM8K canary, so we can compare
    learning dynamics directly).

Run:
    python -m scripts.train_canary_tla --smoke
    python -m scripts.train_canary_tla --max-steps 200

Notes:
  - Per-step time is dominated by TLC reward calls. With 16 completions/step
    and 4 reward workers, expect 60-90s/step (most TLC calls are <2s on the
    bounded harness state space; the cap is 20s).
  - 73 unique prompts means we cycle the dataset every 73 / 2 ≈ 37 steps.
    For a 200-step run that's ~5 epochs, plenty for the model to internalize
    the action grammar without overfitting (each prompt generates 8 distinct
    completions per visit).
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
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--corpus", default="data/processed/diamond_curated.jsonl")
    parser.add_argument("--output-dir", default="outputs/checkpoints_canary_tla")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--logging-steps", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--smoke", action="store_true",
                        help="2 steps, no checkpoint, sanity-check the wiring.")
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument("--vllm-gpu-memory", type=float, default=0.35)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from src.rlvr_canary.tla_dataset import load_tla_action_prompts
    from src.rlvr_canary.tla_reward import per_action_tlc_reward

    if args.smoke:
        args.max_steps = 2
        args.save_steps = 1000
        args.logging_steps = 1

    print(f"[tla-canary] loading TLA+ harness corpus from {args.corpus} ...")
    examples = load_tla_action_prompts(args.corpus)
    if not examples:
        sys.exit(f"No usable harnesses in {args.corpus}.")
    print(f"[tla-canary] loaded {len(examples)} harnesses")

    train_ds = Dataset.from_list([
        {
            "prompt": ex.prompt,
            # Harness fields as separate string columns — Arrow doesn't
            # serialize dataclass objects reliably.
            "harness_prefix": ex.harness.prefix,
            "harness_suffix": ex.harness.suffix,
            "harness_module": ex.harness.module_name,
        }
        for ex in examples
    ])

    print(f"[tla-canary] loading model {args.model} ...")
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

    gen_batch_size = args.per_device_batch_size * args.num_generations
    vllm_kwargs: dict = {}
    if not args.no_vllm:
        vllm_kwargs = dict(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=args.vllm_gpu_memory,
        )

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
        gradient_checkpointing=True,
        report_to=["none"],
        remove_unused_columns=False,
        seed=20260407,
        **vllm_kwargs,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=train_ds,
        reward_funcs=[per_action_tlc_reward],
    )

    print(f"[tla-canary] starting GRPO training: {args.max_steps} steps, "
          f"{args.num_generations} gens/prompt, "
          f"effective rollouts/step = {gen_batch_size}")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[tla-canary] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
