#!/usr/bin/env python3
"""RLVR canary: full-FT GRPO on Llama-3.2-1B-Instruct + GSM8K.

This is the Phase-1 canary from `project_rlvr_validation_plan.md`. The
purpose is **not** to win at GSM8K — it's to confirm that the RL training
stack on this machine can lift a known-easy task with a known-good base.
Once this trains and reaches a reasonable pass@1, we add LoRA (Phase 2)
and only then swap the task to TLA+ (Phase 3).

Why these defaults
------------------
  - Base: meta-llama/Llama-3.2-1B-Instruct. Tiny enough to full-FT on a
    single Quadro RTX 8000, well-supported by TRL, and the 1B size echoes
    the FormaLLM finding that small reasoning-aligned models beat 70B
    siblings on TLA+ — same size class we'd eventually target.
  - Trainer: TRL's GRPOTrainer. Same TRL the existing ChatTLA SFT pipeline
    uses, so we minimize stack-switching variables.
  - Generations per prompt: 8 (TRL default; standard for GSM8K-GRPO recipes).
  - Reward: src.rlvr_canary.reward.binary_correctness_reward.
  - max_completion_length: 384 — GSM8K answers + CoT comfortably fit.

Run:
    python -m scripts.train_canary_gsm8k                 # full default run
    python -m scripts.train_canary_gsm8k --smoke         # 4 steps, 16 prompts
    python -m scripts.train_canary_gsm8k --max-steps 200 --train-limit 1024
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# transformers auto-imports system TensorFlow if available, which breaks on
# numpy 2.x. Disable TF + JAX detection before any transformers import.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Default switched from Llama-3.2-1B-Instruct (gated on HF) to Qwen2.5-1.5B-Instruct
    # so the canary runs out-of-the-box with no HF token. Pass --model meta-llama/...
    # if you want Llama and have access.
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="outputs/checkpoints_canary_gsm8k")
    parser.add_argument("--train-limit", type=int, default=2048,
                        help="Cap on training prompts (the canary doesn't need all 7473).")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=8,
                        help="GRPO group size — number of completions sampled per prompt.")
    parser.add_argument("--per-device-batch-size", type=int, default=2,
                        help="Distinct prompts per device per step. Effective groups = this * num_generations.")
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty against the reference model.")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run: 16 prompts, 4 steps. Sanity-checks the wiring without burning GPU time.")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Disable vLLM rollouts (slower but avoids vLLM-specific bugs).")
    parser.add_argument("--vllm-gpu-memory", type=float, default=0.35,
                        help="Fraction of GPU memory vLLM may use in colocate mode.")
    args = parser.parse_args()

    # Imports are deferred so `--help` works without loading torch.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from src.rlvr_canary import binary_correctness_reward, load_gsm8k_prompts

    if args.smoke:
        args.train_limit = 16
        args.max_steps = 4
        args.save_steps = 1000  # don't write a checkpoint during smoke
        args.logging_steps = 1

    print(f"[canary] loading {args.train_limit} GSM8K train prompts ...")
    train_records = load_gsm8k_prompts(split="train", limit=args.train_limit)
    if not train_records:
        sys.exit("Failed to load GSM8K — check internet / HF cache.")

    # TRL wants a HF Dataset object so the columns travel through to the
    # reward function via **kwargs (we need `answer`).
    from datasets import Dataset
    train_ds = Dataset.from_list([
        {"prompt": r["prompt"], "answer": r["answer"]} for r in train_records
    ])

    print(f"[canary] loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only generation requires left padding so the tail of every
    # prompt aligns at the same column — otherwise the first generated token
    # is conditioned on stale pad positions.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # Phase 1 = full FT. No LoRA wrapping. Phase 2 will add it here.
    model.config.use_cache = False

    # TRL 0.29 GRPO: `per_device_train_batch_size` is in completions, not prompts;
    # `generation_batch_size` must be a multiple of `num_generations`. We size
    # generation_batch_size as (prompts_per_device * num_generations) so the
    # CLI flag still reads as "distinct prompts per device per step".
    gen_batch_size = args.per_device_batch_size * args.num_generations
    vllm_kwargs: dict = {}
    if not args.no_vllm:
        # Colocate mode: vLLM shares the GPU with the trainer. Cuts step time
        # ~3-5x vs HF .generate(). Needs a memory budget so we don't OOM the
        # trainer half. 0.35 leaves room for the policy + ref model + grads.
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
        remove_unused_columns=False,  # keep `answer` column for the reward fn
        seed=20260407,
        **vllm_kwargs,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=train_ds,
        reward_funcs=[binary_correctness_reward],
    )

    print(f"[canary] starting GRPO training: {args.max_steps} steps, "
          f"{args.num_generations} gens/prompt, "
          f"effective rollouts/step = {args.per_device_batch_size * args.num_generations}")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[canary] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
