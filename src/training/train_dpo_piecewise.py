"""
train_dpo_piecewise.py — DPO training on piecewise + full-spec preference pairs.

Loads pairs from data/processed/piecewise_dpo_pairs.jsonl, which contains:
  - Piecewise pairs: (SANY-passing piece, SANY-failing piece) for VARIABLES,
    TypeOK, Init, Next, Invariants
  - Full-spec pairs: (higher partial_credit spec, lower partial_credit spec)

Curriculum: Trains in order VARIABLES → TypeOK → Init → Next → Invariants → full_spec,
with easier pieces first (DeepSeek-Prover-V2 subgoal decomposition).

Usage:
    # Standalone (loads base model + fresh LoRA)
    python -m src.training.train_dpo_piecewise --base-model outputs/merged_model_v14

    # Smoke test
    python -m src.training.train_dpo_piecewise --base-model outputs/merged_model_v14 --smoke
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DPO_PAIRS = _REPO_ROOT / "data" / "processed" / "piecewise_dpo_pairs.jsonl"
_OUTPUT_DIR = _REPO_ROOT / "outputs" / "checkpoints_dpo_piecewise"

# Curriculum order: easiest → hardest
_PIECE_ORDER = ["VARIABLES", "TypeOK", "Init", "Next", "Invariants", "full_spec"]


def _load_pairs(
    max_samples: int | None = None,
    max_chars: int = 6000,
    min_reward_gap: float = 0.01,
) -> list[dict]:
    """Load and filter piecewise DPO pairs."""
    if not _DPO_PAIRS.is_file():
        print(f"[dpo_piecewise] No pairs file at {_DPO_PAIRS}")
        return []

    rows: list[dict] = []
    skipped = 0
    with _DPO_PAIRS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            p, c, r = o.get("prompt", ""), o.get("chosen", ""), o.get("rejected", "")
            if not p or not c or not r:
                continue
            # Skip if reward gap is too small (noise)
            gap = abs(o.get("chosen_reward", 1.0) - o.get("rejected_reward", 0.0))
            if gap < min_reward_gap:
                skipped += 1
                continue
            # Skip outlier rows that would blow up GPU memory
            if len(p) + max(len(c), len(r)) > max_chars:
                skipped += 1
                continue
            rows.append({
                "prompt": p,
                "chosen": c,
                "rejected": r,
                "piece_name": o.get("piece_name", "unknown"),
            })
            if max_samples and len(rows) >= max_samples:
                break

    if skipped:
        print(f"[dpo_piecewise] Skipped {skipped} rows (too long or small reward gap)")

    # Sort by curriculum order
    piece_rank = {name: i for i, name in enumerate(_PIECE_ORDER)}
    rows.sort(key=lambda r: piece_rank.get(r["piece_name"], 99))

    return rows


def _build_dataset(tokenizer, max_samples: int | None = None) -> Dataset | None:
    """Build HF Dataset from piecewise DPO pairs."""
    raw = _load_pairs(max_samples=max_samples)
    if len(raw) < 2:
        print(f"[dpo_piecewise] Need >=2 pairs, found {len(raw)}")
        return None

    formatted: list[dict] = []
    for row in raw:
        # The prompt is already formatted from build_piecewise_dpo.py
        # For DPO, we just need prompt + chosen + rejected as strings
        formatted.append({
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        })

    print(f"[dpo_piecewise] Loaded {len(formatted)} pairs")
    # Report by piece type
    from collections import Counter
    counts = Counter(r["piece_name"] for r in raw)
    for piece in _PIECE_ORDER:
        if counts[piece]:
            print(f"  {piece}: {counts[piece]} pairs")

    return Dataset.from_list(formatted)


def run_dpo_piecewise(
    model,
    tokenizer,
    *,
    max_length: int = 2048,
    smoke_test: bool = False,
    peft_config=None,
) -> bool:
    """Run piecewise DPO on the given model."""
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        print(f"[dpo_piecewise] SKIP: {e}")
        return False

    ds = _build_dataset(tokenizer, max_samples=8 if smoke_test else None)
    if ds is None:
        return False

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_kw = dict(
        output_dir=str(_OUTPUT_DIR),
        per_device_train_batch_size=1,          # OOM at 2 on 2x RTX 8000
        gradient_accumulation_steps=8,          # effective batch=8
        learning_rate=3e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=2,
        max_steps=4 if smoke_test else -1,
        logging_steps=1,
        save_steps=100 if not smoke_test else 4,
        save_total_limit=3,
        beta=0.05,  # lower than full-spec DPO (0.1) — finer-grained pairs
        max_length=min(max_length, 1536),       # reduced from 2048 to fit memory
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )
    try:
        dpo_args = DPOConfig(**cfg_kw, is_peft_model=True)
    except TypeError:
        dpo_args = DPOConfig(**cfg_kw)

    print(f"[dpo_piecewise] Starting DPO: {len(ds)} pairs, beta={dpo_args.beta}, "
          f"lr={dpo_args.learning_rate}")
    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    except TypeError:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=ds,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

    try:
        trainer.train()
    except Exception as e:
        print(f"[dpo_piecewise] Training failed: {e}")
        return False

    trainer.save_model(str(_OUTPUT_DIR))
    print(f"[dpo_piecewise] Complete. Checkpoints: {_OUTPUT_DIR}")
    return True


def main() -> int:
    import argparse
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True,
                        help="Merged model path (e.g., outputs/merged_model_v14)")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    base = Path(args.base_model)
    if not base.is_dir():
        print(f"[dpo_piecewise] Missing base model: {base}")
        return 1

    print(f"[dpo_piecewise] Loading base model {base} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(base))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Fresh LoRA for DPO
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ok = run_dpo_piecewise(
        model, tokenizer,
        max_length=args.max_length,
        smoke_test=args.smoke,
        peft_config=lora_config,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
