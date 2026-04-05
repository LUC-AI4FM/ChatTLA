"""
train_kto.py — KTO (Kahneman-Tversky Optimization) for TLA+ spec quality.

KTO works with unpaired binary feedback (desirable/undesirable) instead of
matched preference pairs. This is a better fit for our data:
  - desirable: gold specs that pass semantic checks (invariants matter, states > 1)
  - undesirable: bronze specs (SANY fails) and silver specs (TLC fails)
  - replay: original SFT training examples mixed in to prevent catastrophic forgetting

Why KTO over SFT/DPO:
  - SFT: trains on vacuously-correct specs (the gold problem)
  - DPO: requires matched pairs, unstable on <100 pairs, catastrophic forgetting
  - KTO: stable with unpaired data, implicit KL constraint prevents forgetting,
    works well with 50-200 labeled examples

Refs: Ethayarajh et al. 2024 "KTO: Model Alignment as Prospect Theoretic Optimization"

Usage (usually invoked from rl_loop.py retrain):
    python -m src.training.train_kto --base-model outputs/merged_model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from datasets import Dataset

from src.training.dataset_builder import _DEVELOPER_PROMPT

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DPO_JSONL = _REPO_ROOT / "data" / "processed" / "rl" / "dpo_pairs.jsonl"
_SFT_JSONL = _REPO_ROOT / "data" / "processed" / "rl" / "sft_examples.jsonl"
_TRAIN_JSONL = _REPO_ROOT / "data" / "processed" / "train.jsonl"
_KTO_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "checkpoints_kto"


def _load_kto_data(
    max_samples: int | None = None,
    max_chars: int = 4000,
    include_replay: bool = True,
) -> list[dict]:
    """Build KTO dataset from RL loop outputs + original training data.

    KTO format: each row has {prompt, completion, label}
      label=True  → desirable (gold specs with real semantic content)
      label=False → undesirable (bronze/silver specs)

    We also include a replay buffer of original SFT data (all labeled True)
    to anchor the model and prevent catastrophic forgetting.
    """
    rows: list[dict] = []
    skipped = 0

    # 1. From DPO pairs: chosen=desirable, rejected=undesirable
    if _DPO_JSONL.is_file():
        with _DPO_JSONL.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p, c, r = o.get("prompt"), o.get("chosen"), o.get("rejected")
                if not p or not c or not r:
                    continue
                if len(str(p)) + max(len(str(c)), len(str(r))) > max_chars:
                    skipped += 1
                    continue
                # Gold chosen → desirable
                if o.get("chosen_tier") == "gold":
                    rows.append({"prompt": str(p), "completion": str(c).strip(), "label": True})
                # Rejected → undesirable
                rows.append({"prompt": str(p), "completion": str(r).strip(), "label": False})

    # 2. From SFT examples (RL-generated gold specs) → desirable
    if _SFT_JSONL.is_file():
        with _SFT_JSONL.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = o.get("messages", [])
                if len(msgs) < 3:
                    continue
                # Extract user prompt and assistant response
                user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), None)
                # Get the final channel response (the actual spec)
                asst_msgs = [m for m in msgs if m.get("role") == "assistant"]
                final_msg = next(
                    (m["content"] for m in asst_msgs if m.get("channel") == "final"),
                    asst_msgs[-1]["content"] if asst_msgs else None,
                )
                if not user_msg or not final_msg:
                    continue
                if len(user_msg) + len(final_msg) > max_chars:
                    skipped += 1
                    continue
                rows.append({"prompt": user_msg, "completion": final_msg.strip(), "label": True})

    # 3. Replay buffer from original SFT training data → desirable
    #    This prevents catastrophic forgetting of base TLA+ knowledge
    if include_replay and _TRAIN_JSONL.is_file():
        replay_count = 0
        with _TRAIN_JSONL.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = o.get("messages", [])
                if len(msgs) < 3:
                    continue
                user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), None)
                asst_msgs = [m for m in msgs if m.get("role") == "assistant"]
                final_msg = next(
                    (m["content"] for m in asst_msgs if m.get("channel") == "final"),
                    asst_msgs[-1]["content"] if asst_msgs else None,
                )
                if not user_msg or not final_msg:
                    continue
                if len(user_msg) + len(final_msg) > max_chars:
                    skipped += 1
                    continue
                rows.append({"prompt": user_msg, "completion": final_msg.strip(), "label": True})
                replay_count += 1
        print(f"[train_kto] Replay buffer: {replay_count} examples from original training data")

    if skipped:
        print(f"[train_kto] Skipped {skipped} rows exceeding {max_chars} char limit")

    if max_samples:
        rows = rows[:max_samples]

    n_pos = sum(1 for r in rows if r["label"])
    n_neg = sum(1 for r in rows if not r["label"])
    print(f"[train_kto] Dataset: {len(rows)} total ({n_pos} desirable, {n_neg} undesirable)")
    return rows


def _build_kto_dataset(tokenizer, max_samples: int | None = None) -> Dataset | None:
    """Build HF Dataset in KTO format with tokenized prompts."""
    raw = _load_kto_data(max_samples=max_samples)
    if len(raw) < 4:
        print(f"[train_kto] Skip: need >= 4 examples, found {len(raw)}")
        return None

    formatted: list[dict] = []
    for row in raw:
        messages = [
            {"role": "developer", "content": _DEVELOPER_PROMPT},
            {"role": "user", "content": row["prompt"]},
        ]
        try:
            prompt_txt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"[train_kto] apply_chat_template failed: {e}")
            return None
        formatted.append({
            "prompt": prompt_txt,
            "completion": row["completion"],
            "label": row["label"],
        })

    return Dataset.from_list(formatted)


def run_kto(
    model,
    tokenizer,
    *,
    device_map: str = "auto",
    max_gpu_memory_mb: int | None = None,
    max_length: int = 3072,
    learning_rate: float = 5e-7,
    num_epochs: int = 1,
    smoke_test: bool = False,
    peft_config=None,
) -> bool:
    """Run KTO training on the model.

    Parameters
    ----------
    learning_rate : 5e-7 default — conservative to prevent forgetting.
                    DPO at 5e-6 caused catastrophic failure; KTO is more
                    stable but we still keep it low.
    """
    try:
        from trl import KTOTrainer, KTOConfig
    except ImportError as e:
        print(f"[train_kto] SKIP: could not import TRL KTO ({e})")
        return False

    ds = _build_kto_dataset(tokenizer, max_samples=16 if smoke_test else None)
    if ds is None:
        return False

    _KTO_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_kw = dict(
        output_dir=str(_KTO_CHECKPOINT_DIR),
        per_device_train_batch_size=2,       # KTO requires batch > 1 for KL term
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=num_epochs,
        max_steps=8 if smoke_test else -1,
        logging_steps=1,
        save_steps=50 if not smoke_test else 8,
        save_total_limit=2,
        # KTO hyperparameters
        beta=0.1,                        # KL penalty weight — moderate
        desirable_weight=1.0,            # weight for desirable examples
        undesirable_weight=1.0,          # weight for undesirable examples
        max_length=min(max_length, 4096),
        max_prompt_length=1024,
        max_completion_length=3072,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    try:
        kto_args = KTOConfig(**cfg_kw)
    except TypeError as e:
        # Some TRL versions have different param names
        print(f"[train_kto] KTOConfig init warning: {e}")
        # Try without the newer params
        for key in ["max_prompt_length", "max_completion_length", "desirable_weight", "undesirable_weight"]:
            cfg_kw.pop(key, None)
        kto_args = KTOConfig(**cfg_kw)

    print(f"[train_kto] Starting KTO on {len(ds)} examples (lr={learning_rate}, beta={kto_args.beta})...")

    try:
        trainer = KTOTrainer(
            model=model,
            args=kto_args,
            train_dataset=ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    except TypeError:
        # Older TRL versions use tokenizer=
        trainer = KTOTrainer(
            model=model,
            args=kto_args,
            train_dataset=ds,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

    try:
        trainer.train()
    except Exception as e:
        print(f"[train_kto] KTO training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"[train_kto] KTO complete. Checkpoints under {_KTO_CHECKPOINT_DIR}")
    return True


def main() -> int:
    import argparse
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="KTO training for TLA+ quality alignment")
    parser.add_argument("--base-model", default=None, help="Path to merged model or HF ID")
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    from src.training.train import load_model_and_tokenizer, load_lora_config

    model, tokenizer = load_model_and_tokenizer(
        device_map="auto",
        base_model=args.base_model,
    )

    lora_cfg = load_lora_config()

    ok = run_kto(
        model,
        tokenizer,
        max_length=args.max_length,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        smoke_test=args.smoke_test,
        peft_config=lora_cfg,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
