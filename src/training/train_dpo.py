"""
train_dpo.py — Optional DPO refinement after SFT using RL-collected preference pairs.

Reads data/processed/rl/dpo_pairs.jsonl (only rows with chosen_tier == "gold").
Formats prompts with the same developer prompt as dataset_builder / SFT.

Requires a compatible trl + rich stack (see requirements.txt). If DPO imports fail,
SFT still completes; this phase is skipped with a warning.

Usage (usually invoked from train.py --dpo-after):
  python -m src.training.train_dpo --checkpoint outputs/checkpoints/checkpoint-100
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from datasets import Dataset

from src.training.dataset_builder import _DEVELOPER_PROMPT

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DPO_JSONL = _REPO_ROOT / "data" / "processed" / "rl" / "dpo_pairs.jsonl"
_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "checkpoints"


def _load_gold_dpo_rows(max_samples: int | None = None) -> list[dict]:
    if not _DPO_JSONL.is_file():
        return []
    rows: list[dict] = []
    with _DPO_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("chosen_tier") != "gold":
                continue
            p, c, r = o.get("prompt"), o.get("chosen"), o.get("rejected")
            if not p or not c or not r:
                continue
            rows.append({"prompt": p, "chosen": str(c).strip(), "rejected": str(r).strip()})
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def _build_dpo_dataset(tokenizer, max_samples: int | None = None) -> Dataset | None:
    raw = _load_gold_dpo_rows(max_samples=max_samples)
    if len(raw) < 2:
        print(f"[train_dpo] Skip: need >=2 gold DPO pairs, found {len(raw)} in {_DPO_JSONL}")
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
            print(f"[train_dpo] apply_chat_template failed: {e}")
            return None
        formatted.append(
            {
                "prompt": prompt_txt,
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
        )

    return Dataset.from_list(formatted)


def run_after_sft(
    model,
    tokenizer,
    *,
    device_map: str = "auto",
    max_gpu_memory_mb: int | None = None,
    max_length: int = 3072,
    smoke_test: bool = False,
) -> bool:
    """
    Run DPO on the in-memory PEFT model (after SFT). Returns True if DPO ran.
    """
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        print(f"[train_dpo] SKIP: could not import TRL DPO ({e}). Upgrade trl/rich: pip install -U trl 'rich>=14'")
        return False

    ds = _build_dpo_dataset(tokenizer, max_samples=8 if smoke_test else None)
    if ds is None:
        return False

    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_kw = dict(
        output_dir=str(_CHECKPOINT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4 if not smoke_test else 1,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=1,
        max_steps=4 if smoke_test else -1,
        logging_steps=1,
        save_steps=50 if not smoke_test else 4,
        save_total_limit=2,
        beta=0.05,
        max_length=min(max_length, 4096),
        max_prompt_length=min(max_length // 2, 2048),
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    try:
        dpo_args = DPOConfig(**cfg_kw, is_peft_model=True)
    except TypeError:
        dpo_args = DPOConfig(**cfg_kw)

    print(f"[train_dpo] Starting DPO on {len(ds)} gold pairs (beta={dpo_args.beta})...")
    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=ds,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=ds,
            tokenizer=tokenizer,
        )
    try:
        trainer.train()
    except Exception as e:
        print(f"[train_dpo] DPO training failed (non-fatal for SFT): {e}")
        return False

    print("[train_dpo] DPO phase complete. Checkpoints under outputs/checkpoints/")
    return True


def main() -> int:
    import argparse
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Standalone DPO pass (loads checkpoint + dpo_pairs)")
    parser.add_argument("--checkpoint", required=True, help="PEFT checkpoint dir")
    parser.add_argument("--max-length", type=int, default=3072)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        print(f"[train_dpo] Missing checkpoint: {ckpt}")
        return 1

    try:
        from trl import DPOConfig  # noqa: F401
    except ImportError as e:
        print(f"[train_dpo] {e}")
        return 1

    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-8B",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(ckpt))
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ok = run_after_sft(
        model,
        tokenizer,
        device_map="auto",
        max_length=args.max_length,
        smoke_test=False,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
