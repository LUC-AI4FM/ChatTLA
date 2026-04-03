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

from datasets import Dataset

from src.training.dataset_builder import _DEVELOPER_PROMPT

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DPO_JSONL = _REPO_ROOT / "data" / "processed" / "rl" / "dpo_pairs.jsonl"
_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "checkpoints"
_DPO_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "checkpoints_dpo"


def _load_gold_dpo_rows(max_samples: int | None = None, max_chars: int = 4000) -> list[dict]:
    if not _DPO_JSONL.is_file():
        return []
    rows: list[dict] = []
    skipped = 0
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
            # Skip outlier rows that would blow up GPU memory
            if len(str(p)) + max(len(str(c)), len(str(r))) > max_chars:
                skipped += 1
                continue
            rows.append({"prompt": p, "chosen": str(c).strip(), "rejected": str(r).strip()})
            if max_samples and len(rows) >= max_samples:
                break
    if skipped:
        print(f"[train_dpo] Skipped {skipped} rows exceeding {max_chars} char limit")
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
    peft_config=None,
) -> bool:
    """
    Run DPO on the model. If peft_config is provided, DPOTrainer wraps the model
    with a fresh LoRA (merge-then-retrain pattern). Otherwise expects a PeftModel.
    """
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        print(f"[train_dpo] SKIP: could not import TRL DPO ({e}). Upgrade trl/rich: pip install -U trl 'rich>=14'")
        return False

    ds = _build_dpo_dataset(tokenizer, max_samples=8 if smoke_test else None)
    if ds is None:
        return False

    _DPO_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_kw = dict(
        output_dir=str(_DPO_CHECKPOINT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2 if not smoke_test else 1,
        learning_rate=5e-6,
        lr_scheduler_type="constant",    # no decay — too few steps for cosine to be useful
        warmup_ratio=0.0,                # no warmup — can't waste steps with tiny dataset
        num_train_epochs=3,              # 3 passes over ~17 samples ≈ 25 gradient updates
        max_steps=4 if smoke_test else -1,
        logging_steps=1,
        save_steps=50 if not smoke_test else 4,
        save_total_limit=2,
        beta=0.1,                        # moderate — 0.05 gave zero signal, stay below 0.2
        max_length=min(max_length, 4096),
        bf16=True,
        gradient_checkpointing=False,
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
        print(f"[train_dpo] DPO training failed (non-fatal for SFT): {e}")
        return False

    print("[train_dpo] DPO phase complete. Checkpoints under outputs/checkpoints_dpo/")
    return True


def main() -> int:
    import argparse
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Standalone DPO pass (loads checkpoint + dpo_pairs)")
    parser.add_argument("--checkpoint", required=True, help="PEFT checkpoint dir")
    parser.add_argument("--max-length", type=int, default=1024)
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

    # Load base model with same device_map strategy as SFT
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        attn_implementation="eager",
        use_cache=False,
        device_map="auto",
        trust_remote_code=True,
    )
    # Load SFT adapter and merge into base weights
    print(f"[train_dpo] Loading SFT adapter from {ckpt} and merging...")
    model = PeftModel.from_pretrained(model, str(ckpt))
    model = model.merge_and_unload()
    # Now model IS the SFT model (no adapter layers).
    # DPOTrainer will wrap it with a fresh LoRA via peft_config.
    # Reference = SFT model (adapter off), Policy = SFT + DPO delta (adapter on).

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Fresh LoRA config for DPO — match SFT's target modules
    import json
    adapter_cfg = json.loads((ckpt / "adapter_config.json").read_text())
    gpu_layers = adapter_cfg.get("layers_to_transform")
    dpo_lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=adapter_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform=gpu_layers,
    )

    ok = run_after_sft(
        model,
        tokenizer,
        device_map="auto",
        max_length=args.max_length,
        smoke_test=False,
        peft_config=dpo_lora,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
