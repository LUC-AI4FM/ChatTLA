"""
merge_lora.py — Merge the trained LoRA adapter back into gpt-oss-20b base weights.

After fine-tuning, `outputs/checkpoints/` contains only the small LoRA adapter
(~300 MB) alongside the base model.  To deploy via Ollama we need a standalone
merged model.  This script:

1. Loads the base gpt-oss-20b in BF16 (no quantization — we want clean weights)
2. Loads the LoRA adapter from the checkpoint
3. Merges adapter weights into base weights (W_merged = W_base + A @ B * scale)
4. Unloads the adapter
5. Saves the merged model in safetensors format to outputs/merged_model/
6. Saves the tokenizer alongside the model

After this script completes, use src/inference/convert_to_gguf.py to convert
the merged model to GGUF for Ollama deployment.

Memory note
-----------
Merging requires holding the full BF16 model in memory (~40 GB).  Prefer
`CUDA_VISIBLE_DEVICES=0,1` (default in this script) so `device_map="auto"`
can shard across two GPUs.  Pinning a single GPU often OOMs during
`merge_and_unload()`.  If CUDA fails, use `--device cpu` (slow; needs ~64 GB+
system RAM).

Usage
-----
    # Merge from the latest checkpoint (uses GPUs 0,1 by default):
    python -m src.training.merge_lora

    # Merge from a specific checkpoint:
    python -m src.training.merge_lora \\
        --checkpoint outputs/checkpoints/checkpoint-1000

    # CPU merge (slow but no VRAM limit):
    python -m src.training.merge_lora --device cpu
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT       = Path(__file__).resolve().parents[2]
_CHECKPOINT_DIR  = _REPO_ROOT / "outputs" / "checkpoints"
_MERGED_OUT      = _REPO_ROOT / "outputs" / "merged_model"

MODEL_ID = "openai/gpt-oss-20b"


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the highest-numbered checkpoint in a directory.

    Skips directories whose name can't be parsed as 'checkpoint-<int>'
    (e.g. renamed/archived checkpoints like 'checkpoint-24-broken-r2').
    """
    valid = []
    for d in checkpoint_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("checkpoint-"):
            continue
        parts = d.name.split("-")
        # expect exactly "checkpoint-<number>"
        if len(parts) == 2:
            try:
                valid.append((int(parts[1]), d))
            except ValueError:
                continue
    valid.sort(key=lambda t: t[0])
    return valid[-1][1] if valid else None


_DPO_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "checkpoints_dpo"
_KTO_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "checkpoints_kto"


def merge(
    checkpoint_path: Path | None = None,
    output_path: Path = _MERGED_OUT,
    device: str = "auto",
    dpo_checkpoint_path: Path | None = None,
    base_model: str | None = None,
) -> None:
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(_CHECKPOINT_DIR)
    # Fallback: check KTO checkpoint dir (KTO trains standalone, not SFT+DPO)
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(_KTO_CHECKPOINT_DIR)
        if checkpoint_path:
            print(f"[merge_lora] Using KTO checkpoint: {checkpoint_path}")
    if checkpoint_path is None:
        print(f"[merge_lora] ERROR: No checkpoint found in {_CHECKPOINT_DIR} or {_KTO_CHECKPOINT_DIR}")
        sys.exit(1)

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Use --base-model if provided (critical for DPO-only mode where LoRA was
    # trained on the merged model, not the original base model).
    model_id = base_model or MODEL_ID
    print(f"[merge_lora] Loading base model: {model_id} (device={device})")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,    # gpt-oss requires custom modeling code
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"[merge_lora] Loading SFT LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, str(checkpoint_path))

    print("[merge_lora] Merging SFT adapter weights into base model...")
    model = model.merge_and_unload()

    # Optional second-stage DPO merge
    if dpo_checkpoint_path is not None:
        print(f"[merge_lora] Loading DPO LoRA adapter from: {dpo_checkpoint_path}")
        model = PeftModel.from_pretrained(model, str(dpo_checkpoint_path))
        print("[merge_lora] Merging DPO adapter weights into SFT-merged model...")
        model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    # Clean old weight files before saving to avoid duplicates (e.g. 39GB safetensors + 39GB pytorch_model.bin)
    for old_weights in output_path.glob("pytorch_model*.bin"):
        old_weights.unlink()
        print(f"[merge_lora] Cleaned old weights: {old_weights.name}")
    for old_weights in output_path.glob("model*.safetensors"):
        old_weights.unlink()
        print(f"[merge_lora] Cleaned old weights: {old_weights.name}")

    print(f"[merge_lora] Saving merged model → {output_path}")
    # Use save_pretrained to write proper safetensors + index files that
    # llama.cpp's convert_hf_to_gguf.py expects.  Fall back to torch.save
    # only if save_pretrained fails (e.g. unsupported weight dtypes).
    try:
        model.save_pretrained(output_path, safe_serialization=True)
        print("[merge_lora] Saved via save_pretrained (safetensors)")
    except Exception as exc:
        print(f"[merge_lora] save_pretrained failed ({exc}), falling back to torch.save")
        state_dict = model.state_dict()
        torch.save(state_dict, output_path / "pytorch_model.bin")
        model.config.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    stage = "SFT+DPO" if dpo_checkpoint_path else "SFT"
    print(f"[merge_lora] Done. Merged model ({stage}) saved to {output_path}")
    print("[merge_lora] Next step: run src/inference/convert_to_gguf.py to build GGUF for Ollama.")


if __name__ == "__main__":
    import argparse

    # Default: same multi-GPU visibility as training (rl_loop / train). Override
    # with CUDA_VISIBLE_DEVICES=0 or =1 for a single card. Pinning only GPU 1
    # often OOMs during merge (full 20B BF16 + LoRA matmuls). Use --device cpu
    # if CUDA_VISIBLE_DEVICES="" (e.g. CPU-only merge subprocess).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

    parser = argparse.ArgumentParser(description="Merge LoRA adapter into gpt-oss-20b base weights")
    parser.add_argument("--checkpoint", default=None, help="SFT checkpoint dir (default: latest in outputs/checkpoints)")
    parser.add_argument("--dpo-checkpoint", default=None, help="DPO checkpoint dir for two-stage merge (default: latest in outputs/checkpoints_dpo if it exists)")
    parser.add_argument("--output",     default=str(_MERGED_OUT), help="Output directory for merged model")
    parser.add_argument("--device",     default="auto", help="Device map: 'auto', 'cuda', or 'cpu'")
    parser.add_argument("--base-model", default=None,
                        help="Path or HF ID to use as base model instead of openai/gpt-oss-20b "
                             "(required when LoRA was trained on a previously merged model)")
    args = parser.parse_args()

    # Auto-detect DPO checkpoint if not specified and checkpoints_dpo/ exists
    dpo_ckpt = None
    if args.dpo_checkpoint:
        dpo_ckpt = Path(args.dpo_checkpoint)
    elif _DPO_CHECKPOINT_DIR.is_dir():
        dpo_ckpt = find_latest_checkpoint(_DPO_CHECKPOINT_DIR)
        if dpo_ckpt:
            print(f"[merge_lora] Auto-detected DPO checkpoint: {dpo_ckpt}")

    merge(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        output_path=Path(args.output),
        device=args.device,
        dpo_checkpoint_path=dpo_ckpt,
        base_model=args.base_model,
    )
