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
Merging requires holding the full BF16 model in memory (~40 GB).  This must
run on GPU 1 alone.  If GPU 1 has insufficient VRAM, use `device_map="cpu"`
(slower but works with 64+ GB system RAM).

Usage
-----
    # Merge from the latest checkpoint:
    CUDA_VISIBLE_DEVICES=1 python -m src.training.merge_lora

    # Merge from a specific checkpoint:
    CUDA_VISIBLE_DEVICES=1 python -m src.training.merge_lora \\
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
    """Find the highest-numbered checkpoint in a directory."""
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def merge(
    checkpoint_path: Path | None = None,
    output_path: Path = _MERGED_OUT,
    device: str = "auto",
) -> None:
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(_CHECKPOINT_DIR)
    if checkpoint_path is None:
        print(f"[merge_lora] ERROR: No checkpoint found in {_CHECKPOINT_DIR}")
        sys.exit(1)

    print(f"[merge_lora] Loading base model: {MODEL_ID} (device={device})")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"[merge_lora] Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, str(checkpoint_path))

    print("[merge_lora] Merging adapter weights into base model...")
    model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[merge_lora] Saving merged model → {output_path}")
    model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    print(f"[merge_lora] Done. Merged model saved to {output_path}")
    print("[merge_lora] Next step: run src/inference/convert_to_gguf.py to build GGUF for Ollama.")


if __name__ == "__main__":
    import argparse

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    parser = argparse.ArgumentParser(description="Merge LoRA adapter into gpt-oss-20b base weights")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint dir (default: latest in outputs/checkpoints)")
    parser.add_argument("--output",     default=str(_MERGED_OUT), help="Output directory for merged model")
    parser.add_argument("--device",     default="auto", help="Device map: 'auto', 'cuda', or 'cpu'")
    args = parser.parse_args()

    merge(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        output_path=Path(args.output),
        device=args.device,
    )
