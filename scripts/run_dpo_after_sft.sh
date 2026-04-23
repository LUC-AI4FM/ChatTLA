#!/bin/bash
# Wait for SFT to finish, find latest checkpoint, run DPO
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -e

SFT_PID=$1
if [ -z "$SFT_PID" ]; then echo "Usage: $0 <sft_pid>"; exit 1; fi

echo "[dpo-runner] Waiting for SFT (PID $SFT_PID) to finish..."
while ps -p "$SFT_PID" > /dev/null 2>&1; do sleep 30; done
echo "[dpo-runner] SFT complete."

# Find latest checkpoint
CKPT=$(ls -td $REPO/outputs/checkpoints/checkpoint-* 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "[dpo-runner] No checkpoint found!"; exit 1; fi
echo "[dpo-runner] Using checkpoint: $CKPT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
python -m src.training.train_dpo --checkpoint "$CKPT" --max-length 1024 2>&1 | tee outputs/logs/dpo_$(date +%Y%m%d_%H%M%S).log
