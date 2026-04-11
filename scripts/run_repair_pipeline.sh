#!/bin/bash
set -euo pipefail
cd /home/espencer2/ChatTLA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=outputs/logs/repair_pipeline.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] ===== Ralph Repair Pipeline =====" | tee -a "$LOG"

# Unload any Ollama models hogging VRAM (except the one we need)
for m in chattla:20b-v17 chattla:20b-v16 chattla:20b-v14; do
    curl -s http://localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" > /dev/null 2>&1 || true
done
sleep 3

# ─── Phase 1: Collect repair trajectories ──────────────────────────────────
echo "[$(ts)] Phase 1: Collecting Ralph trajectories..." | tee -a "$LOG"
.venv/bin/python -u -m scripts.collect_ralph_trajectories \
    --model chattla:20b \
    --max-iters 6 \
    --workers 4 \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] Trajectory collection FAILED" | tee -a "$LOG"
        exit 1
    }
echo "[$(ts)] Phase 1 complete." | tee -a "$LOG"

# Check we got enough data
PAIRS=$(wc -l < data/processed/ralph_repair_pairs.jsonl)
echo "[$(ts)] Got $PAIRS repair pairs" | tee -a "$LOG"
if [ "$PAIRS" -lt 20 ]; then
    echo "[$(ts)] Too few pairs ($PAIRS < 20). Aborting." | tee -a "$LOG"
    exit 1
fi

# ─── Phase 2: Unload Ollama, free VRAM for training ───────────────────────
echo "[$(ts)] Unloading Ollama models for GRPO training..." | tee -a "$LOG"
curl -s http://localhost:11434/api/generate -d '{"model":"chattla:20b","keep_alive":0}' > /dev/null 2>&1 || true
sleep 5

# ─── Phase 3: Repair GRPO training ────────────────────────────────────────
echo "[$(ts)] Phase 3: Repair GRPO training..." | tee -a "$LOG"
.venv/bin/python -u -m scripts.train_rl_repair \
    --max-steps 300 \
    --num-generations 4 \
    --max-completion-length 1024 \
    --difficulty all \
    --save-steps 50 \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] Repair GRPO FAILED" | tee -a "$LOG"
        exit 1
    }
echo "[$(ts)] Phase 3 complete." | tee -a "$LOG"

# ─── Phase 4: Merge LoRA + deploy to Ollama ──────────────────────────────
echo "[$(ts)] Phase 4: Merging LoRA and deploying..." | tee -a "$LOG"

# Find latest checkpoint
CKPT=$(ls -td outputs/checkpoints_rl_repair/checkpoint-* 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    CKPT="outputs/checkpoints_rl_repair/final"
fi
echo "[$(ts)] Using checkpoint: $CKPT" | tee -a "$LOG"

.venv/bin/python -m src.training.merge_lora "$CKPT" \
    --output outputs/merged_model_repair \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] LoRA merge FAILED" | tee -a "$LOG"
        exit 1
    }

.venv/bin/python -m src.training.publish_ollama \
    --model-dir outputs/merged_model_repair \
    --tag chattla:20b-repair \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] Ollama publish FAILED" | tee -a "$LOG"
        exit 1
    }

# ─── Phase 5: Evaluate with Ralph ────────────────────────────────────────
echo "[$(ts)] Phase 5: Evaluating repair model with Ralph..." | tee -a "$LOG"

# Quick 3-problem Ralph eval
for PROMPT in \
    "A mutual exclusion algorithm for N processes" \
    "A two-phase commit protocol over N resource managers" \
    "A bounded FIFO queue with enqueue and dequeue operations"; do
    echo "" | tee -a "$LOG"
    echo "[$(ts)] Ralph eval: $PROMPT" | tee -a "$LOG"
    python /home/espencer2/ralph-tla/ralph_tla.py \
        --model chattla:20b-repair \
        --iters 6 \
        --out outputs/eval/ralph_repair \
        "$PROMPT" \
        2>&1 | tee -a "$LOG" || true
done

# Holdout eval
echo "[$(ts)] Running holdout eval..." | tee -a "$LOG"
.venv/bin/python -u -m scripts.eval_3shot_tlc_tlaps 30 \
    --model chattla:20b-repair \
    --output outputs/eval/holdout_repair.json \
    2>&1 | tee -a "$LOG" || true

echo "" | tee -a "$LOG"
echo "[$(ts)] ===== REPAIR PIPELINE COMPLETE =====" | tee -a "$LOG"
