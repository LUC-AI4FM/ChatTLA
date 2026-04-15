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
EXISTING_PAIRS=0
if [ -f data/processed/ralph_repair_pairs.jsonl ]; then
    EXISTING_PAIRS=$(wc -l < data/processed/ralph_repair_pairs.jsonl)
fi

if [ "$EXISTING_PAIRS" -ge 200 ]; then
    echo "[$(ts)] Phase 1 SKIPPED: reusing $EXISTING_PAIRS existing repair pairs" | tee -a "$LOG"
else
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
fi

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
# v2 fixes (2026-04-11): the prior run crashed at step 157 on a 12k-token
# prompt (eager attention OOM, 39 GiB attn matrix) and reward_std stayed
# at ~0 because 62% of pairs have before_score=0 (unparseable broken specs;
# both completions also fail to parse → both stuck at 0.15 baseline → no
# variance → no learning).
#
# Filters now applied at dataset load:
#   --min-before-score 0.10  drop unparseable pairs
#   --max-before-score 0.80  drop already-good pairs (no headroom)
#   --max-prompt-tokens 1600 drop the long-tail OOM offenders
# This leaves ~430 gradable pairs centered on score≈0.45.
echo "[$(ts)] Phase 3: Repair GRPO training (4 gens, 384 completion, filtered)..." | tee -a "$LOG"
.venv/bin/python -u -m scripts.train_rl_repair \
    --max-steps 300 \
    --num-generations 4 \
    --max-completion-length 384 \
    --max-prompt-tokens 1600 \
    --min-before-score 0.10 \
    --max-before-score 0.80 \
    --difficulty all \
    --save-steps 25 \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] Repair GRPO failed at 4x384 — retrying at 2x384" | tee -a "$LOG"
        .venv/bin/python -u -m scripts.train_rl_repair \
            --max-steps 300 \
            --num-generations 2 \
            --max-completion-length 384 \
            --max-prompt-tokens 1600 \
            --min-before-score 0.10 \
            --max-before-score 0.80 \
            --difficulty all \
            --save-steps 25 \
            2>&1 | tee -a "$LOG" || {
                echo "[$(ts)] Repair GRPO FAILED" | tee -a "$LOG"
                exit 1
            }
    }
echo "[$(ts)] Phase 3 complete." | tee -a "$LOG"

# ─── Phase 4: Merge LoRA + deploy to Ollama ──────────────────────────────
echo "[$(ts)] Phase 4: Merging LoRA and deploying..." | tee -a "$LOG"

# Find latest numbered checkpoint; skip stale smoke-run checkpoint-2 if training
# produced something later (save-steps=25, so first real ckpt is checkpoint-25).
CKPT=""
for c in $(ls -1d outputs/checkpoints_rl_repair/checkpoint-* 2>/dev/null | sort -t- -k2 -n -r); do
    num=${c##*checkpoint-}
    if [ "$num" -ge 25 ]; then
        CKPT="$c"
        break
    fi
done
if [ -z "$CKPT" ]; then
    CKPT="outputs/checkpoints_rl_repair/final"
fi
echo "[$(ts)] Using checkpoint: $CKPT" | tee -a "$LOG"

# Merge output goes to /data/sdb (root disk is 99% full)
MERGE_OUT=/data/sdb/espencer2/chattla/merged_model_repair
mkdir -p /data/sdb/espencer2/chattla
.venv/bin/python -m src.training.merge_lora \
    --checkpoint "$CKPT" \
    --output "$MERGE_OUT" \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] LoRA merge FAILED" | tee -a "$LOG"
        exit 1
    }

# Symlink from the expected location so downstream tooling finds it
rm -f outputs/merged_model_repair
ln -s "$MERGE_OUT" outputs/merged_model_repair
echo "[$(ts)] Merged model symlinked: outputs/merged_model_repair -> $MERGE_OUT" | tee -a "$LOG"

# Ollama publish: publish_ollama module doesn't exist yet — skip with warning.
# Merged weights are on /data/sdb ready for manual GGUF convert + ollama create.
echo "[$(ts)] Phase 4 Ollama publish SKIPPED (publish_ollama not implemented)" | tee -a "$LOG"
echo "[$(ts)]   To publish manually: convert $MERGE_OUT to GGUF then 'ollama create chattla:20b-repair'" | tee -a "$LOG"

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
