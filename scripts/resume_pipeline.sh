#!/bin/bash
# resume_pipeline.sh — Resume from Phase 1B (DPO training) after OOM fix.
# Phase 1A already completed: 189 DPO pairs in piecewise_dpo_pairs.jsonl
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -euo pipefail

cd "$REPO"
PY=".venv/bin/python -u"
LOG=outputs/logs/pipeline_master.log
mkdir -p outputs/logs outputs/eval

ts() { date '+%Y-%m-%d %H:%M:%S'; }
abort() {
    echo "[$(ts)] *** PIPELINE ABORTED at stage: $1 (exit=$2) ***" | tee -a "$LOG"
    exit "$2"
}

echo "" | tee -a "$LOG"
echo "[$(ts)] === PIPELINE RESUME (from Phase 1B) ===" | tee -a "$LOG"
echo "[$(ts)] Disk: $(df -h "$REPO" | tail -1)" | tee -a "$LOG"
echo "[$(ts)] GPUs: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"

DPO_COUNT=$(wc -l < data/processed/piecewise_dpo_pairs.jsonl 2>/dev/null || echo 0)
echo "[$(ts)] DPO pairs available: $DPO_COUNT" | tee -a "$LOG"

# ── Unload Ollama to free GPU for training ───────────────────────────────
echo "[$(ts)] Unloading Ollama models from GPU..." | tee -a "$LOG"
for m in chattla:20b chattla:20b-v17 chattla:20b-v16 chattla:20b-v14; do
    curl -s http://localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" > /dev/null 2>&1 || true
done
sleep 5
echo "[$(ts)] GPU after unload: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"

# ── Phase 1B: Train Piecewise DPO ───────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[$(ts)] ===== PHASE 1B: TRAIN PIECEWISE DPO (batch=1, accum=8) =====" | tee -a "$LOG"

if [ "$DPO_COUNT" -ge 20 ]; then
    echo "[$(ts)] Training DPO on $DPO_COUNT pairs..." | tee -a "$LOG"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -m src.training.train_dpo_piecewise \
        --base-model outputs/merged_model_v13 \
        2>&1 | tee -a "$LOG" \
        || abort "phase1b_dpo_train" $?

    echo "[$(ts)] Phase 1B complete: DPO training done" | tee -a "$LOG"
else
    echo "[$(ts)] Phase 1B SKIPPED: only $DPO_COUNT pairs (need >= 20)" | tee -a "$LOG"
fi

# ── Phase 2: Full-Spec GRPO ─────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[$(ts)] ===== PHASE 2: FULL-SPEC GRPO (200 steps) =====" | tee -a "$LOG"
echo "[$(ts)] Disk before GRPO: $(df -h "$REPO" | tail -1)" | tee -a "$LOG"

# Unload Ollama again (DPO may have loaded things)
for m in chattla:20b chattla:20b-v17 chattla:20b-v16 chattla:20b-v14; do
    curl -s http://localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" > /dev/null 2>&1 || true
done
sleep 3

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -m scripts.train_rl_fullspec \
    --max-steps 200 \
    --num-generations 4 \
    --max-completion-length 1536 \
    --save-steps 50 \
    2>&1 | tee -a "$LOG" \
    || {
        echo "[$(ts)] GRPO OOM at 4 gens/1536 — falling back to 2 gens/1024" | tee -a "$LOG"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY -m scripts.train_rl_fullspec \
            --max-steps 200 \
            --num-generations 2 \
            --max-completion-length 1024 \
            --save-steps 50 \
            2>&1 | tee -a "$LOG" \
            || abort "phase2_grpo" $?
    }

echo "[$(ts)] Phase 2 complete" | tee -a "$LOG"

# ── Phase 3: Flywheel (3 cycles) ────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[$(ts)] ===== PHASE 3: FLYWHEEL (3 cycles) =====" | tee -a "$LOG"

# Reload Ollama model for inference
curl -s http://localhost:11434/api/generate -d '{"model":"chattla:20b","prompt":"test","stream":false,"options":{"num_predict":1}}' > /dev/null 2>&1
sleep 5

$PY -m scripts.flywheel \
    --cycles 3 \
    --n-prompts 50 \
    --model chattla:20b \
    2>&1 | tee -a "$LOG" \
    || echo "[$(ts)] WARNING: Flywheel exited non-zero" | tee -a "$LOG"

# ── Final ────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[$(ts)] ===== PIPELINE COMPLETE =====" | tee -a "$LOG"
echo "[$(ts)] Disk: $(df -h "$REPO" | tail -1)" | tee -a "$LOG"
if [ -f outputs/logs/flywheel_metrics.jsonl ]; then
    echo "[$(ts)] Flywheel metrics:" | tee -a "$LOG"
    tail -5 outputs/logs/flywheel_metrics.jsonl | tee -a "$LOG"
fi
