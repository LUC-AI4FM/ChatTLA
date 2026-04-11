#!/bin/bash
# resume_phase2.sh — Resume from Phase 2 (GRPO) after DPO adapter merge.
# Phase 1A: 189 DPO pairs ✓
# Phase 1B: DPO training ✓ (adapter at checkpoints_dpo_piecewise/)
# Now: merge DPO adapter → run GRPO → flywheel
set -euo pipefail

cd /home/espencer2/ChatTLA
PY=".venv/bin/python -u"
LOG=outputs/logs/pipeline_master.log
mkdir -p outputs/logs outputs/eval

ts() { date '+%Y-%m-%d %H:%M:%S'; }
abort() {
    echo "[$(ts)] *** PIPELINE ABORTED at stage: $1 (exit=$2) ***" | tee -a "$LOG"
    exit "$2"
}

echo "" | tee -a "$LOG"
echo "[$(ts)] === PIPELINE RESUME (Phase 2: merge + GRPO) ===" | tee -a "$LOG"

# ── Unload Ollama ────────────────────────────────────────────────────────
echo "[$(ts)] Unloading Ollama models..." | tee -a "$LOG"
for m in chattla:20b chattla:20b-v17 chattla:20b-v16 chattla:20b-v14; do
    curl -s http://localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" > /dev/null 2>&1 || true
done
sleep 3

# ── Merge DPO adapter into base model ───────────────────────────────────
echo "[$(ts)] ===== MERGE DPO ADAPTER =====" | tee -a "$LOG"

DPO_CKPT=outputs/checkpoints_dpo_piecewise
BASE_MODEL=outputs/merged_model_v13
MERGED_OUT=outputs/merged_model_dpo_piecewise

if [ -d "$MERGED_OUT" ] && [ -f "$MERGED_OUT/config.json" ]; then
    echo "[$(ts)] Merged model already exists at $MERGED_OUT, skipping merge" | tee -a "$LOG"
else
    echo "[$(ts)] Merging $DPO_CKPT onto $BASE_MODEL -> $MERGED_OUT (~10 min)" | tee -a "$LOG"
    $PY -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

print('Loading base model...')
model = AutoModelForCausalLM.from_pretrained(
    '$BASE_MODEL', torch_dtype=torch.bfloat16, device_map='auto', use_cache=False)
print('Loading DPO adapter...')
model = PeftModel.from_pretrained(model, '$DPO_CKPT')
print('Merging...')
model = model.merge_and_unload()
print('Saving merged model to $MERGED_OUT ...')
model.save_pretrained('$MERGED_OUT')
tok = AutoTokenizer.from_pretrained('$BASE_MODEL')
tok.save_pretrained('$MERGED_OUT')
print('Merge complete.')
" 2>&1 | tee -a "$LOG" \
    || abort "merge_dpo" $?
fi

echo "[$(ts)] Merged model: $(du -sh $MERGED_OUT | cut -f1)" | tee -a "$LOG"

# ── Phase 2: Full-Spec GRPO ─────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[$(ts)] ===== PHASE 2: FULL-SPEC GRPO (200 steps) =====" | tee -a "$LOG"
echo "[$(ts)] Disk: $(df -h /home/espencer2 | tail -1)" | tee -a "$LOG"

# Free GPU memory from the merge
$PY -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>&1 | tee -a "$LOG"
# Also try to unload any lingering processes
sleep 3

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -m scripts.train_rl_fullspec \
    --model "$MERGED_OUT" \
    --max-steps 200 \
    --num-generations 4 \
    --max-completion-length 1536 \
    --save-steps 50 \
    2>&1 | tee -a "$LOG" \
    || {
        echo "[$(ts)] GRPO failed at 4 gens/1536 — trying 2 gens/1024" | tee -a "$LOG"
        # Clear GPU
        $PY -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
        sleep 5
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY -m scripts.train_rl_fullspec \
            --model "$MERGED_OUT" \
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
echo "[$(ts)] Disk: $(df -h /home/espencer2 | tail -1)" | tee -a "$LOG"
if [ -f outputs/logs/flywheel_metrics.jsonl ]; then
    echo "[$(ts)] Flywheel metrics:" | tee -a "$LOG"
    tail -5 outputs/logs/flywheel_metrics.jsonl | tee -a "$LOG"
fi
