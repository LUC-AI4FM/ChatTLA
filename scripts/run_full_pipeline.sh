#!/bin/bash
# run_full_pipeline.sh — Master pipeline: Phase 1 (DPO) → Phase 2 (GRPO) → Phase 3 (Flywheel)
#
# Designed to run unattended for 36+ hours in tmux.
# Each phase is gated — failure aborts with a clear message.
#
# Usage:
#   tmux new-session -d -s ralph "./scripts/run_full_pipeline.sh" 2>&1 | tee outputs/logs/pipeline_master.log
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -euo pipefail

cd "$REPO"
PY=".venv/bin/python -u"  # unbuffered stdout/stderr
LOG=outputs/logs/pipeline_master.log
mkdir -p outputs/logs outputs/eval

ts() { date '+%Y-%m-%d %H:%M:%S'; }

abort() {
    echo "[$(ts)] *** PIPELINE ABORTED at stage: $1 (exit=$2) ***" | tee -a "$LOG"
    exit "$2"
}

echo "[$(ts)] === FULL PIPELINE START ===" | tee -a "$LOG"
echo "[$(ts)] Disk: $(df -h "$REPO" | tail -1)" | tee -a "$LOG"
echo "[$(ts)] GPUs: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG"

# ── Phase 0: Smoke tests ────────────────────────────────────────────────
echo ""
echo "[$(ts)] ===== PHASE 0: SMOKE TESTS =====" | tee -a "$LOG"

echo "[$(ts)] Smoke: fullspec reward..." | tee -a "$LOG"
$PY -c "
from src.rlvr_canary.fullspec_reward import fullspec_component_reward
r = fullspec_component_reward(completions=['---- MODULE T ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x\\' = (x + 1) % 3\nvars == << x >>\nSpec == Init /\\\\ [][Next]_vars\nTypeOK == x \\\\in 0..2\n===='])
assert r == [1.0], f'Expected [1.0], got {r}'
print('  fullspec_reward: PASS')
" || abort "smoke_reward" $?

echo "[$(ts)] Smoke: fullspec dataset..." | tee -a "$LOG"
$PY -c "
from src.rlvr_canary.fullspec_dataset import load_fullspec_prompts
ex = load_fullspec_prompts(max_per_source=3)
assert len(ex) > 0, 'No prompts loaded'
print(f'  fullspec_dataset: PASS ({len(ex)} prompts)')
" || abort "smoke_dataset" $?

echo "[$(ts)] Smoke tests PASSED" | tee -a "$LOG"

# ── Phase 1: Generate Piecewise DPO Pairs ───────────────────────────────
echo ""
echo "[$(ts)] ===== PHASE 1A: GENERATE PIECEWISE DPO PAIRS =====" | tee -a "$LOG"
echo "[$(ts)] This will take ~4-8 hours (98 specs x 5 pieces x 8 candidates)" | tee -a "$LOG"

# Check Ollama is serving the model
curl -s http://localhost:11434/api/tags > /dev/null || abort "ollama_not_running" 1

$PY -m scripts.build_piecewise_dpo \
    --model chattla:20b \
    --n-candidates 8 \
    --output data/processed/piecewise_dpo_pairs.jsonl \
    2>&1 | tee -a "$LOG" \
    || abort "phase1a_dpo_gen" $?

DPO_COUNT=$(wc -l < data/processed/piecewise_dpo_pairs.jsonl 2>/dev/null || echo 0)
echo "[$(ts)] Phase 1A complete: $DPO_COUNT DPO pairs generated" | tee -a "$LOG"

if [ "$DPO_COUNT" -lt 10 ]; then
    echo "[$(ts)] WARNING: Only $DPO_COUNT pairs — too few for meaningful DPO. Continuing to GRPO." | tee -a "$LOG"
fi

# ── Phase 1B: Train Piecewise DPO ───────────────────────────────────────
echo ""
echo "[$(ts)] ===== PHASE 1B: TRAIN PIECEWISE DPO =====" | tee -a "$LOG"

if [ "$DPO_COUNT" -ge 20 ]; then
    echo "[$(ts)] Training DPO on $DPO_COUNT pairs (~3 hours)..." | tee -a "$LOG"

    $PY -m src.training.train_dpo_piecewise \
        --base-model outputs/merged_model_v13 \
        2>&1 | tee -a "$LOG" \
        || abort "phase1b_dpo_train" $?

    echo "[$(ts)] Phase 1B complete: DPO training done" | tee -a "$LOG"
else
    echo "[$(ts)] Phase 1B SKIPPED: only $DPO_COUNT pairs (need >= 20)" | tee -a "$LOG"
fi

# ── Phase 2: Full-Spec GRPO ─────────────────────────────────────────────
echo ""
echo "[$(ts)] ===== PHASE 2: FULL-SPEC GRPO (200 steps) =====" | tee -a "$LOG"
echo "[$(ts)] This will take ~12-16 hours" | tee -a "$LOG"
echo "[$(ts)] Disk before GRPO: $(df -h "$REPO" | tail -1)" | tee -a "$LOG"

$PY -m scripts.train_rl_fullspec \
    --max-steps 200 \
    --num-generations 4 \
    --max-completion-length 1536 \
    --save-steps 50 \
    2>&1 | tee -a "$LOG" \
    || {
        echo "[$(ts)] GRPO failed — trying fallback: 2 gens, 1024 completion length" | tee -a "$LOG"
        $PY -m scripts.train_rl_fullspec \
            --max-steps 200 \
            --num-generations 2 \
            --max-completion-length 1024 \
            --save-steps 50 \
            2>&1 | tee -a "$LOG" \
            || abort "phase2_grpo" $?
    }

echo "[$(ts)] Phase 2 complete: GRPO training done" | tee -a "$LOG"

# ── Phase 3: Flywheel (3 cycles) ────────────────────────────────────────
echo ""
echo "[$(ts)] ===== PHASE 3: FLYWHEEL (3 cycles) =====" | tee -a "$LOG"
echo "[$(ts)] Each cycle ~4-6 hours" | tee -a "$LOG"

$PY -m scripts.flywheel \
    --cycles 3 \
    --n-prompts 50 \
    --model chattla:20b \
    2>&1 | tee -a "$LOG" \
    || echo "[$(ts)] WARNING: Flywheel exited non-zero" | tee -a "$LOG"

echo "[$(ts)] Phase 3 complete" | tee -a "$LOG"

# ── Final Summary ────────────────────────────────────────────────────────
echo ""
echo "[$(ts)] ===== PIPELINE COMPLETE =====" | tee -a "$LOG"
echo "[$(ts)] Disk: $(df -h "$REPO" | tail -1)" | tee -a "$LOG"

# Print final metrics if available
if [ -f outputs/logs/flywheel_metrics.jsonl ]; then
    echo "[$(ts)] Flywheel metrics:" | tee -a "$LOG"
    tail -5 outputs/logs/flywheel_metrics.jsonl | tee -a "$LOG"
fi

echo "[$(ts)] === FULL PIPELINE END ===" | tee -a "$LOG"
