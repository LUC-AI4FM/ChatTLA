#!/bin/bash
# Repair Flywheel Round 3 — recovery from R2 regression.
#
# Changes vs R2 (the only levers touched):
#   1. Harvest more raw pairs: --max-iters 6 -> 9
#   2. Relax Phase 2 dedup key: (nl[:80], round(score, 1)) -> (nl[:120], round(score, 2))
#   3. Restore score floor: --min-before-score 0.10 -> 0.02
#   4. Stop before overtraining: --max-steps 300 -> 175
#   5. Checkpoint picker: closest to step 150 (reward peak observed in R1)
#   6. Abort guard: require >=300 pairs in [0.02, 0.80] after dedup
# Everything else (lr, beta, temp, num_generations, completion_length) matches R1/R2.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -euo pipefail
cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=outputs/logs/repair_flywheel_r3.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }
mkdir -p outputs/logs

echo "[$(ts)] ===== Repair Flywheel Round 3 =====" | tee -a "$LOG"

# ─── Phase 1: Collect trajectories from both checkpoints ─────────────────
echo "[$(ts)] Phase 1a: Collecting base model trajectories (chattla:20b)..." | tee -a "$LOG"
.venv/bin/python -u -m scripts.collect_ralph_trajectories \
    --model chattla:20b \
    --max-iters 9 \
    --workers 4 \
    --out-trajectories data/processed/ralph_trajectories_r3_base.jsonl \
    --out-pairs data/processed/ralph_repair_pairs_r3_base.jsonl \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] Base trajectory collection FAILED" | tee -a "$LOG"
        exit 1
    }
BASE_PAIRS=$(wc -l < data/processed/ralph_repair_pairs_r3_base.jsonl)
echo "[$(ts)] Phase 1a complete: $BASE_PAIRS base model pairs" | tee -a "$LOG"

echo "[$(ts)] Phase 1b: Collecting repair model trajectories (chattla:20b-repair)..." | tee -a "$LOG"
.venv/bin/python -u -m scripts.collect_ralph_trajectories \
    --model chattla:20b-repair \
    --max-iters 9 \
    --workers 4 \
    --out-trajectories data/processed/ralph_trajectories_r3_repair.jsonl \
    --out-pairs data/processed/ralph_repair_pairs_r3_repair.jsonl \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] Repair trajectory collection FAILED" | tee -a "$LOG"
        exit 1
    }
REPAIR_PAIRS=$(wc -l < data/processed/ralph_repair_pairs_r3_repair.jsonl)
echo "[$(ts)] Phase 1b complete: $REPAIR_PAIRS repair model pairs" | tee -a "$LOG"

# ─── Phase 2: Merge + relaxed dedup + stratification guard ───────────────
echo "[$(ts)] Phase 2: Merging and deduping..." | tee -a "$LOG"

.venv/bin/python -u -c "
import json, sys
from pathlib import Path

base = Path('data/processed/ralph_repair_pairs_r3_base.jsonl')
repair = Path('data/processed/ralph_repair_pairs_r3_repair.jsonl')
out = Path('data/processed/ralph_repair_pairs.jsonl')

seen_keys = set()
pairs = []

# Repair model pairs first (harder, higher-value signal)
for path, tag in [(repair, 'repair'), (base, 'base')]:
    if not path.exists():
        continue
    with path.open() as f:
        for line in f:
            row = json.loads(line.strip())
            # R3 relaxed key: longer nl prefix + 2-decimal score bucket.
            key = (row['nl'][:120], round(row['before_score'], 2))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            row['repair_id'] = f'{tag}_{row[\"repair_id\"]}'
            pairs.append(row)

with out.open('w') as f:
    for p in pairs:
        f.write(json.dumps(p) + '\n')

print(f'Merged: {len(pairs)} unique pairs -> {out}')
print(f'  from repair model: {sum(1 for p in pairs if p[\"repair_id\"].startswith(\"repair_\"))}')
print(f'  from base model:   {sum(1 for p in pairs if p[\"repair_id\"].startswith(\"base_\"))}')

# Stratification diagnostic + abort guard against a recurrence of R2's starvation.
in_band = [p for p in pairs if 0.02 <= p['before_score'] <= 0.80]
easy   = sum(1 for p in in_band if p['before_score'] <  0.10)
medium = sum(1 for p in in_band if 0.10 <= p['before_score'] < 0.40)
hard   = sum(1 for p in in_band if p['before_score'] >= 0.40)
print(f'  in-band [0.02,0.80]: {len(in_band)}  (easy={easy} medium={medium} hard={hard})')
if len(in_band) < 300:
    sys.exit(f'R3 abort: only {len(in_band)} in-band pairs after dedup (need >=300)')
" 2>&1 | tee -a "$LOG"

MERGED_PAIRS=$(wc -l < data/processed/ralph_repair_pairs.jsonl)
echo "[$(ts)] Phase 2 complete: $MERGED_PAIRS merged pairs" | tee -a "$LOG"

# ─── Phase 3: Unload Ollama, free VRAM ──────────────────────────────────
echo "[$(ts)] Unloading Ollama models for GRPO training..." | tee -a "$LOG"
curl -s http://localhost:11434/api/generate -d '{"model":"chattla:20b-repair","keep_alive":0}' > /dev/null 2>&1 || true
curl -s http://localhost:11434/api/generate -d '{"model":"chattla:20b","keep_alive":0}' > /dev/null 2>&1 || true
sleep 5

# ─── Phase 4: Repair GRPO round 3 ───────────────────────────────────────
echo "[$(ts)] Phase 4: Repair GRPO round 3 (continue from R1, 175 steps)..." | tee -a "$LOG"
.venv/bin/python -u -m scripts.train_rl_repair \
    --model ${CHATTLA_MODEL_DIR:-$REPO/outputs}/merged_model_repair \
    --output-dir outputs/checkpoints_rl_repair_r3 \
    --max-steps 175 \
    --num-generations 4 \
    --max-completion-length 384 \
    --max-prompt-tokens 1600 \
    --min-before-score 0.02 \
    --max-before-score 0.80 \
    --difficulty all \
    --save-steps 25 \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] Repair GRPO r3 failed at 4x384 — retrying at 2x384" | tee -a "$LOG"
        .venv/bin/python -u -m scripts.train_rl_repair \
            --model ${CHATTLA_MODEL_DIR:-$REPO/outputs}/merged_model_repair \
            --output-dir outputs/checkpoints_rl_repair_r3 \
            --max-steps 175 \
            --num-generations 2 \
            --max-completion-length 384 \
            --max-prompt-tokens 1600 \
            --min-before-score 0.02 \
            --max-before-score 0.80 \
            --difficulty all \
            --save-steps 25 \
            2>&1 | tee -a "$LOG" || {
                echo "[$(ts)] Repair GRPO r3 FAILED" | tee -a "$LOG"
                exit 1
            }
    }
echo "[$(ts)] Phase 4 complete." | tee -a "$LOG"

# ─── Phase 5: Merge LoRA + deploy to Ollama ─────────────────────────────
echo "[$(ts)] Phase 5: Merging LoRA and deploying..." | tee -a "$LOG"

# Pick the checkpoint closest to step 150 (R1's observed reward peak window).
BEST_CKPT=""
BEST_DIST=9999
for c in $(ls -1d outputs/checkpoints_rl_repair_r3/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    num=${c##*checkpoint-}
    [ -z "$num" ] && continue
    if [ "$num" -ge 150 ]; then
        dist=$(( num - 150 ))
    else
        dist=$(( 150 - num ))
    fi
    if [ "$dist" -lt "$BEST_DIST" ]; then
        BEST_DIST=$dist
        BEST_CKPT="$c"
    fi
done
CKPT="${BEST_CKPT:-outputs/checkpoints_rl_repair_r3/final}"
echo "[$(ts)] Using checkpoint: $CKPT" | tee -a "$LOG"

MERGE_OUT=${CHATTLA_MODEL_DIR:-$REPO/outputs}/merged_model_repair_r3
mkdir -p "${CHATTLA_MODEL_DIR:-$REPO/outputs}"
.venv/bin/python -m src.training.merge_lora \
    --checkpoint "$CKPT" \
    --output "$MERGE_OUT" \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] LoRA merge FAILED" | tee -a "$LOG"
        exit 1
    }

rm -f outputs/merged_model_repair_r3
ln -s "$MERGE_OUT" outputs/merged_model_repair_r3
echo "[$(ts)] Merged: outputs/merged_model_repair_r3 -> $MERGE_OUT" | tee -a "$LOG"

echo "[$(ts)] Converting to GGUF..." | tee -a "$LOG"
ORIG_SYMLINK=$(readlink -f outputs/merged_model 2>/dev/null || true)
rm -f outputs/merged_model
ln -s "$MERGE_OUT" outputs/merged_model

.venv/bin/python -m src.inference.convert_to_gguf \
    --quant Q8_0 \
    --model-name chattla:20b-repair-r3 \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] GGUF conversion FAILED — skipping Ollama deploy" | tee -a "$LOG"
    }

rm -f outputs/merged_model
if [ -n "$ORIG_SYMLINK" ] && [ -d "$ORIG_SYMLINK" ]; then
    ln -s "$ORIG_SYMLINK" outputs/merged_model
fi

# ─── Phase 6: Evaluate ─────────────────────────────────────────────────
echo "[$(ts)] Phase 6: Evaluating repair-r3 model..." | tee -a "$LOG"
.venv/bin/python -u -m scripts.eval_3shot_tlc_tlaps 30 \
    --model chattla:20b-repair-r3 \
    --output outputs/eval/holdout_repair_r3.json \
    2>&1 | tee -a "$LOG" || true

echo "" | tee -a "$LOG"
echo "[$(ts)] ===== REPAIR FLYWHEEL R3 COMPLETE =====" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== COMPARISON ===" | tee -a "$LOG"
echo "Baseline (SFT v14):    4/30 (13%)" | tee -a "$LOG"
echo "Repair GRPO r1:        9/30 (30%)" | tee -a "$LOG"
echo "Repair GRPO r2:        6/30 (20%)" | tee -a "$LOG"
echo "Repair GRPO r3:        see holdout_repair_r3.json" | tee -a "$LOG"
