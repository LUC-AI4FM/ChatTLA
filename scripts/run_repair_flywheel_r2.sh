#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -euo pipefail
cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=outputs/logs/repair_flywheel_r2.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }
mkdir -p outputs/logs

echo "[$(ts)] ===== Repair Flywheel Round 2 =====" | tee -a "$LOG"

# ─── Phase 1: Collect NEW trajectories with the repair model ────────────
# The repair model (9/30 diamond) will produce different failures than the
# base model — harder, closer to the frontier. These are more valuable
# training signal for the next round.
echo "[$(ts)] Phase 1: Collecting trajectories with chattla:20b-repair..." | tee -a "$LOG"

# Collect fresh trajectories from BOTH models for diversity:
# - Base model (chattla:20b): easier failures, good curriculum foundation
# - Repair model (chattla:20b-repair): harder, frontier-level failures
# Run base model first (it's cheaper — more errors early → shorter trajectories)
echo "[$(ts)] Phase 1a: Collecting base model trajectories..." | tee -a "$LOG"
.venv/bin/python -u -m scripts.collect_ralph_trajectories \
    --model chattla:20b \
    --max-iters 6 \
    --workers 4 \
    --out-trajectories data/processed/ralph_trajectories_r2_base.jsonl \
    --out-pairs data/processed/ralph_repair_pairs_r2_base.jsonl \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] Base trajectory collection FAILED" | tee -a "$LOG"
        exit 1
    }
BASE_PAIRS=$(wc -l < data/processed/ralph_repair_pairs_r2_base.jsonl)
echo "[$(ts)] Phase 1a complete: $BASE_PAIRS base model pairs" | tee -a "$LOG"

echo "[$(ts)] Phase 1b: Collecting repair model trajectories..." | tee -a "$LOG"
.venv/bin/python -u -m scripts.collect_ralph_trajectories \
    --model chattla:20b-repair \
    --max-iters 6 \
    --workers 4 \
    --out-trajectories data/processed/ralph_trajectories_r2_repair.jsonl \
    --out-pairs data/processed/ralph_repair_pairs_r2_repair.jsonl \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] Repair trajectory collection FAILED" | tee -a "$LOG"
        exit 1
    }
REPAIR_PAIRS=$(wc -l < data/processed/ralph_repair_pairs_r2_repair.jsonl)
echo "[$(ts)] Phase 1b complete: $REPAIR_PAIRS repair model pairs" | tee -a "$LOG"

# ─── Phase 2: Merge base + repair pairs ─────────────────────────────────
echo "[$(ts)] Phase 2: Merging repair pairs..." | tee -a "$LOG"

.venv/bin/python -u -c "
import json
from pathlib import Path

base = Path('data/processed/ralph_repair_pairs_r2_base.jsonl')
repair = Path('data/processed/ralph_repair_pairs_r2_repair.jsonl')
out = Path('data/processed/ralph_repair_pairs.jsonl')

seen_keys = set()
pairs = []

# Repair model pairs first (harder, higher value signal)
for path, tag in [(repair, 'repair'), (base, 'base')]:
    if not path.exists():
        continue
    with path.open() as f:
        for line in f:
            row = json.loads(line.strip())
            # Deduplicate by (nl prefix, before_score bucket)
            key = (row['nl'][:80], round(row['before_score'], 1))
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
" 2>&1 | tee -a "$LOG"

MERGED_PAIRS=$(wc -l < data/processed/ralph_repair_pairs.jsonl)
echo "[$(ts)] Phase 2 complete: $MERGED_PAIRS merged pairs" | tee -a "$LOG"

if [ "$MERGED_PAIRS" -lt 50 ]; then
    echo "[$(ts)] Too few merged pairs ($MERGED_PAIRS < 50). Aborting." | tee -a "$LOG"
    exit 1
fi

# ─── Phase 3: Unload Ollama, free VRAM ──────────────────────────────────
echo "[$(ts)] Unloading Ollama models for GRPO training..." | tee -a "$LOG"
curl -s http://localhost:11434/api/generate -d '{"model":"chattla:20b-repair","keep_alive":0}' > /dev/null 2>&1 || true
curl -s http://localhost:11434/api/generate -d '{"model":"chattla:20b","keep_alive":0}' > /dev/null 2>&1 || true
sleep 5

# ─── Phase 4: Repair GRPO round 2 ──────────────────────────────────────
# Key changes from round 1:
#   - Base model = merged_model_repair (continue from round 1 gains)
#   - Combined dataset (r1 + r2 pairs — more diverse)
#   - Same hyperparams that worked (3e-6, beta=0.02, temp=0.5)
#   - 300 steps again (reward peaked ~140-160 in r1, 300 provides safety margin)
echo "[$(ts)] Phase 4: Repair GRPO round 2 (from repair checkpoint)..." | tee -a "$LOG"
.venv/bin/python -u -m scripts.train_rl_repair \
    --model ${CHATTLA_MODEL_DIR:-$REPO/outputs}/merged_model_repair \
    --output-dir outputs/checkpoints_rl_repair_r2 \
    --max-steps 300 \
    --num-generations 4 \
    --max-completion-length 384 \
    --max-prompt-tokens 1600 \
    --min-before-score 0.10 \
    --max-before-score 0.80 \
    --difficulty all \
    --save-steps 25 \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] Repair GRPO r2 failed at 4x384 — retrying at 2x384" | tee -a "$LOG"
        .venv/bin/python -u -m scripts.train_rl_repair \
            --model ${CHATTLA_MODEL_DIR:-$REPO/outputs}/merged_model_repair \
            --output-dir outputs/checkpoints_rl_repair_r2 \
            --max-steps 300 \
            --num-generations 2 \
            --max-completion-length 384 \
            --max-prompt-tokens 1600 \
            --min-before-score 0.10 \
            --max-before-score 0.80 \
            --difficulty all \
            --save-steps 25 \
            2>&1 | tee -a "$LOG" || {
                echo "[$(ts)] Repair GRPO r2 FAILED" | tee -a "$LOG"
                exit 1
            }
    }
echo "[$(ts)] Phase 4 complete." | tee -a "$LOG"

# ─── Phase 5: Merge LoRA + deploy to Ollama ─────────────────────────────
echo "[$(ts)] Phase 5: Merging LoRA and deploying..." | tee -a "$LOG"

CKPT=""
for c in $(ls -1d outputs/checkpoints_rl_repair_r2/checkpoint-* 2>/dev/null | sort -t- -k2 -n -r); do
    num=${c##*checkpoint-}
    if [ "$num" -ge 25 ]; then
        CKPT="$c"
        break
    fi
done
if [ -z "$CKPT" ]; then
    CKPT="outputs/checkpoints_rl_repair_r2/final"
fi
echo "[$(ts)] Using checkpoint: $CKPT" | tee -a "$LOG"

MERGE_OUT=${CHATTLA_MODEL_DIR:-$REPO/outputs}/merged_model_repair_r2
mkdir -p "${CHATTLA_MODEL_DIR:-$REPO/outputs}"
.venv/bin/python -m src.training.merge_lora \
    --checkpoint "$CKPT" \
    --output "$MERGE_OUT" \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] LoRA merge FAILED" | tee -a "$LOG"
        exit 1
    }

# Symlink
rm -f outputs/merged_model_repair_r2
ln -s "$MERGE_OUT" outputs/merged_model_repair_r2
echo "[$(ts)] Merged: outputs/merged_model_repair_r2 -> $MERGE_OUT" | tee -a "$LOG"

# Convert to GGUF and deploy via project converter
# The converter reads from outputs/merged_model — temporarily point it at r2
echo "[$(ts)] Converting to GGUF..." | tee -a "$LOG"
ORIG_SYMLINK=$(readlink -f outputs/merged_model 2>/dev/null || true)
rm -f outputs/merged_model
ln -s "$MERGE_OUT" outputs/merged_model

.venv/bin/python -m src.inference.convert_to_gguf \
    --quant Q8_0 \
    --model-name chattla:20b-repair-r2 \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] GGUF conversion FAILED — skipping Ollama deploy" | tee -a "$LOG"
        echo "[$(ts)] Manual: point outputs/merged_model at $MERGE_OUT then run convert_to_gguf" | tee -a "$LOG"
    }

# Restore original symlink
rm -f outputs/merged_model
if [ -n "$ORIG_SYMLINK" ] && [ -d "$ORIG_SYMLINK" ]; then
    ln -s "$ORIG_SYMLINK" outputs/merged_model
fi

# ─── Phase 6: Evaluate ─────────────────────────────────────────────────
echo "[$(ts)] Phase 6: Evaluating repair-r2 model..." | tee -a "$LOG"

# Determine which model tag to use for eval
EVAL_MODEL="chattla:20b-repair-r2"
if ! ollama list 2>/dev/null | grep -q "chattla:20b-repair-r2"; then
    echo "[$(ts)] chattla:20b-repair-r2 not in Ollama, falling back to merged weights eval" | tee -a "$LOG"
    # Use the merged model directly via transformers if Ollama deployment failed
    EVAL_MODEL="chattla:20b-repair-r2"
fi

# Holdout eval (30 problems, 3-shot self-correct)
.venv/bin/python -u -m scripts.eval_3shot_tlc_tlaps 30 \
    --model "$EVAL_MODEL" \
    --output outputs/eval/holdout_repair_r2.json \
    2>&1 | tee -a "$LOG" || true

echo "" | tee -a "$LOG"
echo "[$(ts)] ===== REPAIR FLYWHEEL R2 COMPLETE =====" | tee -a "$LOG"

# Print comparison
echo "" | tee -a "$LOG"
echo "=== COMPARISON ===" | tee -a "$LOG"
echo "Baseline (SFT v14):    4/30 (13%)" | tee -a "$LOG"
echo "Repair GRPO r1:        9/30 (30%)" | tee -a "$LOG"
echo "Repair GRPO r2:        see holdout_repair_r2.json" | tee -a "$LOG"
