#!/bin/bash
set -euo pipefail
cd /home/espencer2/ChatTLA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=outputs/logs/pipeline_master.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] ===== GRPO 200-step run (channel fix + temp 0.7) =====" | tee -a "$LOG"

for m in chattla:20b chattla:20b-v17 chattla:20b-v16 chattla:20b-v14; do
    curl -s http://localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" > /dev/null 2>&1 || true
done
sleep 3

.venv/bin/python -u -m scripts.train_rl_fullspec \
    --model outputs/merged_model_dpo_piecewise \
    --max-steps 200 \
    --num-generations 4 \
    --max-completion-length 1536 \
    --temperature 0.7 \
    --save-steps 50 \
    2>&1 | tee -a "$LOG" || {
        echo "[$(ts)] 4 gens OOM, trying 2 gens" | tee -a "$LOG"
        .venv/bin/python -u -m scripts.train_rl_fullspec \
            --model outputs/merged_model_dpo_piecewise \
            --max-steps 200 \
            --num-generations 2 \
            --max-completion-length 1024 \
            --temperature 0.7 \
            --save-steps 50 \
            2>&1 | tee -a "$LOG" || echo "[$(ts)] GRPO FAILED" | tee -a "$LOG"
    }

echo "[$(ts)] GRPO complete. Running eval..." | tee -a "$LOG"

# Deploy and evaluate
curl -s http://localhost:11434/api/generate -d '{"model":"chattla:20b","prompt":"test","stream":false,"options":{"num_predict":1}}' > /dev/null 2>&1
sleep 5

.venv/bin/python -u -m scripts.eval_3shot_tlc_tlaps 30 \
    --model chattla:20b \
    --output outputs/eval/holdout_grpo_fullspec.json \
    2>&1 | tee -a "$LOG" || true

echo "[$(ts)] === GRPO RUN COMPLETE ===" | tee -a "$LOG"
