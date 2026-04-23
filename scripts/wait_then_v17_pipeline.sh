#!/bin/bash
# wait_then_v17_pipeline.sh — continuation: stages 4-6 only.
#
# Stages 0-3 completed on prior invocations:
#   stage 0: checkpoint-401 passed gate
#   stage 1: merge OK (outputs/merged_model written)
#   stage 2: GGUF OK (chattla:20b-v17 registered in Ollama)
#   stage 3: eval OK (1/30 diamond, outputs/eval/holdout_v17.json written)
#
# This invocation runs:
#   stage 4: cleanup + ollama unload
#   stage 5: RL canary smoke (--no-vllm)
#   stage 6: RL canary full 100-step (--no-vllm)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -u
cd "$REPO"

PIPELINE_LOG=outputs/logs/v17_pipeline.log
RL_LOG=outputs/logs/rl_canary_tla_v3.log
RL_SMOKE_LOG=outputs/logs/rl_canary_tla_v3_smoke.log

PY=.venv/bin/python
export TOKENIZERS_PARALLELISM=false

ts() { date '+%Y-%m-%d %H:%M:%S'; }

abort() {
  echo "[$(ts)] *** PIPELINE ABORTED at stage: $1 (exit=$2) ***"
  exit "$2"
}

{
  echo "[$(ts)] v17 pipeline RESUMED from stage 4 (stages 0-3 already done)"

  # ---------- Stage 4: free disk + unload ollama ----------
  echo "[$(ts)] === stage 4: cleanup v17 merged HF + GGUF (already in Ollama) ==="
  rm -rf outputs/merged_model outputs/gguf
  df -h outputs/ | tail -1

  echo "[$(ts)] unloading chattla:20b-v17 from ollama VRAM..."
  curl -s http://localhost:11434/api/generate \
       -d '{"model": "chattla:20b-v17", "keep_alive": 0}' >/dev/null 2>&1
  sleep 15
  echo "[$(ts)] GPU state after ollama unload:"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

  # ---------- Stage 5: RL canary smoke (--no-vllm) ----------
  echo "[$(ts)] === stage 5: RL canary smoke (--no-vllm) ==="
  $PY -m scripts.train_canary_tla --smoke --no-vllm > "$RL_SMOKE_LOG" 2>&1
  rc=$?
  echo "[$(ts)] smoke exit code: $rc"
  if [ "$rc" -ne 0 ]; then
    tail -60 "$RL_SMOKE_LOG"
    abort "rl-smoke" $rc
  fi

  # ---------- Stage 6: RL canary full 100-step run ----------
  echo "[$(ts)] === stage 6: RL canary full 100-step (--no-vllm) ==="
  $PY -m scripts.train_canary_tla --max-steps 100 --no-vllm 2>&1 | tee "$RL_LOG"
  rc=${PIPESTATUS[0]}
  echo "[$(ts)] full RL canary exit code: $rc"

  echo "[$(ts)] === pipeline complete ==="
} >> "$PIPELINE_LOG" 2>&1
