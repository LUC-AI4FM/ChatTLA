#!/usr/bin/env bash
# run_fork_a_sft.sh — launch Fork A SFT in a detached tmux session.
#
# Usage:
#   scripts/run_fork_a_sft.sh tlc
#   scripts/run_fork_a_sft.sh tlaps
#
# Trains incrementally on top of the current best merged model
# (/data/sdb/espencer2/chattla/merged_model_repair) so we keep the v14 SFT +
# repair-GRPO r1 capability and ADD the validator-segregated tlaplus/examples
# data. Checkpoints land in outputs/checkpoints_fork_a_<target>/.
set -euo pipefail

TARGET="${1:-}"
if [[ "$TARGET" != "tlc" && "$TARGET" != "tlaps" ]]; then
    echo "usage: $0 {tlc|tlaps}" >&2
    exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TRAIN_FILE="data/processed/fork_a_${TARGET}_sft.jsonl"
OUT_DIR="outputs/checkpoints_fork_a_${TARGET}"
EXPERIMENT="ChatTLA-ForkA-${TARGET^^}"
BASE_MODEL="/data/sdb/espencer2/chattla/merged_model_repair"
LOG="outputs/logs/fork_a_${TARGET}_sft.log"
SESSION="fork_a_${TARGET}"

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "[error] missing $TRAIN_FILE — run scripts/build_fork_a_corpora.py first" >&2
    exit 1
fi
if [[ ! -d "$BASE_MODEL" ]]; then
    echo "[error] missing base model dir $BASE_MODEL" >&2
    exit 1
fi

mkdir -p outputs/logs

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[error] tmux session '$SESSION' already exists; attach with:"
    echo "        tmux attach -t $SESSION"
    exit 1
fi

echo "[fork_a] target=$TARGET"
echo "[fork_a] train=$TRAIN_FILE"
echo "[fork_a] base=$BASE_MODEL"
echo "[fork_a] out=$OUT_DIR"
echo "[fork_a] log=$LOG"
echo "[fork_a] tmux session=$SESSION"

CMD=$(cat <<EOF
cd $REPO && source .venv/bin/activate && \
python3 -m src.training.train \
  --train-file $TRAIN_FILE \
  --output-dir $OUT_DIR \
  --experiment-name $EXPERIMENT \
  --base-model $BASE_MODEL \
  --epochs 1 \
  --lr 2e-5 \
  --max-length 2048 \
  --max-gpu-memory-mb 36000 \
  2>&1 | tee $LOG
EOF
)

tmux new-session -d -s "$SESSION" "$CMD"
echo ""
echo "[fork_a] launched. watch progress with:"
echo "        tail -F $LOG"
echo "        tmux attach -t $SESSION"
