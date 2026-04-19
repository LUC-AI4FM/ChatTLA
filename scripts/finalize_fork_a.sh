#!/usr/bin/env bash
# finalize_fork_a.sh — post-training pipeline for Fork A.
#
# After scripts/run_fork_a_sft.sh {target} completes, this script:
#   1. Merges the LoRA adapter into the base (merged_model_repair)
#   2. Converts the merged BF16 to GGUF Q8_0
#   3. Registers with Ollama as chattla:20b-fork-a-{target}
#   4. Runs the 30-spec Diamond holdout eval
#
# Usage:
#   scripts/finalize_fork_a.sh tlc
#   scripts/finalize_fork_a.sh tlaps
set -euo pipefail

TARGET="${1:-}"
if [[ "$TARGET" != "tlc" && "$TARGET" != "tlaps" ]]; then
    echo "usage: $0 {tlc|tlaps}" >&2
    exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT_DIR="outputs/checkpoints_fork_a_${TARGET}"
MERGED_OUT="/data/sdb/espencer2/chattla/merged_model_fork_a_${TARGET}"
BASE_MODEL="/data/sdb/espencer2/chattla/merged_model_repair"
OLLAMA_TAG="chattla:20b-fork-a-${TARGET}"
GGUF_NAME="chattla-20b-fork-a-${TARGET}"
EVAL_OUT="outputs/eval/fork_a_${TARGET}.json"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "[error] missing $CHECKPOINT_DIR — run scripts/run_fork_a_sft.sh $TARGET first" >&2
    exit 1
fi

LATEST_CKPT=$(ls -d "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [[ -z "$LATEST_CKPT" ]]; then
    echo "[error] no checkpoint-* in $CHECKPOINT_DIR" >&2
    exit 1
fi

echo "=== Fork A finalize: $TARGET ==="
echo "checkpoint = $LATEST_CKPT"
echo "base model = $BASE_MODEL"
echo "merged out = $MERGED_OUT"
echo "ollama tag = $OLLAMA_TAG"

echo ""
echo "=== Step 1: Merge LoRA into base ==="
python -m src.training.merge_lora \
    --checkpoint "$LATEST_CKPT" \
    --base-model "$BASE_MODEL" \
    --output "$MERGED_OUT"

echo ""
echo "=== Step 2: Convert to GGUF + register with Ollama ==="
python -m src.inference.convert_to_gguf \
    --quant Q8_0 \
    --merged-model "$MERGED_OUT" \
    --gguf-name "$GGUF_NAME" \
    --model-name "$OLLAMA_TAG"

echo ""
echo "=== Step 3: Diamond holdout eval (30 specs) ==="
mkdir -p outputs/eval
python3 scripts/eval_diamond_holdout.py \
    --model "$OLLAMA_TAG" \
    --out "$EVAL_OUT"

echo ""
echo "=== Step 4: Baseline comparison ==="
BASELINE="outputs/eval/holdout_v15.json"
if [[ -f "$BASELINE" ]]; then
    python3 scripts/eval_diamond_holdout.py --compare "$BASELINE" "$EVAL_OUT"
else
    echo "[note] no baseline at $BASELINE — skipping comparison"
    echo "[note] run: python3 scripts/eval_diamond_holdout.py --model chattla:20b --out $BASELINE"
fi

echo ""
echo "=== Fork A $TARGET finalize complete ==="
echo "results: $EVAL_OUT"
