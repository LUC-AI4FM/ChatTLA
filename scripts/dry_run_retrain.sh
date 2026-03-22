#!/usr/bin/env bash
# Dry run of the RL retrain pipeline — validates all steps in <10 min.
# Uses --smoke-test for training (5 steps) instead of full run.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "=== Dry run: dataset_builder ==="
python3 -m src.training.dataset_builder --sany-only --include-augmented --include-description-sft --bugfix-oversample 2 || { echo "FAIL: dataset_builder"; exit 1; }

echo ""
echo "=== Dry run: train (smoke test, 5 steps) ==="
python3 -m src.training.train --smoke-test --max-gpu-memory-mb 36000 || { echo "FAIL: train"; exit 1; }

echo ""
echo "=== Dry run: merge_lora ==="
python3 -m src.training.merge_lora || { echo "FAIL: merge_lora"; exit 1; }

echo ""
echo "=== Dry run: convert_to_gguf (Q8_0) — skip with DRY_RUN_SKIP_GGUF=1 ==="
if [ "${DRY_RUN_SKIP_GGUF:-0}" = "1" ]; then
  echo "Skipping GGUF (DRY_RUN_SKIP_GGUF=1)"
else
  python3 -m src.inference.convert_to_gguf --quant Q8_0 || { echo "FAIL: convert_to_gguf"; exit 1; }
fi

echo ""
echo "=== DRY RUN COMPLETE — all steps passed ==="
