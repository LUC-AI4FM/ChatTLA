#!/bin/bash
# deploy_diamond_model.sh — Post-training pipeline: merge LoRA → GGUF → Ollama → benchmark
# Usage: ./scripts/deploy_diamond_model.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Step 1: Merge LoRA adapter into base model ==="
python -m src.training.merge_lora
echo "  → Merged model saved to outputs/merged_model/"

echo ""
echo "=== Step 2: Convert to GGUF ==="
python -m src.inference.convert_to_gguf --quant Q8_0 --model-name chattla:20b
echo "  → GGUF created and registered with Ollama"

echo ""
echo "=== Step 3: Smoke test — generate and validate one spec ==="
python3 -c "
from src.inference.ollama_client import ChatTLAClient
from src.validators.tlc_validator import validate_string
from pathlib import Path
import re

client = ChatTLAClient(model='chattla:20b')
spec = client.generate_spec('A mutual exclusion algorithm for 3 processes.')
print('Generated spec:')
print(spec[:200] + '...')

jar = Path('src/shared/tlc/tla2tools.jar').resolve()
m = re.search(r'----\s*MODULE\s+(\w+)', spec)
mod = m.group(1) if m else 'Spec'
result = validate_string(spec, module_name=mod, jar=jar, timeout=60)
print(f'Tier: {result.tier}')
print(f'Diamond: {result.is_diamond}')
if result.tier == 'bronze':
    print(f'SANY errors: {result.sany_errors[:2]}')
if result.tier == 'silver':
    print(f'TLC violations: {result.tlc_violations[:2]}')
"

echo ""
echo "=== Step 4: Full benchmark (20 problems, 2 attempts, self-correct) ==="
python -m src.inference.benchmark \
    --model chattla:20b \
    --attempts 2 \
    --self-correct \
    --output "outputs/benchmark_results/benchmark_results_diamond_sft_$(date +%Y%m%d_%H%M%S).csv"

echo ""
echo "=== Diamond SFT deployment complete ==="
