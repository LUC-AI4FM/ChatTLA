#!/usr/bin/env bash
# deploy_and_launch.sh — Wait for in-progress training, deploy, and start the RL loop.
#
# Usage:
#   ./scripts/deploy_and_launch.sh          # wait for PID, deploy, start RL
#   ./scripts/deploy_and_launch.sh --skip-wait  # deploy immediately, then start RL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Activate venv
export PATH="$REPO_ROOT/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ -f .env ]] && set -a && source .env && set +a

TRAIN_PID=$(pgrep -f "src.training.train" 2>/dev/null | head -1 || true)

if [[ "$*" != *"--skip-wait"* ]] && [[ -n "$TRAIN_PID" ]]; then
    echo "[deploy] Waiting for training PID $TRAIN_PID to finish..."
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        # Show progress every 5 minutes
        CKPT=$(ls -1d outputs/checkpoints/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
        echo "[deploy] $(date +%H:%M:%S) Training still running... latest checkpoint: $(basename "$CKPT" 2>/dev/null || echo none)"
        sleep 300
    done
    echo "[deploy] Training finished!"
fi

# Step 1: Find latest checkpoint
LATEST_CKPT=$(ls -1d outputs/checkpoints/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
if [[ -z "$LATEST_CKPT" ]]; then
    echo "[deploy] ERROR: No checkpoint found in outputs/checkpoints/"
    exit 1
fi
echo "[deploy] Using checkpoint: $LATEST_CKPT"

# Step 2: Merge LoRA
echo "[deploy] Step 1/3: Merging LoRA weights..."
CUDA_VISIBLE_DEVICES=0,1 python -m src.training.merge_lora --checkpoint "$LATEST_CKPT"

# Step 3: Convert to GGUF and register with Ollama
echo "[deploy] Step 2/3: Converting to GGUF + registering with Ollama..."
python -m src.inference.convert_to_gguf --quant Q8_0 --model-name chattla:20b

# Step 4: Smoke test
echo "[deploy] Step 3/3: Smoke test..."
RESULT=$(python3 -c "
from src.inference.ollama_client import ChatTLAClient
client = ChatTLAClient(model='chattla:20b')
spec = client.generate_spec('A simple counter that counts from 0 to N.', module_name='Counter')
print(spec[:200])
from src.validators.sany_validator import validate_string
r = validate_string(spec, module_name='Counter')
print(f'SANY valid: {r.valid}')
if not r.valid:
    print(f'Errors: {r.errors[:3]}')
" 2>&1)
echo "$RESULT"

# Step 5: Start RL loop
echo "[deploy] Deployment complete. Starting RL loop..."
exec "$REPO_ROOT/scripts/launch_rl.sh" start \
    --allow-daytime-retrain \
    --retrain-threshold 25 \
    --benchmark-every 3
