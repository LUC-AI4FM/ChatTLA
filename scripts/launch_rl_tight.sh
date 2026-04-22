#!/bin/bash
# launch_rl_tight.sh — Launch the tight RL loop in tmux with monitoring
#
# Usage:
#   ./scripts/launch_rl_tight.sh start      # Start new session
#   ./scripts/launch_rl_tight.sh stop       # Stop gracefully  
#   ./scripts/launch_rl_tight.sh status     # Check status
#   ./scripts/launch_rl_tight.sh logs       # Tail logs
#   ./scripts/launch_rl_tight.sh attach     # Attach to session
#   ./scripts/launch_rl_tight.sh smoke      # Quick test run

set -e
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

SESSION_NAME="rl_tight"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_FILE="${REPO_ROOT}/outputs/logs/rl_tight/rl_tight.log"

# Ensure venv exists
if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "Creating venv..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

# Load env vars
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
fi

case "${1:-start}" in
    start)
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "Session '$SESSION_NAME' already running. Use 'attach' or 'stop' first."
            exit 1
        fi
        
        echo "Starting RL tight loop in tmux session '$SESSION_NAME'..."
        mkdir -p outputs/logs/rl_tight
        
        # Create tmux session with the RL loop
        tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" \
            "CUDA_VISIBLE_DEVICES=0,1 $VENV_PYTHON -u scripts/rl_tight_loop.py --cycles 100 >> \"$LOG_FILE\" 2>&1"
        
        echo "Started. Use './scripts/launch_rl_tight.sh attach' to view."
        echo "Logs: $LOG_FILE"
        ;;
    
    smoke)
        echo "Running smoke test..."
        CUDA_VISIBLE_DEVICES=0,1 $VENV_PYTHON scripts/rl_tight_loop.py --smoke --reset
        ;;
    
    stop)
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "Sending SIGINT to gracefully stop..."
            tmux send-keys -t "$SESSION_NAME" C-c
            sleep 2
            if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
                echo "Session still running, force killing..."
                tmux kill-session -t "$SESSION_NAME"
            fi
            echo "Stopped."
        else
            echo "Session '$SESSION_NAME' not running."
        fi
        ;;
    
    status)
        echo "=== RL Tight Loop Status ==="
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "✓ Session running"
        else
            echo "✗ Session not running"
        fi
        
        # Show state
        STATE_FILE="${REPO_ROOT}/data/processed/rl_tight/state.json"
        if [[ -f "$STATE_FILE" ]]; then
            echo ""
            echo "State:"
            cat "$STATE_FILE" | python3 -m json.tool 2>/dev/null || cat "$STATE_FILE"
        fi
        
        # Show recent metrics
        HISTORY_FILE="${REPO_ROOT}/outputs/logs/rl_tight/history.jsonl"
        if [[ -f "$HISTORY_FILE" ]]; then
            echo ""
            echo "Last 3 cycles:"
            tail -3 "$HISTORY_FILE" | while read line; do
                echo "$line" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
print(f\"  Cycle {d['cycle']}: gold={d['gold']}, TLC={d['tlc_rate']:.1%}, trained={d['trained']}, {d['duration_s']/60:.1f}min\")
" 2>/dev/null || echo "  $line"
            done
        fi
        
        # GPU usage
        echo ""
        echo "GPU Status:"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"
        ;;
    
    logs)
        if [[ -f "$LOG_FILE" ]]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found at $LOG_FILE"
        fi
        ;;
    
    attach)
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            tmux attach -t "$SESSION_NAME"
        else
            echo "Session '$SESSION_NAME' not running."
        fi
        ;;
    
    metrics)
        # Show metrics summary
        HISTORY_FILE="${REPO_ROOT}/outputs/logs/rl_tight/history.jsonl"
        if [[ -f "$HISTORY_FILE" ]]; then
            $VENV_PYTHON -c "
import json
import sys

metrics = []
with open('$HISTORY_FILE') as f:
    for line in f:
        try:
            metrics.append(json.loads(line))
        except:
            pass

if not metrics:
    print('No metrics yet')
    sys.exit(0)

print(f'Total cycles: {len(metrics)}')
print(f'Total gold specs: {sum(m[\"gold\"] for m in metrics)}')
print(f'Total DPO pairs: {sum(m[\"dpo_pairs_created\"] for m in metrics)}')
print(f'Total trains: {sum(1 for m in metrics if m[\"trained\"])}')
print()

# TLC rate over time
tlc_rates = [m['tlc_rate'] for m in metrics if m['tlc_rate'] > 0]
holdout_rates = [m['holdout_tlc_rate'] for m in metrics if m['holdout_tlc_rate'] > 0]

if tlc_rates:
    print(f'TLC rate: {min(tlc_rates):.1%} - {max(tlc_rates):.1%} (avg {sum(tlc_rates)/len(tlc_rates):.1%})')
if holdout_rates:
    print(f'Holdout TLC: {min(holdout_rates):.1%} - {max(holdout_rates):.1%} (avg {sum(holdout_rates)/len(holdout_rates):.1%})')
"
        else
            echo "No metrics yet"
        fi
        ;;
    
    *)
        echo "Usage: $0 {start|stop|status|logs|attach|smoke|metrics}"
        exit 1
        ;;
esac
