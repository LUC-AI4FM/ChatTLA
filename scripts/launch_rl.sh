#!/usr/bin/env bash
# launch_rl.sh — Start the ChatTLA autonomous RL loop in a tmux session.
#
# Usage:
#   ./scripts/launch_rl.sh          # start/restart the RL loop
#   ./scripts/launch_rl.sh stop     # gracefully stop the loop
#   ./scripts/launch_rl.sh status   # show current status
#   ./scripts/launch_rl.sh logs     # tail the log file
#
# The loop runs inside a tmux session named "chattla-rl".
# GPU usage is automatically throttled during daytime hours (06:00-22:00).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SESSION_NAME="chattla-rl"
LOG_FILE="$REPO_ROOT/outputs/logs/rl_loop.log"
HISTORY_FILE="$REPO_ROOT/outputs/logs/rl_history.jsonl"

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [start|stop|status|logs|restart]"
    echo ""
    echo "Commands:"
    echo "  start    Start the RL loop in a tmux session (default)"
    echo "  stop     Gracefully stop the running loop"
    echo "  status   Show loop status and recent stats"
    echo "  logs     Tail the log file"
    echo "  restart  Stop and restart the loop"
    echo ""
}

is_running() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

do_start() {
    if is_running; then
        echo -e "${YELLOW}RL loop already running in tmux session '$SESSION_NAME'.${NC}"
        echo "Use '$0 restart' to restart, or '$0 logs' to view output."
        return 1
    fi

    # Ensure log directory exists
    mkdir -p "$REPO_ROOT/outputs/logs"
    mkdir -p "$REPO_ROOT/data/processed/rl"

    echo -e "${GREEN}Starting ChatTLA autonomous RL loop...${NC}"
    echo -e "  Session:   ${BLUE}$SESSION_NAME${NC}"
    echo -e "  Repo:      $REPO_ROOT"
    echo -e "  Log:       $LOG_FILE"
    echo -e "  History:   $HISTORY_FILE"
    echo ""

    # Create tmux session and run the loop
    tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" \
        "bash -c '
        echo \"[$(date)] ChatTLA RL Loop starting...\"
        echo \"PID: \$\$\"
        echo \"GPU status:\"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
        echo \"\"
        echo \"=== Starting RL loop ===\"
        nice -n 10 python3 scripts/rl_loop.py \
            --cycle-hours 1.5 \
            --retrain-threshold 10 \
            --allow-daytime-retrain \
            --benchmark-every 3 \
            2>&1 | tee -a \"$LOG_FILE\"
        echo \"\"
        echo \"[$(date)] RL loop exited. Press any key to close.\"
        read -n 1
        '"

    sleep 1

    if is_running; then
        echo -e "${GREEN}RL loop started successfully!${NC}"
        echo ""
        echo "Useful commands:"
        echo "  tmux attach -t $SESSION_NAME     # attach to the session"
        echo "  $0 status                         # check status"
        echo "  $0 logs                           # tail logs"
        echo "  $0 stop                           # graceful shutdown"
    else
        echo -e "${RED}Failed to start RL loop. Check $LOG_FILE${NC}"
        return 1
    fi
}

do_stop() {
    if ! is_running; then
        echo -e "${YELLOW}No RL loop running.${NC}"
        return 0
    fi

    echo -e "${YELLOW}Sending graceful shutdown signal (SIGTERM)...${NC}"
    # Send SIGTERM to the python process inside the tmux session
    tmux send-keys -t "$SESSION_NAME" C-c
    echo "Waiting for current phase to complete (this may take a few minutes)..."

    # Wait up to 5 minutes for graceful shutdown
    for i in $(seq 1 60); do
        if ! is_running; then
            echo -e "${GREEN}RL loop stopped gracefully.${NC}"
            return 0
        fi
        sleep 5
    done

    echo -e "${RED}Loop didn't stop within 5 minutes. Force killing...${NC}"
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    echo "Session killed."
}

do_status() {
    echo -e "${BLUE}=== ChatTLA RL Loop Status ===${NC}"
    echo ""

    # Check if running
    if is_running; then
        echo -e "  State:     ${GREEN}RUNNING${NC}"
        echo -e "  Session:   $SESSION_NAME"
    else
        echo -e "  State:     ${RED}STOPPED${NC}"
    fi
    echo ""

    # GPU status
    echo -e "${BLUE}GPU Status:${NC}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
    echo ""

    # History stats
    if [ -f "$HISTORY_FILE" ]; then
        total_cycles=$(wc -l < "$HISTORY_FILE")
        total_gold=$(python3 -c "
import json
with open('$HISTORY_FILE') as f:
    cycles = [json.loads(l) for l in f if l.strip()]
gold = sum(c.get('gold_count', 0) for c in cycles)
silver = sum(c.get('silver_count', 0) for c in cycles)
retrains = sum(1 for c in cycles if c.get('retrained'))
last = cycles[-1] if cycles else {}
print(f'Total cycles: {len(cycles)}')
print(f'  Gold specs:    {gold}')
print(f'  Silver specs:  {silver}')
print(f'  Retrains:      {retrains}')
if last:
    print(f'  Last cycle:    #{last.get(\"cycle_id\", \"?\")} at {last.get(\"timestamp\", \"?\")[:19]}')
    print(f'    SANY pass:   {last.get(\"sany_pass\", 0)}/{last.get(\"prompts_tried\", 0)}')
    print(f'    TLC pass:    {last.get(\"tlc_pass\", 0)}/{last.get(\"prompts_tried\", 0)}')
    if last.get('benchmark_run'):
        print(f'    Benchmark:   SANY={last.get(\"benchmark_sany_rate\",0):.0%} TLC={last.get(\"benchmark_tlc_rate\",0):.0%}')
" 2>/dev/null)
        echo -e "${BLUE}History:${NC}"
        echo "$total_gold" | sed 's/^/  /'
    else
        echo -e "  ${YELLOW}No history yet (no cycles completed).${NC}"
    fi
    echo ""

    # Data stats
    echo -e "${BLUE}Data:${NC}"
    [ -f "$REPO_ROOT/data/processed/augmented.jsonl" ] && \
        echo "  Augmented examples: $(wc -l < "$REPO_ROOT/data/processed/augmented.jsonl")" || \
        echo "  Augmented examples: 0"
    [ -f "$REPO_ROOT/data/processed/rl/dpo_pairs.jsonl" ] && \
        echo "  DPO pairs:          $(wc -l < "$REPO_ROOT/data/processed/rl/dpo_pairs.jsonl")" || \
        echo "  DPO pairs:          0"
    [ -f "$REPO_ROOT/data/processed/train.jsonl" ] && \
        echo "  Training examples:  $(wc -l < "$REPO_ROOT/data/processed/train.jsonl")" || \
        echo "  Training examples:  0"
    echo ""

    # Recent log
    if [ -f "$LOG_FILE" ]; then
        echo -e "${BLUE}Recent log (last 10 lines):${NC}"
        tail -10 "$LOG_FILE" | sed 's/^/  /'
    fi
}

do_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${YELLOW}No log file yet.${NC}"
        return 0
    fi
    echo -e "${BLUE}Tailing $LOG_FILE (Ctrl-C to stop)${NC}"
    echo ""
    tail -f "$LOG_FILE"
}

do_restart() {
    do_stop
    sleep 2
    do_start
}

# ─── Main ────────────────────────────────────────────────────────────────────
CMD="${1:-start}"

case "$CMD" in
    start)   do_start   ;;
    stop)    do_stop    ;;
    status)  do_status  ;;
    logs)    do_logs    ;;
    restart) do_restart ;;
    -h|--help) usage    ;;
    *)
        echo -e "${RED}Unknown command: $CMD${NC}"
        usage
        exit 1
        ;;
esac
