#!/bin/bash
# ralph_watchdog.sh — checks tmux ralph is alive, logs status
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=$REPO/outputs/logs/ralph_watchdog.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }

if tmux has-session -t ralph 2>/dev/null; then
    echo "[$(ts)] ralph: ALIVE" >> "$LOG"
    # Log last line of pipeline
    LAST=$(tmux capture-pane -t ralph -p 2>/dev/null | grep -v "^$" | tail -1)
    echo "[$(ts)]   last: $LAST" >> "$LOG"
else
    echo "[$(ts)] ralph: DEAD — restarting pipeline" >> "$LOG"
    cd "$REPO"
    tmux new-session -d -s ralph -c $REPO \
        "bash -c './scripts/run_full_pipeline.sh 2>&1 | tee outputs/logs/pipeline_master.log; echo DONE; read'"
    echo "[$(ts)]   restarted ralph session" >> "$LOG"
fi

# Disk check
FREE=$(df --output=avail . | tail -1 | tr -d ' ')
FREE_GB=$((FREE / 1024 / 1024))
echo "[$(ts)]   disk: ${FREE_GB}GB free" >> "$LOG"
if [ "$FREE_GB" -lt 10 ]; then
    echo "[$(ts)]   *** WARNING: DISK LOW (${FREE_GB}GB) ***" >> "$LOG"
fi
