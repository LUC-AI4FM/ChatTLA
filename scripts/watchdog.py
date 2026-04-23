#!/usr/bin/env python3
"""
watchdog.py — Cron-based watchdog for ChatTLA RL loop.

Checks:
  1. Heartbeat: Is rl_loop.log still being written to?
  2. Process alive: Is the rl_loop.py PID still running?
  3. Metric trends: Are SANY/TLC rates declining over recent cycles?
  4. Disk space: Is there enough room for retraining?

Sends email alerts via scripts/alerting.py when problems are detected.

Install as cron job (every 15 minutes):
    crontab -e
    */15 * * * * cd /path/to/ChatTLA && .venv/bin/python scripts/watchdog.py >> outputs/logs/watchdog.log 2>&1

Or install automatically:
    python scripts/watchdog.py --install-cron
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.alerting import (
    alert_disk_critical,
    alert_heartbeat_dead,
    alert_metric_drop,
    send_email,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("watchdog")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
HEARTBEAT_TIMEOUT_MIN = 30       # alert if no log write in this many minutes
SANY_FLOOR = 0.25                # alert if SANY rate below this for N cycles
TLC_FLOOR = 0.0                  # alert if TLC rate stays at 0 for N cycles
METRIC_WINDOW = 5                # consecutive cycles to check
DISK_WARNING_GB = 30             # warn if free disk below this

_LOG_FILE = _REPO_ROOT / "outputs" / "logs" / "rl_loop.log"
_HISTORY_FILE = _REPO_ROOT / "outputs" / "logs" / "rl_history.jsonl"
_WATCHDOG_STATE = _REPO_ROOT / "outputs" / "logs" / "watchdog_state.json"


def check_heartbeat() -> bool:
    """Check if rl_loop.log was written to recently. Returns True if healthy."""
    if not _LOG_FILE.exists():
        log.warning("Log file does not exist — loop may not have started yet")
        return True  # don't alert on first boot

    mtime = _LOG_FILE.stat().st_mtime
    age_min = (datetime.datetime.now().timestamp() - mtime) / 60

    if age_min > HEARTBEAT_TIMEOUT_MIN:
        log.warning(f"Heartbeat DEAD: log file last modified {age_min:.0f} min ago")
        alert_heartbeat_dead(age_min)
        return False

    log.info(f"Heartbeat OK: log modified {age_min:.1f} min ago")
    return True


def check_process_alive() -> bool:
    """Check if rl_loop.py is running (by looking for its process)."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "rl_loop.py"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split("\n")
        pids = [p for p in pids if p.strip()]
        if pids:
            log.info(f"Process alive: PIDs {', '.join(pids)}")
            return True
        log.warning("No rl_loop.py process found")
        return False
    except Exception as e:
        log.warning(f"Could not check process: {e}")
        return True  # don't alert on pgrep failure


def check_metric_trends() -> list[str]:
    """Check recent cycles for sustained metric drops. Returns list of issues."""
    if not _HISTORY_FILE.exists():
        return []

    issues = []
    with open(_HISTORY_FILE) as f:
        cycles = [json.loads(line) for line in f if line.strip()]

    if len(cycles) < METRIC_WINDOW:
        return []

    recent = cycles[-METRIC_WINDOW:]

    # Check SANY rate
    sany_rates = []
    for c in recent:
        tried = c.get("prompts_tried", 0) or c.get("specs_generated", 1)
        sany = c.get("sany_pass", 0)
        if tried > 0:
            sany_rates.append(sany / tried)

    if sany_rates and all(r < SANY_FLOOR for r in sany_rates):
        avg = sum(sany_rates) / len(sany_rates)
        issues.append(f"SANY rate below {SANY_FLOOR:.0%} for {METRIC_WINDOW} cycles (avg {avg:.0%})")
        alert_metric_drop(
            "SANY_rate", avg, SANY_FLOOR,
            window=METRIC_WINDOW,
            cycle_id=recent[-1].get("cycle_id", 0),
        )

    # Check for zero gold streaks
    gold_counts = [c.get("gold_count", 0) for c in recent]
    if all(g == 0 for g in gold_counts):
        issues.append(f"Zero gold specs for {METRIC_WINDOW} consecutive cycles")
        alert_metric_drop(
            "gold_rate", 0.0, 0.01,
            window=METRIC_WINDOW,
            cycle_id=recent[-1].get("cycle_id", 0),
        )

    # Check for rising regressions
    regressions = [c.get("prompt_regressions", 0) for c in recent]
    if sum(regressions) > METRIC_WINDOW * 2:
        issues.append(f"High regressions: {sum(regressions)} in last {METRIC_WINDOW} cycles")

    # Check benchmark TLC trend (if available)
    bench_tlc = [c.get("benchmark_tlc_rate") for c in recent if c.get("benchmark_tlc_rate") is not None]
    if len(bench_tlc) >= 3 and all(t == 0 for t in bench_tlc[-3:]):
        issues.append("Benchmark TLC has been 0% for 3+ cycles")
        alert_metric_drop(
            "benchmark_TLC", 0.0, 0.05,
            window=3,
            cycle_id=recent[-1].get("cycle_id", 0),
        )

    return issues


def check_disk_space() -> bool:
    """Check if disk has enough free space. Returns True if OK."""
    try:
        stat = os.statvfs(str(_REPO_ROOT))
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)

        if free_gb < DISK_WARNING_GB:
            log.warning(f"Disk low: {free_gb:.1f} GB free (threshold: {DISK_WARNING_GB} GB)")
            alert_disk_critical(free_gb)
            return False

        log.info(f"Disk OK: {free_gb:.1f} GB free")
        return True
    except OSError as e:
        log.warning(f"Could not check disk: {e}")
        return True


def run_all_checks():
    """Run all watchdog checks and report."""
    log.info("=" * 40)
    log.info("Watchdog check starting")

    heartbeat_ok = check_heartbeat()
    process_ok = check_process_alive()
    disk_ok = check_disk_space()
    metric_issues = check_metric_trends()

    # If process is dead AND heartbeat is dead, send a combined alert
    if not process_ok and not heartbeat_ok:
        send_email(
            "RL LOOP DOWN — process dead + heartbeat timeout",
            f"The RL loop process is not running and the log file hasn't been updated.\n\n"
            f"This requires manual intervention.\n\n"
            f"Restart with:\n"
            f"  cd {_REPO_ROOT}\n"
            f"  bash scripts/launch_rl.sh restart\n\n"
            f"Time: {datetime.datetime.now().isoformat()}",
            category="loop_down",
        )

    if metric_issues:
        for issue in metric_issues:
            log.warning(f"Metric issue: {issue}")

    status = "HEALTHY" if (heartbeat_ok and process_ok and disk_ok and not metric_issues) else "ISSUES DETECTED"
    log.info(f"Watchdog check complete: {status}")
    log.info("=" * 40)


def install_cron():
    """Install this script as a cron job running every 15 minutes."""
    venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path(sys.executable)

    script_path = Path(__file__).resolve()
    log_path = _REPO_ROOT / "outputs" / "logs" / "watchdog.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cron_line = (
        f"*/15 * * * * cd {_REPO_ROOT} && {venv_python} {script_path} "
        f">> {log_path} 2>&1"
    )

    # Check if already installed
    try:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        existing = result.stdout if result.returncode == 0 else ""
    except Exception:
        existing = ""

    if "watchdog.py" in existing:
        print("Watchdog cron job already installed. Current crontab:")
        print(existing)
        return

    new_crontab = existing.rstrip("\n") + "\n" + cron_line + "\n"
    proc = subprocess.run(
        ["crontab", "-"],
        input=new_crontab, text=True, capture_output=True,
    )
    if proc.returncode == 0:
        print(f"Installed cron job (every 15 min):")
        print(f"  {cron_line}")
    else:
        print(f"Failed to install cron: {proc.stderr}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChatTLA RL loop watchdog")
    parser.add_argument("--install-cron", action="store_true",
                        help="Install as cron job (every 15 minutes)")
    args = parser.parse_args()

    if args.install_cron:
        install_cron()
    else:
        run_all_checks()
