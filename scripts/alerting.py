#!/usr/bin/env python3
"""
alerting.py — Email alerting for ChatTLA RL loop via SMTP.

Sends email notifications for:
  - Loop crash / heartbeat timeout
  - Per-cycle summary (on regression or periodic)
  - Retrain outcomes (success, failure, rollback)
  - Metric alerts (SANY/TLC rate drops)

Uses Gmail SMTP with app password. Set CHATTLA_SMTP_PASSWORD env var
or store in ~/.chattla_smtp_password.

Usage (standalone test):
    python scripts/alerting.py --test
"""
from __future__ import annotations

import datetime
import logging
import os
import smtplib
import socket
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

log = logging.getLogger("rl_loop.alerting")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SMTP_HOST = os.environ.get("CHATTLA_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("CHATTLA_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("CHATTLA_SMTP_USER", "")
SMTP_TO = os.environ.get("CHATTLA_SMTP_TO", SMTP_USER)
_PASSWORD_FILE = Path.home() / ".chattla_smtp_password"

# Rate limiting: don't spam more than 1 email per category per this many seconds
_RATE_LIMIT_S = 600  # 10 minutes
_last_sent: dict[str, float] = {}


def _get_password() -> Optional[str]:
    """Get SMTP password from env or file."""
    pw = os.environ.get("CHATTLA_SMTP_PASSWORD")
    if pw:
        return pw.strip()
    if _PASSWORD_FILE.exists():
        return _PASSWORD_FILE.read_text().strip()
    return None


def _rate_ok(category: str) -> bool:
    """Check rate limit for a given alert category."""
    now = datetime.datetime.now().timestamp()
    last = _last_sent.get(category, 0)
    if now - last < _RATE_LIMIT_S:
        return False
    _last_sent[category] = now
    return True


def send_email(subject: str, body: str, category: str = "general") -> bool:
    """Send an email alert. Returns True on success, False on failure.

    Silently fails (logs warning) — alerting must never crash the RL loop.
    """
    if not SMTP_USER:
        log.debug(
            "[alert] SMTP not configured (set CHATTLA_SMTP_USER); skipping."
        )
        return False

    if not _rate_ok(category):
        log.debug(f"[alert] Rate-limited: {category} (sent <{_RATE_LIMIT_S}s ago)")
        return False

    password = _get_password()
    if not password:
        log.warning(
            "[alert] No SMTP password found. Set CHATTLA_SMTP_PASSWORD env var "
            f"or create {_PASSWORD_FILE}"
        )
        return False

    hostname = socket.gethostname()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = f"[ChatTLA] {subject}"
    msg["From"] = SMTP_USER
    msg["To"] = SMTP_TO
    msg["X-ChatTLA-Category"] = category
    msg["X-ChatTLA-Host"] = hostname

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, password)
            server.sendmail(SMTP_USER, [SMTP_TO], msg.as_string())
        log.info(f"[alert] Sent email: {subject}")
        return True
    except Exception as e:
        log.warning(f"[alert] Failed to send email: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built alert types
# ─────────────────────────────────────────────────────────────────────────────

def alert_loop_crash(error: str, traceback_str: str = "", cycle_id: int = 0):
    """Alert: RL loop crashed."""
    body = (
        f"RL loop crashed at cycle {cycle_id}.\n\n"
        f"Error: {error}\n\n"
        f"Traceback:\n{traceback_str[:2000]}\n\n"
        f"Host: {socket.gethostname()}\n"
        f"Time: {datetime.datetime.now().isoformat()}\n\n"
        "The loop may have restarted automatically, or may need manual intervention.\n"
        "Check: tmux attach -t chattla-rl"
    )
    send_email(f"CRASH cycle {cycle_id}: {error[:80]}", body, category="crash")


def alert_heartbeat_dead(last_log_age_min: float):
    """Alert: Loop heartbeat missed (called by watchdog)."""
    body = (
        f"The RL loop has not written to its log in {last_log_age_min:.0f} minutes.\n\n"
        f"This likely means the loop crashed or hung.\n\n"
        f"Host: {socket.gethostname()}\n"
        f"Time: {datetime.datetime.now().isoformat()}\n\n"
        "Check:\n"
        "  tmux attach -t chattla-rl\n"
        "  tail -20 outputs/logs/rl_loop.log\n"
        "  bash scripts/launch_rl.sh status"
    )
    send_email(
        f"HEARTBEAT DEAD — no log activity for {last_log_age_min:.0f}min",
        body,
        category="heartbeat",
    )


def alert_cycle_summary(
    cycle_id: int,
    sany_rate: float,
    tlc_rate: float,
    gold: int,
    silver: int,
    bronze: int,
    retrained: bool,
    benchmark_tlc: Optional[float] = None,
    regressions: int = 0,
    duration_min: float = 0,
    disk_free_gb: float = 0,
    accumulated: int = 0,
):
    """Send cycle summary. Only sends if there's something notable."""
    notable = []
    if regressions > 0:
        notable.append(f"{regressions} prompt regressions")
    if sany_rate < 0.30:
        notable.append(f"low SANY rate ({sany_rate:.0%})")
    if tlc_rate == 0 and gold == 0:
        notable.append("zero gold this cycle")
    if retrained:
        notable.append("RETRAINED this cycle")
    if disk_free_gb < 30:
        notable.append(f"low disk ({disk_free_gb:.0f}GB)")

    if not notable:
        return  # nothing worth emailing about

    subject = f"C{cycle_id}: {', '.join(notable[:3])}"

    lines = [
        f"Cycle {cycle_id} completed in {duration_min:.1f} min\n",
        f"  SANY rate:  {sany_rate:.0%}",
        f"  TLC rate:   {tlc_rate:.0%}",
        f"  Gold/Silver/Bronze: {gold}/{silver}/{bronze}",
        f"  Retrained:  {retrained}",
        f"  Accumulated SFT: {accumulated}",
    ]
    if benchmark_tlc is not None:
        lines.append(f"  Benchmark TLC: {benchmark_tlc:.0%}")
    if regressions > 0:
        lines.append(f"  Prompt regressions: {regressions}")
    if disk_free_gb > 0:
        lines.append(f"  Disk free: {disk_free_gb:.0f} GB")
    lines.append(f"\nHost: {socket.gethostname()}")
    lines.append(f"Time: {datetime.datetime.now().isoformat()}")

    send_email(subject, "\n".join(lines), category="cycle_summary")


def alert_retrain_outcome(
    cycle_id: int,
    outcome: str,
    mode: str = "",
    tlc_before: float = 0,
    tlc_after: float = 0,
    rolled_back: bool = False,
):
    """Alert on retrain result — especially regressions and rollbacks."""
    if outcome == "ok" and not rolled_back and tlc_after >= tlc_before:
        # Success, no regression — only email if significant improvement
        if tlc_after - tlc_before < 0.05:
            return
        subject = f"C{cycle_id} RETRAIN OK: TLC {tlc_before:.0%} -> {tlc_after:.0%}"
    elif rolled_back:
        subject = f"C{cycle_id} RETRAIN ROLLED BACK: TLC {tlc_before:.0%} -> {tlc_after:.0%}"
    else:
        subject = f"C{cycle_id} RETRAIN {outcome.upper()}: {mode}"

    body = (
        f"Retrain outcome for cycle {cycle_id}:\n\n"
        f"  Mode: {mode}\n"
        f"  Outcome: {outcome}\n"
        f"  TLC before: {tlc_before:.0%}\n"
        f"  TLC after:  {tlc_after:.0%}\n"
        f"  Rolled back: {rolled_back}\n\n"
        f"Host: {socket.gethostname()}\n"
        f"Time: {datetime.datetime.now().isoformat()}"
    )
    send_email(subject, body, category="retrain")


def alert_metric_drop(
    metric_name: str,
    current: float,
    threshold: float,
    window: int = 3,
    cycle_id: int = 0,
):
    """Alert when a metric drops below threshold for N consecutive cycles."""
    body = (
        f"{metric_name} has been below {threshold:.0%} for {window} consecutive cycles.\n\n"
        f"  Current value: {current:.0%}\n"
        f"  Threshold: {threshold:.0%}\n"
        f"  Cycle: {cycle_id}\n\n"
        "This may indicate model degradation. Consider:\n"
        "  - Checking recent prompt regressions\n"
        "  - Reviewing training data quality\n"
        "  - Rolling back to a previous GGUF backup\n\n"
        f"Host: {socket.gethostname()}\n"
        f"Time: {datetime.datetime.now().isoformat()}"
    )
    send_email(
        f"C{cycle_id} METRIC DROP: {metric_name} at {current:.0%} (< {threshold:.0%})",
        body,
        category=f"metric_{metric_name}",
    )


def alert_disk_critical(free_gb: float, cycle_id: int = 0):
    """Alert: disk critically low, retrain will fail."""
    body = (
        f"Disk space critically low: {free_gb:.1f} GB free.\n\n"
        f"Retrain requires ~25 GB free. The loop will skip retraining until space is freed.\n\n"
        "Largest consumers (typical):\n"
        "  outputs/merged_model/  ~39 GB\n"
        "  outputs/gguf/          ~20 GB\n"
        "  outputs/gguf_backup/   ~20 GB\n\n"
        "Consider cleaning old merged_model snapshots or unused checkpoints.\n\n"
        f"Host: {socket.gethostname()}\n"
        f"Time: {datetime.datetime.now().isoformat()}"
    )
    send_email(f"DISK CRITICAL: {free_gb:.0f}GB free", body, category="disk")


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Send a test email")
    args = parser.parse_args()

    if args.test:
        ok = send_email(
            "Test alert — pipeline monitoring active",
            f"This is a test email from ChatTLA alerting.\n\n"
            f"Host: {socket.gethostname()}\n"
            f"Time: {datetime.datetime.now().isoformat()}\n\n"
            "If you received this, SMTP alerting is working correctly.",
            category="test",
        )
        if ok:
            print("Test email sent successfully!")
        else:
            print("Failed to send test email. Check password setup:")
            print(f"  Option 1: export CHATTLA_SMTP_PASSWORD='your-app-password'")
            print(f"  Option 2: echo 'your-app-password' > {_PASSWORD_FILE}")
            print()
            print("For Gmail, generate an App Password at:")
            print("  https://myaccount.google.com/apppasswords")
    else:
        parser.print_help()
