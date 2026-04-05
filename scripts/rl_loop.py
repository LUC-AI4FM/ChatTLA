#!/usr/bin/env python3
"""
rl_loop.py — Autonomous, continuously-looping RL pipeline for ChatTLA.

Generates TLA+ specs, validates them through SANY/TLC with granular
line-by-line feedback, builds DPO-style training pairs (chosen vs rejected),
retrains the model, deploys, evaluates, and repeats.

Designed to run unattended in tmux for days/weeks.

Schedule
--------
- Nighttime (22:00–06:00): full speed, both GPUs available
- Daytime (06:00–22:00): throttled, single GPU, longer sleeps

Each cycle runs phases back-to-back by default (`--cycle-hours 0`). Use `--cycle-hours 1.5`
to pad with sleep so cycles are spaced (e.g. shared GPU etiquette).

Phase 1 can process multiple prompts in parallel (``--phase1-workers`` or ``CHATTLA_PHASE1_WORKERS``).
Full benchmark runs every ``--benchmark-every N`` cycles (default 3); other cycles use quick eval only.

Usage
-----
    # Launched by scripts/launch_rl.sh (recommended)
    python -m scripts.rl_loop

    # Or directly:
    python scripts/rl_loop.py --cycle-hours 1.5
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

RetrainOutcome = Literal["ok", "skipped_min_train", "failed"]


def _check_training_deps() -> None:
    """Fail fast with clear message if training deps (datasets, etc.) are missing."""
    missing = []
    try:
        import datasets  # noqa: F401
    except ImportError:
        missing.append("datasets")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    if missing:
        print(
            f"\nERROR: Missing training dependencies: {', '.join(missing)}\n"
            "Install with:  pip install -r requirements.txt\n"
            "Or minimal:   pip install datasets transformers trl peft\n",
            file=sys.stderr,
        )
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_AUGMENTED_JSONL = _REPO_ROOT / "data" / "processed" / "augmented.jsonl"
_RL_DATA_DIR     = _REPO_ROOT / "data" / "processed" / "rl"
_DPO_JSONL       = _RL_DATA_DIR / "dpo_pairs.jsonl"
_RL_STATE_FILE   = _RL_DATA_DIR / "state.json"
_RL_LOG_DIR      = _REPO_ROOT / "outputs" / "logs"
_TLC_ERRORS_JSONL = _RL_LOG_DIR / "tlc_errors.jsonl"
_RL_HISTORY      = _RL_LOG_DIR / "rl_history.jsonl"
_BENCHMARK_JSON  = _REPO_ROOT / "data" / "benchmarks" / "benchmark_suite.json"
_BENCHMARK_TO_MODULE = _REPO_ROOT / "data" / "benchmarks" / "benchmark_to_module.json"
_TLA_DESCRIPTIONS_JSON = _REPO_ROOT / "data" / "derived" / "tla_descriptions.json"
_TRAIN_JSONL     = _REPO_ROOT / "data" / "processed" / "train.jsonl"
_EVAL_JSONL      = _REPO_ROOT / "data" / "processed" / "eval.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CYCLE_HOURS        = 0.0          # 0 = no sleep between cycles; >0 pads to target hours
RETRAIN_THRESHOLD  = 10           # retrain after 10 new examples (was 25; train more, generate less)
NIGHTTIME_START    = 22           # 10 PM
NIGHTTIME_END      = 6            # 6 AM
GPU_VRAM_CAP_DAY   = 0.75         # 75% VRAM cap during daytime (leave 25%)
GPU_VRAM_CAP_NIGHT = 0.90         # 90% VRAM cap at night
MAX_PROMPTS_DAY    = 10           # fewer prompts during daytime (was 25; spend more time training)
MAX_PROMPTS_NIGHT  = 15           # lighter at night too (was 40; quality over quantity)
SFT_EPOCHS         = 3            # 15 caused catastrophic forgetting on 234 examples (2026-04-05); keep low
BENCHMARK_EVERY_N  = 3            # run full benchmark every N cycles
TEMPERATURE_BASE   = 0.3
TEMPERATURE_RANGE  = (0.1, 0.6)   # diversity range for multi-attempt
MAX_REPAIR_ROUNDS  = 1             # RLVR: verifier-guided repair attempts per silver spec
MAX_SANY_REPAIR_ROUNDS = 2        # SANY: error-guided repair attempts for bronze specs
DAY_PROMPT_COOLDOWN_S = 3.0       # pause between prompts during daytime
NIGHT_PROMPT_COOLDOWN_S = 0.5     # lighter pause at night
QUICK_EVAL_LIMIT = 12             # mini-eval every cycle (trend signal; full suite = ground truth)
QUICK_EVAL_ATTEMPTS = 2

# Hugging Face Hub (after successful merge + GGUF). Requires HF_TOKEN in env.
PUBLISH_HF_DEFAULT = True
_HF_REPO = "EricSpencer00/chattla-20b"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
_RL_LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("rl_loop")
log.setLevel(logging.INFO)

_fh = logging.FileHandler(_RL_LOG_DIR / "rl_loop.log")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(_fh)

_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_sh)


# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ─────────────────────────────────────────────────────────────────────────────
_SHUTDOWN = False

def _signal_handler(signum, frame):
    global _SHUTDOWN
    log.info(f"Received signal {signum}. Will shut down after current phase completes.")
    _SHUTDOWN = True

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SpecResult:
    prompt_id: str
    prompt_text: str
    spec: str
    tier: str                       # gold / silver / bronze
    sany_pass: bool
    tlc_pass: bool
    tlc_violations: list[str] = field(default_factory=list)
    tlc_raw_output: str = ""
    fixes_applied: list[str] = field(default_factory=list)
    structural_score: float = 0.0
    attempts: int = 1
    temperature: float = 0.3
    tlc_timeout: bool = False
    # RLVR/SANY repair: set when this result was produced by verifier-guided repair
    repair_from_spec: str = ""
    repair_from_violations: list[str] = field(default_factory=list)
    repair_type: str = ""            # "sany" or "tlc" — which verifier drove the repair
    error_class: str = ""            # classified error: syntax|invariant_violation|deadlock|unbounded_state|undefined_op|timeout|other
    critique_applied: bool = False   # whether self-critique was applied before validation


@dataclass
class CycleStats:
    cycle_id: int = 0
    timestamp: str = ""
    is_nighttime: bool = False
    prompts_tried: int = 0
    specs_generated: int = 0
    sany_pass: int = 0
    tlc_pass: int = 0
    gold_count: int = 0
    silver_count: int = 0
    bronze_count: int = 0
    new_train_examples: int = 0
    new_dpo_pairs: int = 0
    retrained: bool = False
    deployed: bool = False
    retrain_skipped_min_data: bool = False
    retrain_deferred_to_night: bool = False
    retrain_deferred_vram: bool = False
    benchmark_run: bool = False
    benchmark_full_suite: bool = False
    benchmark_sany_rate: float = 0.0
    benchmark_tlc_rate: float = 0.0
    # Error taxonomy — counts by category
    errors_syntax: int = 0
    errors_invariant_violation: int = 0
    errors_deadlock: int = 0
    errors_unbounded_state: int = 0
    errors_undefined_op: int = 0
    errors_timeout: int = 0
    errors_other: int = 0
    # Per-prompt regression tracking across retrains
    prompt_regressions: int = 0
    prompt_improvements: int = 0
    # Repair stats
    sany_repairs_attempted: int = 0
    sany_repairs_succeeded: int = 0
    critiques_applied: int = 0
    cycle_duration_s: float = 0.0
    error: str = ""
    full_traceback: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Self-Healing
# ─────────────────────────────────────────────────────────────────────────────
def _self_heal(error_msg: str, tb: str):
    """
    Experimental: Report loop crashes to GitHub Copilot / Subagent for autonomous fix.
    Requires external supervisor or periodic log scraping to trigger 'apply'.
    """
    heal_log = _REPO_ROOT / "outputs" / "logs" / "self_healing.jsonl"
    heal_log.parent.mkdir(parents=True, exist_ok=True)
    
    # We strip common noise to make the prompt cleaner
    clean_tb = "\n".join([line for line in tb.splitlines() if "site-packages" not in line])
    
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "error": error_msg,
        "traceback": clean_tb,
        "status": "pending_analysis"
    }
    
    with open(heal_log, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    log.info(f"[self-heal] Logged loop crash for autonomous repair: {error_msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def is_nighttime() -> bool:
    h = datetime.datetime.now().hour
    return h >= NIGHTTIME_START or h < NIGHTTIME_END


def gpu_memory_free_mb(device: int = 1) -> int:
    """Return free VRAM in MB for the given GPU device."""
    try:
        import torch
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return int(free_bytes // (1024 * 1024))
    except Exception:
        return 40000  # assume 40GB if can't detect


def total_gpu_memory_free_mb() -> int:
    """Return total free VRAM in MB across GPUs 0 and 1 (used for retrain)."""
    try:
        import torch
        total = 0
        for d in (0, 1):
            if d < torch.cuda.device_count():
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(d)
                    total += int(free_bytes // (1024 * 1024))
                except Exception:
                    # If we can't get info for one GPU, assume ~40GB free
                    total += 40000
        return total if total > 0 else 40000
    except Exception:
        log.warning("[memory] Could not detect GPU VRAM; assuming 40GB available")
        return 40000


def max_length_for_vram(free_mb: int) -> int:
    """Choose max_length based on available VRAM to avoid OOM."""
    if free_mb < 10_000:
        return 512    # critical (~10 GiB free, ~38 GiB in use)
    if free_mb < 15_000:
        return 768    # very tight (~15 GiB free)
    if free_mb < 22_000:
        return 1024   # very tight (~24 GiB reported in use on shared 48 GiB cards)
    if free_mb < 35_000:
        return 1536   # moderate (~1 GPU shared)
    if free_mb < 55_000:
        return 2048   # ~1 GPU free or 2 heavily shared
    if free_mb < 80_000:
        return 3072   # moderate
    return 4096       # full 2x48GB


def log_history(stats: CycleStats):
    """Append cycle stats to rl_history.jsonl."""
    _RL_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(_RL_HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(stats), ensure_ascii=False) + "\n")


def load_accumulated_new() -> int:
    """Load persisted accumulated_new from state file (survives restarts)."""
    if _RL_STATE_FILE.exists():
        try:
            with open(_RL_STATE_FILE) as f:
                data = json.load(f)
            return int(data.get("accumulated_new", 0))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return 0


def load_gold_prompt_ids() -> set[str]:
    """Load set of prompt IDs that have ever produced gold results."""
    if _RL_STATE_FILE.exists():
        try:
            with open(_RL_STATE_FILE) as f:
                data = json.load(f)
            ids = data.get("gold_prompt_ids", [])
            return set(ids) if ids else set()
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return set()


def load_gold_prompt_ids_from_benchmarks() -> set[str]:
    """Load gold prompt IDs from benchmark CSV results (bootstraps from past runs)."""
    import glob as _glob
    gold_ids = set()
    
    # Check both old location and new location
    for pattern in [
        str(_REPO_ROOT / "outputs" / "benchmark_results_*_full_*.csv"),
        str(_REPO_ROOT / "outputs" / "benchmark_results" / "benchmark_results_*_full_*.csv"),
    ]:
        for path in _glob.glob(pattern):
            try:
                with open(path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Gold = passes both SANY and TLC
                        sany = str(row.get("sany_pass", "")).strip() in ("1", "True", "true", "yes")
                        tlc = str(row.get("tlc_pass", "")).strip() in ("1", "True", "true", "yes")
                        if sany and tlc:
                            bid = row.get("benchmark_id") or row.get("prompt_id")
                            if bid:
                                gold_ids.add(str(bid).strip())
            except (OSError, csv.Error):
                pass
    
    return gold_ids


def _atomic_write_json(path: Path, data: object) -> None:
    """Write JSON atomically: write to temp file, then rename.

    Prevents corruption if the process is killed mid-write.
    """
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=path.stem + "_",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)          # atomic on POSIX
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_accumulated_new(accumulated_new: int, gold_prompt_ids: set[str] | None = None) -> None:
    """Persist accumulated_new and gold_prompt_ids so they survive restarts."""
    _RL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing gold_prompt_ids if not provided
    if gold_prompt_ids is None:
        gold_prompt_ids = load_gold_prompt_ids()

    _atomic_write_json(_RL_STATE_FILE, {
        "accumulated_new": accumulated_new,
        "gold_prompt_ids": sorted(gold_prompt_ids),
    })


# ── Per-prompt tier tracking (regression detection across retrains) ──────────
_PROMPT_TIER_FILE = _RL_DATA_DIR / "prompt_tiers.json"


def load_prompt_tiers() -> dict[str, str]:
    """Load previous cycle's prompt→tier mapping for regression detection."""
    if _PROMPT_TIER_FILE.exists():
        try:
            with open(_PROMPT_TIER_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_prompt_tiers(tiers: dict[str, str]) -> None:
    """Persist per-prompt tier mapping for next cycle."""
    _RL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(_PROMPT_TIER_FILE, tiers)


def compute_prompt_deltas(
    results: list["SpecResult"], previous_tiers: dict[str, str]
) -> tuple[int, int, list[str]]:
    """
    Compare current per-prompt best tiers with previous cycle.

    Returns (regressions, improvements, regression_details).
    """
    tier_rank = {"gold": 3, "silver": 2, "bronze": 1}
    regressions = 0
    improvements = 0
    details: list[str] = []

    # Get current best tier per prompt
    current: dict[str, str] = {}
    for r in results:
        prev_best = current.get(r.prompt_id)
        if prev_best is None or tier_rank.get(r.tier, 0) > tier_rank.get(prev_best, 0):
            current[r.prompt_id] = r.tier

    for pid, cur_tier in current.items():
        prev_tier = previous_tiers.get(pid)
        if prev_tier is None:
            continue  # first time seeing this prompt
        cur_rank = tier_rank.get(cur_tier, 0)
        prev_rank = tier_rank.get(prev_tier, 0)
        if cur_rank < prev_rank:
            regressions += 1
            details.append(f"{pid}: {prev_tier}→{cur_tier}")
        elif cur_rank > prev_rank:
            improvements += 1

    return regressions, improvements, details


# ── Gold spec cache (cross-cycle training pair mining) ───────────────────────
_GOLD_SPEC_CACHE = _RL_DATA_DIR / "gold_spec_cache.jsonl"


def load_gold_spec_cache() -> dict[str, str]:
    """Load prompt_id → gold spec mapping from cache (for cross-cycle bugfix pairs)."""
    cache: dict[str, str] = {}
    if _GOLD_SPEC_CACHE.exists():
        try:
            with open(_GOLD_SPEC_CACHE, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            cache[entry["prompt_id"]] = entry["spec"]
                        except (json.JSONDecodeError, KeyError):
                            pass
        except OSError:
            pass
    return cache


def update_gold_spec_cache(results: list["SpecResult"]) -> None:
    """Append new gold specs to the cache file. One entry per prompt_id (latest wins)."""
    existing = load_gold_spec_cache()
    new_golds = {}
    for r in results:
        if r.tier == "gold" and r.prompt_id not in existing:
            new_golds[r.prompt_id] = r.spec

    if new_golds:
        _GOLD_SPEC_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(_GOLD_SPEC_CACHE, "a", encoding="utf-8") as f:
            for pid, spec in new_golds.items():
                f.write(json.dumps({"prompt_id": pid, "spec": spec}, ensure_ascii=False) + "\n")
        log.info(f"[gold_cache] Added {len(new_golds)} new gold specs to cross-cycle cache")


def add_gold_prompts(results: list, gold_prompt_ids: set[str]) -> set[str]:
    """Extract prompt IDs that produced gold tier results and add to the set."""
    updated = gold_prompt_ids.copy()
    for r in results:
        if hasattr(r, 'tier') and r.tier == "gold":
            updated.add(r.prompt_id)
    return updated


_DIAG_JSONL = _RL_LOG_DIR / "diagnostics.jsonl"

# Stall detection state (in-process, across cycles)
_stall_state: dict = {
    "zero_sft_streak": 0,
    "retrain_fail_streak": 0,
    "last_error": "",
    "error_repeat_count": 0,
}


def _write_diag(kind: str, details: dict) -> None:
    """Append a structured diagnostic entry to diagnostics.jsonl."""
    _DIAG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "kind": kind,
        **details,
    }
    with open(_DIAG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.warning(f"[diag:{kind}] {details.get('summary', '')}")


def diagnose_and_fix(stats: CycleStats, accumulated_new: int) -> int:
    """
    After each cycle: detect failure/stall patterns and apply safe corrections.
    Returns potentially-adjusted accumulated_new.
    """
    global _stall_state

    # ── Error repeat detection ───────────────────────────────────────────────
    if stats.error:
        if stats.error == _stall_state["last_error"]:
            _stall_state["error_repeat_count"] += 1
        else:
            _stall_state["last_error"] = stats.error
            _stall_state["error_repeat_count"] = 1

        if _stall_state["error_repeat_count"] >= 3:
            _write_diag("repeated_error", {
                "summary": f"Same error repeated {_stall_state['error_repeat_count']}x: {stats.error[:200]}",
                "error": stats.error,
                "count": _stall_state["error_repeat_count"],
                "action": "none — manual investigation needed",
            })
    else:
        _stall_state["last_error"] = ""
        _stall_state["error_repeat_count"] = 0

    # ── Retrain failure streak ───────────────────────────────────────────────
    # Only count real pipeline failures (train/merge/GGUF). Do not count:
    # - MIN_TRAIN_EXAMPLES gate (intentional; accumulate more data)
    # - deferred daytime retrain (threshold met but waiting for night window)
    threshold_met = accumulated_new >= RETRAIN_THRESHOLD
    deferred = getattr(stats, "retrain_deferred_to_night", False)
    skipped_min = getattr(stats, "retrain_skipped_min_data", False)
    if threshold_met and not deferred and not stats.retrained:
        if skipped_min:
            _stall_state["retrain_fail_streak"] = 0
        else:
            _stall_state["retrain_fail_streak"] += 1
            if _stall_state["retrain_fail_streak"] >= 2:
                _write_diag("retrain_fail_streak", {
                    "summary": f"Retrain failed {_stall_state['retrain_fail_streak']} times in a row",
                    "accumulated_new": accumulated_new,
                    "action": "reset accumulated_new to 0 to stop hammering retrain",
                })
                log.warning("[diag] Resetting accumulated_new to 0 after repeated retrain failures")
                accumulated_new = 0
                _stall_state["retrain_fail_streak"] = 0
    else:
        _stall_state["retrain_fail_streak"] = 0

    # ── SFT accumulation stall ───────────────────────────────────────────────
    if stats.new_train_examples == 0:
        _stall_state["zero_sft_streak"] += 1
    else:
        _stall_state["zero_sft_streak"] = 0

    if _stall_state["zero_sft_streak"] >= 4:
        # Check if augmented.jsonl has grown suspiciously saturated
        n_aug = 0
        if _AUGMENTED_JSONL.exists():
            with open(_AUGMENTED_JSONL) as f:
                n_aug = sum(1 for l in f if l.strip())

        _write_diag("sft_stall", {
            "summary": f"0 new SFT for {_stall_state['zero_sft_streak']} consecutive cycles",
            "augmented_rows": n_aug,
            "gold_this_cycle": stats.gold_count,
            "action": "investigate dedup or prompt diversity",
        })
        _stall_state["zero_sft_streak"] = 0  # reset so we don't spam

    # ── Gold rate cliff ──────────────────────────────────────────────────────
    if stats and (stats.specs_generated or 0) >= 10:
        gold_rate = (stats.gold_count or 0) / (stats.specs_generated or 1)
        if gold_rate == 0.0:
            _write_diag("zero_gold_cycle", {
                "summary": f"Zero gold specs in cycle {stats.cycle_id} ({stats.specs_generated} specs tried)",
                "sany_pass": stats.sany_pass,
                "bronze_count": stats.bronze_count,
                "action": "model may have regressed; check recent retrain",
            })

    # ── Benchmark TLC regression ─────────────────────────────────────────────
    if stats.benchmark_run and stats.benchmark_tlc_rate == 0.0 and stats.benchmark_sany_rate > 0:
        _write_diag("tlc_regression", {
            "summary": f"Benchmark TLC=0% but SANY={stats.benchmark_sany_rate:.0%} in cycle {stats.cycle_id}",
            "retrained_this_cycle": stats.retrained,
            "action": "retrain may have hurt TLC; consider reverting or skipping next retrain",
        })

    return accumulated_new


def log_tlc_error(prompt_id: str, prompt_text: str, tier: str, violations: list[str], raw_output: str, spec_preview: str) -> None:
    """Log TLC failures for failure analysis and clustering."""
    _TLC_ERRORS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt_id": prompt_id,
        "prompt_text": prompt_text[:500],
        "tier": tier,
        "violations": violations[:10],
        "raw_snippet": raw_output[:1500] if raw_output else "",
        "spec_preview": spec_preview[:800] if spec_preview else "",
    }
    with open(_TLC_ERRORS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt bank — expanded and difficulty-graded
# ─────────────────────────────────────────────────────────────────────────────
_EXTRA_PROMPTS = [
    # Difficulty 1 — trivial state machines
    {"id": "RL001", "prompt": "A boolean flag that can be toggled between TRUE and FALSE.", "difficulty": 1, "module_hint": "BoolFlag"},
    {"id": "RL002", "prompt": "A counter that increments from 0 to MAX and then stops (use CONSTANTS MAX, set MAX=5 in TLC config). TypeOK must constrain the counter to 0..MAX.", "difficulty": 1, "module_hint": "SimpleCounter"},
    {"id": "RL003", "prompt": "A two-state light switch: ON and OFF.", "difficulty": 1, "module_hint": "LightSwitch"},
    {"id": "RL004", "prompt": "A variable that holds a natural number bounded by MAX (use CONSTANTS MAX, set MAX=8 in TLC config) and can be doubled (if 2*x <= MAX) or reset to 1.", "difficulty": 1, "module_hint": "DoublerReset"},
    # Difficulty 2 — simple protocols
    {"id": "RL005", "prompt": "A ticket dispenser bounded by MAX tickets (use CONSTANTS MAX, set MAX=5 in TLC config): customers take increasing ticket numbers up to MAX, and a counter shows the currently-served number. The system halts gracefully when all tickets are dispensed.", "difficulty": 2, "module_hint": "TicketDispenser"},
    {"id": "RL006", "prompt": "A simple FIFO queue with enqueue and dequeue operations. The queue has bounded capacity N.", "difficulty": 2, "module_hint": "BoundedFIFO"},
    {"id": "RL007", "prompt": "A readers-writers lock with at most N readers (use CONSTANTS N, set N=3 in TLC config). Multiple readers can hold the lock simultaneously but writers are exclusive. TypeOK constrains readers count to 0..N.", "difficulty": 2, "module_hint": "ReadersWriters"},
    {"id": "RL008", "prompt": "A circular buffer of size N with a read pointer and write pointer.", "difficulty": 2, "module_hint": "CircularBuffer"},
    {"id": "RL009", "prompt": "A simple state machine for an ATM: Idle, CardInserted, PINEntered, Dispensing, Done.", "difficulty": 2, "module_hint": "ATM"},
    {"id": "RL010", "prompt": "A parking lot with N spaces. Cars can enter if spaces available and leave when parked.", "difficulty": 2, "module_hint": "ParkingLot"},
    # Difficulty 3 — moderate distributed systems
    {"id": "RL011", "prompt": "A distributed lock service where N nodes can request, acquire, and release a lock. At most one node holds the lock at any time.", "difficulty": 3, "module_hint": "DistLock"},
    {"id": "RL012", "prompt": "A gossip protocol where N nodes propagate a rumor. Each round a node tells a random neighbor. Eventually all nodes know.", "difficulty": 3, "module_hint": "GossipRumor"},
    {"id": "RL013", "prompt": "A producer-consumer system with a bounded buffer. Multiple producers and one consumer. Buffer must never overflow or underflow.", "difficulty": 3, "module_hint": "MultiProducerConsumer"},
    {"id": "RL014", "prompt": "A simple commit protocol: a coordinator asks N participants to vote (yes/no). If all vote yes, coordinator commits; otherwise aborts.", "difficulty": 3, "module_hint": "SimpleCommit"},
    {"id": "RL015", "prompt": "A barrier synchronization primitive where N processes must all reach the barrier before any can proceed.", "difficulty": 3, "module_hint": "Barrier"},
    {"id": "RL016", "prompt": "A lease-based mutual exclusion protocol. A node acquires a time-limited lease. Other nodes wait until the lease expires.", "difficulty": 3, "module_hint": "LeaseProtocol"},
    # Difficulty 4 — advanced
    {"id": "RL017", "prompt": "A replicated state machine with a primary and N-1 backups. The primary sequences operations and replicates to backups before acknowledging.", "difficulty": 4, "module_hint": "PrimaryBackup"},
    {"id": "RL018", "prompt": "A chain replication protocol where writes go through a chain of N nodes from head to tail, and reads are served by the tail.", "difficulty": 4, "module_hint": "ChainReplication"},
    {"id": "RL019", "prompt": "A simple Paxos-like consensus algorithm for a single value: proposers, acceptors, and learners with majority quorums.", "difficulty": 4, "module_hint": "SimplePaxos"},
    {"id": "RL020", "prompt": "A two-phase locking protocol for transactions that ensures serializability. Transactions acquire locks before reading/writing and release all locks at commit.", "difficulty": 4, "module_hint": "TwoPhaseLock"},
    # Difficulty 5 — hard
    {"id": "RL021", "prompt": "Lamport's bakery algorithm for N processes ensuring mutual exclusion using ticket numbers.", "difficulty": 5, "module_hint": "LamportBakery"},
    {"id": "RL022", "prompt": "A multi-decree Paxos (Multi-Paxos) where a stable leader sequences multiple values, each in a separate Paxos instance.", "difficulty": 5, "module_hint": "MultiPaxos"},
]


def load_prompt_bank(difficulty_cap: int = 5) -> list[dict]:
    """Load benchmark prompts + extra RL prompts, filtered by difficulty."""
    prompts = []

    # Benchmark id → Examples module name (for condensed description injection)
    bm_to_module: dict[str, str] = {}
    if _BENCHMARK_TO_MODULE.exists():
        try:
            meta = json.loads(_BENCHMARK_TO_MODULE.read_text(encoding="utf-8"))
            for m in meta.get("mappings", []):
                bid = m.get("benchmark_id")
                mn = m.get("module_name")
                if bid and mn:
                    bm_to_module[str(bid)] = str(mn)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"[prompt_bank] Could not load benchmark_to_module: {e}")

    desc_by_module: dict[str, dict] = {}
    if _TLA_DESCRIPTIONS_JSON.exists():
        try:
            rows = json.loads(_TLA_DESCRIPTIONS_JSON.read_text(encoding="utf-8"))
            for r in rows:
                mn = r.get("module_name")
                if mn:
                    desc_by_module[str(mn)] = r
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"[prompt_bank] Could not load tla_descriptions: {e}")

    _tds = _REPO_ROOT / "scripts" / "tla_description_sources"
    if str(_tds) not in sys.path:
        sys.path.insert(0, str(_tds))
    from description_prompt import condense_description_row  # noqa: PLC0415

    n_enriched = 0

    # Benchmark suite
    if _BENCHMARK_JSON.exists():
        with open(_BENCHMARK_JSON) as f:
            benchmarks = json.load(f)
        for bm in benchmarks:
            if bm.get("difficulty", 1) <= difficulty_cap:
                base = bm["description"] + (f"\n\nHints: {bm['hints']}" if bm.get("hints") else "")
                bid = str(bm.get("id", ""))
                mid = bm_to_module.get(bid)
                if mid and mid in desc_by_module:
                    block = condense_description_row(
                        desc_by_module[mid],
                        max_narrative_chars=800,
                        max_next_chars=1200,
                        max_init_chars=600,
                    )
                    base = (
                        "Reference (from tlaplus/Examples-style analysis of the target module; "
                        "do not copy verbatim — use as guidance only):\n\n"
                        + block
                        + "\n\n---\n\nTask:\n"
                        + base
                    )
                    n_enriched += 1
                prompts.append({
                    "id": bm["id"],
                    "prompt": base,
                    "difficulty": bm.get("difficulty", 3),
                    "module_hint": bm["name"].replace(" ", "").replace("'", ""),
                })
    if n_enriched:
        log.info(f"[prompt_bank] Injected condensed descriptions for {n_enriched} benchmark prompt(s)")

    # Self-improve synthetic prompts
    from src.training.self_improve import load_prompts as load_si_prompts
    si_prompts = load_si_prompts()
    for p in si_prompts:
        if p["id"].startswith("SYN"):
            prompts.append({
                "id": p["id"],
                "prompt": p["prompt"],
                "difficulty": 2,
                "module_hint": p.get("module_hint", "Spec"),
            })

    # Extra RL prompts
    for p in _EXTRA_PROMPTS:
        if p["difficulty"] <= difficulty_cap:
            prompts.append(p)

    return prompts


# ─────────────────────────────────────────────────────────────────────────────
# TLC granular feedback extractor
# ─────────────────────────────────────────────────────────────────────────────
def extract_tlc_feedback(spec_result: SpecResult) -> str:
    """
    Parse TLC output to produce line-by-line, actionable feedback.
    Expects SpecResult (has tlc_raw_output, tlc_violations), NOT TLCResult (raw_output).
    """
    raw = spec_result.tlc_raw_output
    violations = spec_result.tlc_violations
    feedback_lines = []

    if not violations and not raw:
        return ""

    # Extract which invariant was violated
    inv_match = re.search(r"Invariant (\w+) is violated", raw, re.IGNORECASE)
    if inv_match:
        feedback_lines.append(f"INVARIANT VIOLATION: {inv_match.group(1)} was violated.")

    # Extract the error state trace
    state_trace = []
    in_trace = False
    for line in raw.splitlines():
        stripped = line.strip()
        if "behavior up to this point" in stripped.lower():
            in_trace = True
            continue
        if in_trace:
            if stripped.startswith("/\\"):
                state_trace.append(stripped)
            elif stripped.startswith("State ") or stripped.startswith("->"):
                state_trace.append(stripped)
            elif stripped == "" and state_trace:
                continue
            elif re.match(r"^\d+ state", stripped):
                break

    if state_trace:
        feedback_lines.append("STATE TRACE (the execution that led to the violation):")
        for s in state_trace[:20]:  # cap to prevent token explosion
            feedback_lines.append(f"  {s}")

    # Extract specific error messages
    for v in violations:
        if "violated" not in v.lower() and "error" in v.lower():
            feedback_lines.append(f"TLC ERROR: {v}")

    # Look for "evaluating" errors that indicate undefined operators
    eval_errors = re.findall(r"Error evaluating.*?(?=\n\n|\Z)", raw, re.DOTALL)
    for e in eval_errors[:3]:
        feedback_lines.append(f"EVALUATION ERROR: {e.strip()[:200]}")

    # Detect common patterns and give actionable advice
    if "not enumerable" in raw.lower() or "cannot enumerate" in raw.lower():
        feedback_lines.append(
            "FIX: State space is not enumerable. Use finite sets and bounded constants. "
            "Replace Int with a bounded range like 0..N, replace SUBSET with explicit enumeration."
        )
    if "CONSTANT" in raw and "is not assigned" in raw.lower():
        feedback_lines.append(
            "FIX: A CONSTANT is declared but not assigned a value in the .cfg file. "
            "Either remove the CONSTANT or provide a default value."
        )
    if "operator" in raw.lower() and "undefined" in raw.lower():
        undef = re.findall(r"(?:Unknown|undefined)\s+operator:\s*(\w+)", raw, re.IGNORECASE)
        if undef:
            feedback_lines.append(f"FIX: Undefined operator(s): {', '.join(undef)}. Check EXTENDS and operator definitions.")

    return "\n".join(feedback_lines) if feedback_lines else "\n".join(violations[:5])


# ─────────────────────────────────────────────────────────────────────────────
# Error taxonomy — classify failures for targeted training
# ─────────────────────────────────────────────────────────────────────────────
def classify_error(spec_result: SpecResult) -> str:
    """
    Classify a non-gold SpecResult into an error category.

    Categories: syntax, invariant_violation, deadlock, unbounded_state,
    undefined_op, timeout, other.  Used for tracking failure distributions
    and targeting training data generation at the weakest failure mode.
    """
    if not spec_result.sany_pass:
        return "syntax"
    if spec_result.tlc_timeout:
        return "timeout"

    raw = (spec_result.tlc_raw_output or "").lower()
    violations = " ".join(spec_result.tlc_violations).lower()
    combined = raw + " " + violations

    if "deadlock" in combined:
        return "deadlock"
    if "invariant" in combined and "violated" in combined:
        return "invariant_violation"
    if "not enumerable" in combined or "cannot enumerate" in combined or "overflow" in combined:
        return "unbounded_state"
    if "undefined" in combined or "unknown operator" in combined:
        return "undefined_op"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# RLVR — Verifier-guided repair
# ─────────────────────────────────────────────────────────────────────────────
_REPAIR_PROMPT_TEMPLATE = """\
This TLA+ specification fails TLC model-checking with the following errors:

--- TLC ERRORS ---
{tlc_feedback}
--- END ERRORS ---

Broken spec:
{broken_spec}

Fix ALL errors. The most common causes are:
1. Unbounded sets: replace \\IN Nat/Int with \\IN 0..N (declare CONSTANTS N)
2. Deadlock: every state must have a successor — add UNCHANGED <<vars>> to actions that terminate
3. Integer overflow: guard incrementing ops with x < MAX precondition
4. Undefined CONSTANTS: declare them and assign values in the TLC config block at the end
5. Non-enumerable initial states: ensure Init assigns concrete finite values

Output ONLY the corrected TLA+ spec. No explanation.\
"""


def _attempt_tlc_repair(
    client,
    p: dict,
    broken_spec: str,
    tlc_violations: list[str],
    tlc_raw: str,
    module_name: str,
    max_rounds: int = MAX_REPAIR_ROUNDS,
) -> list["SpecResult"]:
    """
    RLVR core: feed TLC verifier errors back to the model and attempt repair.

    Each successful repair (SANY+TLC pass) becomes a strong training signal:
    the (broken_spec, tlc_error) → fixed_spec trajectory is stored on the
    returned SpecResult for use in build_training_data.

    Returns a list of SpecResult objects produced by repair rounds.
    """
    from src.inference.benchmark import score_structural
    from src.validators.sany_validator import validate_string as sany_validate
    from src.validators.tlc_validator import validate_string as tlc_validate
    from src.training.self_improve import fix_tla_syntax

    pid = p["id"]
    prompt_text = p["prompt"]
    module_hint = p.get("module_hint", "Spec")
    repair_results: list[SpecResult] = []

    current_broken = broken_spec
    current_violations = tlc_violations
    current_raw = tlc_raw

    for round_i in range(max_rounds):
        if _SHUTDOWN:
            break

        # Build concise, actionable TLC feedback
        feedback_parts = []
        for v in current_violations[:6]:
            if v and "TLC_TIMEOUT" not in v:
                feedback_parts.append(v)
        if "not enumerable" in current_raw.lower():
            feedback_parts.append("FIX: replace \\\\IN Nat/Int with bounded range like 0..N")
        if "deadlock" in current_raw.lower():
            feedback_parts.append("FIX: add UNCHANGED <<vars>> to ensure every state has a successor")
        if "overflow" in current_raw.lower():
            feedback_parts.append("FIX: guard the operation with a bound check (e.g., x < MAX)")
        tlc_feedback_str = "\n".join(feedback_parts) if feedback_parts else "\n".join(current_violations[:4])

        repair_prompt = _REPAIR_PROMPT_TEMPLATE.format(
            tlc_feedback=tlc_feedback_str[:800],
            broken_spec=current_broken[:2500],
        )

        try:
            repaired = client.generate_spec(repair_prompt, module_name=module_hint, temperature=0.15)
        except Exception as e:
            log.warning(f"[{pid}] RLVR repair round {round_i + 1} failed: {e}")
            break

        m = re.search(r"----\s*MODULE\s+(\w+)", repaired)
        rep_module = m.group(1) if m else module_name

        fix_result = fix_tla_syntax(repaired)
        if fix_result.fixes_applied:
            repaired = fix_result.fixed_spec

        sany_result = sany_validate(repaired, module_name=rep_module)
        if not sany_result.valid:
            log.debug(f"[{pid}] RLVR round {round_i + 1}: SANY still failing — stopping repair")
            break

        rep_timeout = False
        tlc_result = tlc_validate(repaired, module_name=rep_module)
        tier = tlc_result.tier
        tlc_ok = (tier == "gold")
        rep_violations = list(tlc_result.tlc_violations)
        rep_raw = tlc_result.raw_output

        if "timed out" in rep_raw.lower():
            rep_timeout = True
            rep_violations.append("TLC_TIMEOUT: state space too large — add CONSTANTS bounds")

        struct_score = score_structural(repaired, [])

        result = SpecResult(
            prompt_id=pid,
            prompt_text=prompt_text,
            spec=repaired,
            tier=tier,
            sany_pass=True,
            tlc_pass=tlc_ok,
            tlc_violations=rep_violations,
            tlc_raw_output=rep_raw,
            fixes_applied=fix_result.fixes_applied or [],
            structural_score=struct_score,
            attempts=round_i + 1,
            temperature=0.15,
            tlc_timeout=rep_timeout,
            repair_from_spec=current_broken,
            repair_from_violations=current_violations,
            repair_type="tlc",
        )
        repair_results.append(result)

        if tlc_ok:
            log.info(f"  [{pid}] RLVR repair round {round_i + 1}: GOLD — verifier-guided fix succeeded")
            break
        elif rep_timeout:
            break  # no point iterating on an unbounded spec
        else:
            # Not gold yet but SANY passes — try one more round with the improved spec
            log.debug(f"[{pid}] RLVR round {round_i + 1}: silver → try again")
            current_broken = repaired
            current_violations = rep_violations
            current_raw = rep_raw

    return repair_results


# ─────────────────────────────────────────────────────────────────────────────
# SANY repair — error-guided repair for bronze (syntax-failing) specs
# ─────────────────────────────────────────────────────────────────────────────
_SANY_REPAIR_PROMPT_TEMPLATE = """\
This TLA+ specification has SANY parse errors (it does not parse):

--- SANY ERRORS ---
{sany_errors}
--- END ERRORS ---

{hints}Broken spec:
{broken_spec}

Fix ALL parse errors. Common causes:
1. PlusCal syntax mixed in — use only pure TLA+ (Init ==, Next ==, primed vars)
2. CONSTANT declarations with values — use 'CONSTANT N' then define 'N == 5' separately
3. Missing ==== closing delimiter
4. Double prime (x'') instead of single prime (x')
5. vars defined as set {{...}} instead of tuple <<...>>
6. UNCHANGED with bare variable instead of tuple <<var>>

Output ONLY the corrected TLA+ spec. No explanation.\
"""


def _attempt_sany_repair(
    client,
    p: dict,
    broken_spec: str,
    sany_errors: list[str],
    sany_raw: str,
    module_name: str,
    max_rounds: int = MAX_SANY_REPAIR_ROUNDS,
) -> list["SpecResult"]:
    """
    SANY repair core: feed SANY parse errors back to the model and attempt repair.

    Mirror of _attempt_tlc_repair but for syntax errors (bronze specs).
    Each successful repair (SANY pass) that reaches gold becomes strong training signal.
    Returns a list of SpecResult objects produced by repair rounds.
    """
    from src.inference.benchmark import score_structural
    from src.validators.sany_validator import validate_string as sany_validate
    from src.validators.tlc_validator import validate_string as tlc_validate
    from src.training.self_improve import fix_tla_syntax
    from src.inference.ollama_client import _diagnose_sany_errors

    pid = p["id"]
    prompt_text = p["prompt"]
    module_hint = p.get("module_hint", "Spec")
    repair_results: list[SpecResult] = []

    current_broken = broken_spec
    current_errors = sany_errors
    current_raw = sany_raw

    for round_i in range(max_rounds):
        if _SHUTDOWN:
            break

        # Build error feedback with targeted hints
        error_str = "\n".join(current_errors[:8]) if current_errors else current_raw[-500:]
        hints = _diagnose_sany_errors(current_broken, error_str)
        hints_block = f"Known issues to fix:\n{hints}\n" if hints else ""

        repair_prompt = _SANY_REPAIR_PROMPT_TEMPLATE.format(
            sany_errors=error_str[:800],
            hints=hints_block,
            broken_spec=current_broken[:2500],
        )

        try:
            repaired = client.generate_spec(repair_prompt, module_name=module_hint, temperature=0.15)
        except Exception as e:
            log.warning(f"[{pid}] SANY repair round {round_i + 1} failed: {e}")
            break

        m = re.search(r"----\s*MODULE\s+(\w+)", repaired)
        rep_module = m.group(1) if m else module_name

        # Apply deterministic fixes between rounds
        fix_result = fix_tla_syntax(repaired)
        if fix_result.fixes_applied:
            repaired = fix_result.fixed_spec

        sany_result = sany_validate(repaired, module_name=rep_module)
        if not sany_result.valid:
            log.debug(f"[{pid}] SANY repair round {round_i + 1}: still failing — retrying")
            current_broken = repaired
            current_errors = sany_result.errors
            current_raw = sany_result.raw_output
            continue  # keep trying (unlike TLC repair, SANY errors are more fixable)

        # SANY passes — run TLC
        tlc_result = tlc_validate(repaired, module_name=rep_module)
        tier = tlc_result.tier
        tlc_ok = (tier == "gold")
        rep_violations = list(tlc_result.tlc_violations)
        rep_raw = tlc_result.raw_output
        rep_timeout = "timed out" in rep_raw.lower()
        if rep_timeout and "TLC_TIMEOUT" not in rep_violations:
            rep_violations.append("TLC_TIMEOUT: state space too large — add CONSTANTS bounds")

        struct_score = score_structural(repaired, [])

        result = SpecResult(
            prompt_id=pid,
            prompt_text=prompt_text,
            spec=repaired,
            tier=tier,
            sany_pass=True,
            tlc_pass=tlc_ok,
            tlc_violations=rep_violations,
            tlc_raw_output=rep_raw,
            fixes_applied=fix_result.fixes_applied or [],
            structural_score=struct_score,
            attempts=round_i + 1,
            temperature=0.15,
            tlc_timeout=rep_timeout,
            repair_from_spec=broken_spec,
            repair_from_violations=sany_errors[:6],
            repair_type="sany",
        )
        repair_results.append(result)

        if tlc_ok:
            log.info(f"  [{pid}] SANY repair round {round_i + 1}: GOLD — syntax fix → gold")
            break
        else:
            log.info(f"  [{pid}] SANY repair round {round_i + 1}: {tier} — SANY fixed, TLC {tier}")
            break  # SANY fixed; further repair is TLC territory (handled by main loop)

    return repair_results


# ─────────────────────────────────────────────────────────────────────────────
# Self-critique — model reviews its own spec before validation
# ─────────────────────────────────────────────────────────────────────────────
_CRITIQUE_PROMPT_TEMPLATE = """\
Review this TLA+ specification for correctness BEFORE we run the model checker.
Check for these common issues:
1. Unbounded sets (\\IN Nat or \\IN Int instead of bounded ranges like 0..N)
2. Missing UNCHANGED clauses (every action must specify unchanged variables)
3. Deadlock potential (can the system get stuck with no valid next state?)
4. Undefined operators or constants not declared
5. Non-finite initial states (Init must assign concrete finite values)

Spec:
{spec}

If you find issues, output the CORRECTED spec. If the spec looks correct, output it unchanged.
Output ONLY the TLA+ spec, no explanation.\
"""


def _critique_spec(client, spec: str, module_hint: str) -> tuple[str, bool]:
    """
    Self-critique: ask the model to review its own spec before validation.

    Returns (possibly_improved_spec, was_changed).
    """
    prompt = _CRITIQUE_PROMPT_TEMPLATE.format(spec=spec[:3000])
    try:
        revised = client.generate_spec(prompt, module_name=module_hint, temperature=0.1)
        # Sanity check: revised should still be a valid TLA+ module shape
        if "MODULE" in revised and len(revised) > 50:
            changed = revised.strip() != spec.strip()
            return revised, changed
    except Exception as e:
        log.debug(f"[critique] Self-critique failed: {e}")
    return spec, False  # fallback to original


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Generate + Validate specs
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_phase1_workers(requested: int) -> int:
    """Parallel prompts against Ollama; TLC/SANY use isolated temp dirs per call."""
    if requested > 0:
        return max(1, requested)
    env = os.environ.get("CHATTLA_PHASE1_WORKERS", "").strip()
    if env.isdigit():
        return max(1, int(env))
    # Optimized for 2x RTX 3090/A6000 (48GB each) + Ollama concurrency
    return 8 if is_nighttime() else 4


def _generate_for_prompt(
    p: dict,
    model: str,
    max_attempts: int,
    client: Optional["ChatTLAClient"] = None,
) -> list[SpecResult]:
    """Generate + validate one prompt. Pass ``client=None`` to create a thread-local client."""
    from src.inference.ollama_client import ChatTLAClient
    
    # If no client passed, create one (standard for worker threads)
    if client is None:
        try:
            client = ChatTLAClient(model=model, reasoning="medium")
        except Exception as e:
            log.error(f"Failed to create ChatTLAClient for prompt {p.get('id', 'unknown')}: {e}")
            return []

    from src.inference.benchmark import score_structural
    from src.validators.sany_validator import validate_string as sany_validate
    from src.validators.tlc_validator import validate_string as tlc_validate
    from src.training.self_improve import fix_tla_syntax

    pid = p["id"]
    prompt_text = p["prompt"]
    module_hint = p.get("module_hint", "Spec")
    attempt_results: list[SpecResult] = []
    out: list[SpecResult] = []

    for attempt in range(max_attempts):
        if _SHUTDOWN:
            break
        temp = TEMPERATURE_BASE + random.uniform(-0.15, 0.25) if attempt > 0 else TEMPERATURE_BASE
        temp = max(TEMPERATURE_RANGE[0], min(TEMPERATURE_RANGE[1], temp))

        try:
            if client:
                client._temp_override = temp if attempt > 0 else None
                spec = client.generate_spec(prompt_text, module_name=module_hint, temperature=temp)
            else:
                log.error(f"[{pid}] Client is None during generation attempt {attempt+1}")
                continue
        except Exception as e:
            log.warning(f"[{pid}] Generation failed (attempt {attempt+1}): {e}")
            continue

        m = re.search(r"----\s*MODULE\s+(\w+)", spec)
        module_name = m.group(1) if m else module_hint

        fix_result = fix_tla_syntax(spec)
        if fix_result.fixes_applied:
            spec = fix_result.fixed_spec

        # ── Self-critique: model reviews its own spec (first attempt only) ──
        critique_changed = False
        if attempt == 0:
            spec, critique_changed = _critique_spec(client, spec, module_hint)
            if critique_changed:
                # Re-extract module name and re-apply syntax fixes after critique
                m = re.search(r"----\s*MODULE\s+(\w+)", spec)
                module_name = m.group(1) if m else module_hint
                fix_result2 = fix_tla_syntax(spec)
                if fix_result2.fixes_applied:
                    spec = fix_result2.fixed_spec
                    fix_result = fix_result2
                log.debug(f"[{pid}] Self-critique revised the spec")

        sany_result = sany_validate(spec, module_name=module_name)
        sany_ok = sany_result.valid

        tlc_ok = False
        tlc_violations: list[str] = []
        tlc_raw = ""
        tier = "bronze"

        tlc_timeout = False
        if sany_ok:
            tlc_result = tlc_validate(spec, module_name=module_name)
            tier = tlc_result.tier
            tlc_ok = (tier == "gold")
            tlc_violations = tlc_result.tlc_violations
            tlc_raw = tlc_result.raw_output
            if "timed out" in tlc_raw.lower() or "timed out" in " ".join(tlc_violations).lower():
                tlc_timeout = True
                if "TLC_TIMEOUT" not in tlc_violations:
                    tlc_violations = list(tlc_violations) + ["TLC_TIMEOUT: state space too large — add CONSTANTS bounds"]

        struct_score = score_structural(spec, [])

        result = SpecResult(
            prompt_id=pid,
            prompt_text=prompt_text,
            spec=spec,
            tier=tier,
            sany_pass=sany_ok,
            tlc_pass=tlc_ok,
            tlc_violations=tlc_violations,
            tlc_raw_output=tlc_raw,
            fixes_applied=fix_result.fixes_applied if fix_result.fixes_applied else [],
            structural_score=struct_score,
            attempts=attempt + 1,
            temperature=temp,
            tlc_timeout=tlc_timeout,
            critique_applied=critique_changed,
        )
        # Classify error for taxonomy tracking
        if tier != "gold":
            result.error_class = classify_error(result)
        attempt_results.append(result)

        if tier in ("silver", "bronze") and (tlc_violations or tlc_raw):
            log_tlc_error(pid, prompt_text, tier, tlc_violations, tlc_raw, spec[:1200])

        if tier == "gold":
            break

        # ── RLVR: verifier-guided repair ─────────────────────────────────
        # If the spec is silver (SANY passes, TLC fails) and not a timeout,
        # feed the TLC errors back to the model and attempt repair.
        # Repair results carry repair_from_spec so build_training_data can
        # create (broken + error → fixed) training pairs.
        if tier == "silver" and sany_ok and not tlc_timeout and tlc_violations:
            repair_results = _attempt_tlc_repair(
                client, p, spec, tlc_violations, tlc_raw, module_name,
                max_rounds=MAX_REPAIR_ROUNDS,
            )
            attempt_results.extend(repair_results)
            if any(r.tier == "gold" for r in repair_results):
                break

        # ── SANY repair: error-guided repair for bronze specs ──────────
        # Feed SANY parse errors back to the model with targeted hints.
        # Creates (broken_spec + sany_errors → fixed_spec) training pairs.
        if not sany_ok:
            sany_repair_results = _attempt_sany_repair(
                client, p, spec,
                sany_result.errors, sany_result.raw_output, module_name,
                max_rounds=MAX_SANY_REPAIR_ROUNDS,
            )
            attempt_results.extend(sany_repair_results)
            if any(r.tier == "gold" for r in sany_repair_results):
                break
            # If SANY repair produced a silver, try TLC repair on it
            silver_repairs = [r for r in sany_repair_results if r.tier == "silver" and r.tlc_violations and not r.tlc_timeout]
            for sr in silver_repairs[:1]:
                tlc_repair_results = _attempt_tlc_repair(
                    client, p, sr.spec, sr.tlc_violations, sr.tlc_raw_output, module_name,
                    max_rounds=MAX_REPAIR_ROUNDS,
                )
                attempt_results.extend(tlc_repair_results)
                if any(r.tier == "gold" for r in tlc_repair_results):
                    break

    if attempt_results:
        tier_rank = {"gold": 3, "silver": 2, "bronze": 1}
        best = max(attempt_results, key=lambda r: tier_rank.get(r.tier, 0))
        out.extend(attempt_results)
        log.info(
            f"  [{pid}] tier={best.tier} sany={best.sany_pass} "
            f"tlc={best.tlc_pass} struct={best.structural_score:.2f} "
            f"attempts={len(attempt_results)} (best of {len(attempt_results)})"
        )

    return out


def generate_and_validate(
    prompts: list[dict],
    model: str = "chattla:20b",
    max_attempts: int = 3,
    prompt_cooldown_s: float = 0.0,
    phase1_max_workers: int = 0,
) -> list[SpecResult]:
    """Generate specs for all prompts, validate each, and return results."""
    from src.inference.ollama_client import ChatTLAClient

    workers = _resolve_phase1_workers(phase1_max_workers)
    results: list[SpecResult] = []

    if len(prompts) == 0:
        return results

    if workers <= 1 or len(prompts) == 1:
        client = None
        try:
            client = ChatTLAClient(model=model, reasoning="medium")
        except Exception as e:
            log.error(f"[phase1] Failed to create client: {e}")
            return results
        for p in prompts:
            if _SHUTDOWN:
                break
            results.extend(_generate_for_prompt(p, model, max_attempts, client))
            if prompt_cooldown_s > 0 and not _SHUTDOWN:
                time.sleep(prompt_cooldown_s)
        try:
            client._temp_override = None
        except AttributeError:
            pass
        return results

    log.info(f"[phase1] Parallel generation: {workers} workers for {len(prompts)} prompts")
    max_w = min(workers, len(prompts))

    # Each worker thread creates its own client to avoid Ollama race conditions.
    with ThreadPoolExecutor(max_workers=max_w) as ex:
        futs = [
            ex.submit(_generate_for_prompt, p, model, max_attempts, None)
            for p in prompts
        ]
        for fut in futs:
            if _SHUTDOWN:
                break
            try:
                results.extend(fut.result())
            except Exception as e:
                log.error(f"[phase1] Prompt worker failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Build training data from results
# ─────────────────────────────────────────────────────────────────────────────

# Use the canonical developer prompt from dataset_builder — same prompt the
# model is trained on.  The extended TLC-specific guidance (bounded state
# space, NEXT totality, overflow prevention) is applied as a user-message
# addendum during generation, not baked into the developer prompt, to avoid
# train/inference distribution mismatch.
from src.training.dataset_builder import _DEVELOPER_PROMPT


def build_training_data(results: list[SpecResult]) -> tuple[list[dict], list[dict]]:
    """
    Build SFT examples and DPO pairs from generation results.

    Returns (sft_examples, dpo_pairs).

    SFT: gold always; silver when best tier is silver (SANY pass, TLC fail) — stored for
    optional merge. DPO chosen must be gold. Bronze never as positive SFT.
    """
    sft_examples = []
    dpo_pairs = []

    # Group results by prompt
    by_prompt: dict[str, list[SpecResult]] = {}
    for r in results:
        by_prompt.setdefault(r.prompt_id, []).append(r)

    for pid, group in by_prompt.items():
        # Sort by tier quality (gold > silver > bronze)
        tier_rank = {"gold": 3, "silver": 2, "bronze": 1}
        group.sort(key=lambda x: tier_rank.get(x.tier, 0), reverse=True)

        best = group[0]

        # SFT: Gold is the target. High-structural Silver (SANY pass, high-score)
        # serves as a fallback but is NOT labeled 'gold'.
        # Never use a timed-out spec as SFT — it means the state space is unbounded.
        if best.tlc_timeout and best.tier != "gold":
            continue

        if best.tier == "gold":
            sft_examples.append({
                "_tier": "gold",
                "_prompt_id": pid,
                "messages": [
                    {"role": "developer", "content": _DEVELOPER_PROMPT},
                    {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{best.prompt_text}"},
                    {"role": "assistant", "channel": "analysis", "content": "I'll write a well-formed TLA+ specification with proper Init, Next, and invariants."},
                    {"role": "assistant", "channel": "final", "content": best.spec.strip()},
                ],
            })
        elif best.tier == "silver" and getattr(best, "structural_score", 0) > 0.85:
            # Silver SFT: Stricter filtering (0.85).
            sft_examples.append({
                "_tier": "silver",
                "_prompt_id": pid,
                "messages": [
                    {"role": "developer", "content": _DEVELOPER_PROMPT},
                    {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{best.prompt_text}"},
                    {"role": "assistant", "channel": "analysis", "content": "I'll ensure the TLA+ syntax is correct and the core logic is sound."},
                    {"role": "assistant", "channel": "final", "content": best.spec.strip()},
                ],
            })

            # NOTE: Bugfix pairs where the "fix" target is silver (TLC-failing) were
            # removed. Only gold specs should be used as correction targets — otherwise
            # we teach the model that semantically broken specs are valid fixes.

        # DPO: Gold (chosen) vs worse — teaches model to prefer TLC-passing specs
        if best.tier == "gold" and len(group) > 1:
            worst = group[-1]
            if worst.tier != "gold" and tier_rank.get(worst.tier, 0) < tier_rank.get(best.tier, 0):
                feedback = ""
                if worst.tlc_violations:
                    feedback = "\n".join(worst.tlc_violations[:5])
                elif not worst.sany_pass:
                    feedback = "SANY parse errors (spec is syntactically invalid)"
                else:
                    feedback = f"TLC failed (silver) vs {best.tier}"

                dpo_pairs.append({
                    "prompt": f"Write a TLA+ specification for the following:\n\n{best.prompt_text}",
                    "chosen": best.spec.strip(),
                    "rejected": worst.spec.strip(),
                    "chosen_tier": best.tier,
                    "rejected_tier": worst.tier,
                    "feedback": feedback,
                })
        # NOTE: Silver specs are ONLY added as SFT via the gated path above
        # (structural_score > 0.85). An unconditional silver SFT path was removed
        # here because it duplicated silver examples and trained on TLC-failing
        # specs as if they were correct.

        # Error-conditioned training: for silver specs with TLC violations,
        # create bug_fix examples if we also have the gold version
        if best.tier == "gold":
            for r in group[1:]:
                if r.tier == "silver" and r.tlc_violations:
                    feedback = extract_tlc_feedback(r)
                    if feedback:
                        sft_examples.append({
                            "_tier": "bugfix",
                            "_prompt_id": f"{pid}_bugfix",
                            "messages": [
                                {"role": "developer", "content": _DEVELOPER_PROMPT},
                                {"role": "user", "content": (
                                    f"This TLA+ spec has TLC model-checking errors:\n\n"
                                    f"TLC feedback:\n{feedback[:500]}\n\n"
                                    f"Buggy spec:\n{r.spec.strip()[:2000]}\n\n"
                                    f"Fix ALL errors and produce a correct spec."
                                )},
                                {"role": "assistant", "channel": "analysis", "content": "I'll analyze the TLC errors and produce a corrected specification."},
                                {"role": "assistant", "channel": "final", "content": best.spec.strip()},
                            ],
                        })
                    break  # only one bug_fix per prompt

        # ── Repair training pairs (TLC + SANY) ────────────────────────────
        # For specs produced by verifier-guided repair, create two signals:
        # 1. SFT: (broken_spec + error → fixed_spec) — teaches the model
        #    how to respond to verifier feedback, not just what good specs look like.
        # 2. DPO: prefer the repaired gold over the original broken —
        #    teaches preference in the repair context.
        for r in group:
            if not r.repair_from_spec or not r.repair_from_violations:
                continue

            repair_feedback = "\n".join(
                v for v in r.repair_from_violations[:6] if "TLC_TIMEOUT" not in v
            )
            is_sany_repair = (r.repair_type == "sany")

            if r.tier == "gold":
                if is_sany_repair:
                    # SANY repair: syntax fix succeeded → gold
                    sft_examples.append({
                        "_tier": "sany_repair",
                        "_prompt_id": f"{pid}_sany_repair",
                        "messages": [
                            {"role": "developer", "content": _DEVELOPER_PROMPT},
                            {"role": "user", "content": (
                                f"This TLA+ spec has SANY parse errors (syntax errors):\n\n"
                                f"SANY errors:\n{repair_feedback[:600]}\n\n"
                                f"Broken spec:\n{r.repair_from_spec.strip()[:2000]}\n\n"
                                f"Fix ALL syntax errors and produce a correct spec."
                            )},
                            {"role": "assistant", "channel": "analysis", "content": (
                                "I'll fix the SANY parse errors by correcting syntax issues."
                            )},
                            {"role": "assistant", "channel": "final", "content": r.spec.strip()},
                        ],
                    })
                    dpo_pairs.append({
                        "prompt": (
                            f"Fix this TLA+ spec. SANY parse errors:\n{repair_feedback[:500]}\n\n"
                            f"Broken spec:\n{r.repair_from_spec.strip()[:1500]}"
                        ),
                        "chosen": r.spec.strip(),
                        "rejected": r.repair_from_spec.strip(),
                        "chosen_tier": "gold",
                        "rejected_tier": "bronze",
                        "feedback": repair_feedback,
                    })
                else:
                    # TLC repair: semantic fix succeeded → gold
                    sft_examples.append({
                        "_tier": "rlvr_repair",
                        "_prompt_id": f"{pid}_rlvr",
                        "messages": [
                            {"role": "developer", "content": _DEVELOPER_PROMPT},
                            {"role": "user", "content": (
                                f"Fix this TLA+ spec. TLC model-checker errors:\n\n"
                                f"{repair_feedback[:600]}\n\n"
                                f"Broken spec:\n{r.repair_from_spec.strip()[:2000]}"
                            )},
                            {"role": "assistant", "channel": "analysis", "content": (
                                "I'll fix the TLC errors by bounding the state space, "
                                "adding UNCHANGED clauses, and declaring any missing CONSTANTS."
                            )},
                            {"role": "assistant", "channel": "final", "content": r.spec.strip()},
                        ],
                    })
                    dpo_pairs.append({
                        "prompt": (
                            f"Fix this TLA+ spec. TLC errors:\n{repair_feedback[:500]}\n\n"
                            f"Broken spec:\n{r.repair_from_spec.strip()[:1500]}"
                        ),
                        "chosen": r.spec.strip(),
                        "rejected": r.repair_from_spec.strip(),
                        "chosen_tier": "gold",
                        "rejected_tier": "silver",
                        "feedback": repair_feedback,
                    })

    # ── Cross-cycle training pairs ────────────────────────────────────────
    # Mine gold specs from previous cycles to create bugfix pairs with
    # current cycle's failures (bronze/silver that didn't reach gold).
    gold_cache = load_gold_spec_cache()
    if gold_cache:
        for pid, group in by_prompt.items():
            tier_rank = {"gold": 3, "silver": 2, "bronze": 1}
            best = max(group, key=lambda x: tier_rank.get(x.tier, 0))
            if best.tier == "gold":
                continue  # already have gold this cycle
            cached_gold = gold_cache.get(pid)
            if not cached_gold:
                continue
            # Use best non-gold spec from this cycle + cached gold
            for r in group:
                if r.tier == "silver" and r.tlc_violations:
                    feedback = extract_tlc_feedback(r)
                    if feedback:
                        sft_examples.append({
                            "_tier": "cross_cycle_bugfix",
                            "_prompt_id": f"{pid}_xcycle",
                            "messages": [
                                {"role": "developer", "content": _DEVELOPER_PROMPT},
                                {"role": "user", "content": (
                                    f"This TLA+ spec has TLC model-checking errors:\n\n"
                                    f"TLC feedback:\n{feedback[:500]}\n\n"
                                    f"Buggy spec:\n{r.spec.strip()[:2000]}\n\n"
                                    f"Fix ALL errors and produce a correct spec."
                                )},
                                {"role": "assistant", "channel": "analysis", "content": "I'll analyze the TLC errors and produce a corrected specification."},
                                {"role": "assistant", "channel": "final", "content": cached_gold.strip()},
                            ],
                        })
                    break
                elif r.tier == "bronze" and not r.sany_pass:
                    sft_examples.append({
                        "_tier": "cross_cycle_syntax",
                        "_prompt_id": f"{pid}_xcycle_syn",
                        "messages": [
                            {"role": "developer", "content": _DEVELOPER_PROMPT},
                            {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{r.prompt_text}"},
                            {"role": "assistant", "channel": "analysis", "content": "I'll write a well-formed TLA+ specification with proper Init, Next, and invariants."},
                            {"role": "assistant", "channel": "final", "content": cached_gold.strip()},
                        ],
                    })
                    break

    return sft_examples, dpo_pairs


def _spec_hash(spec: str) -> str:
    """Short hash of spec content for dedup (different specs for same prompt are valuable)."""
    import hashlib
    return hashlib.sha256(spec.strip().encode()).hexdigest()[:16]


def persist_training_data(sft_examples: list[dict], dpo_pairs: list[dict]) -> tuple[int, int]:
    """Append new training data to augmented.jsonl and dpo_pairs.jsonl."""
    _RL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Dedup on spec content hash — different gold specs for the same prompt
    # are valuable training signal (diversity). Old prompt-only dedup was
    # causing accumulation to stall after ~30 cycles.
    existing_hashes = set()
    if _AUGMENTED_JSONL.exists():
        with open(_AUGMENTED_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        ex = json.loads(line)
                        msgs = ex.get("messages", [])
                        spec = next((m["content"] for m in msgs if m.get("role") == "assistant" and m.get("channel") == "final"), "")
                        existing_hashes.add(_spec_hash(spec))
                    except (json.JSONDecodeError, KeyError):
                        pass

    n_sft = 0
    if sft_examples:
        with open(_AUGMENTED_JSONL, "a", encoding="utf-8") as f:
            for ex in sft_examples:
                msgs = ex.get("messages", [])
                spec = next((m["content"] for m in msgs if m.get("role") == "assistant" and m.get("channel") == "final"), "")
                h = _spec_hash(spec)
                if h not in existing_hashes:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    existing_hashes.add(h)
                    n_sft += 1

    n_dpo = 0
    gold_pairs = [p for p in dpo_pairs if p.get("chosen_tier") == "gold"]
    if gold_pairs:
        _TIER_VAL = {"gold": 3, "bugfix": 3, "silver": 2, "bronze": 1}

        # Load existing pairs into an ordered dict keyed by prompt prefix.
        existing: dict[str, dict] = {}
        if _DPO_JSONL.exists():
            with open(_DPO_JSONL, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            p = json.loads(line)
                            existing[p.get("prompt", "")[:200]] = p
                        except (json.JSONDecodeError, KeyError):
                            pass

        for pair in gold_pairs:
            key = pair["prompt"][:200]
            prev = existing.get(key)
            if prev is None:
                existing[key] = pair
                n_dpo += 1
            elif _TIER_VAL.get(pair["chosen_tier"], 0) > _TIER_VAL.get(prev.get("chosen_tier", ""), 0):
                existing[key] = pair
                n_dpo += 1

        # Atomic write: build content, write to temp, rename.
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=_DPO_JSONL.parent, suffix=".tmp", prefix="dpo_",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                for p in existing.values():
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            os.replace(tmp_path, _DPO_JSONL)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    return n_sft, n_dpo


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Rebuild dataset + Retrain
# ─────────────────────────────────────────────────────────────────────────────
def _count_dpo_gold_pairs() -> int:
    """Gold-only DPO rows (matches train_dpo filter)."""
    if not _DPO_JSONL.exists():
        return 0
    n = 0
    try:
        with open(_DPO_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if o.get("chosen_tier") == "gold":
                    n += 1
    except OSError:
        return 0
    return n


# Lowered from 250 → 200: the best-per-prompt dedup caps augmented rows at ~86
# (one per unique prompt), so the dataset plateaus at ~234.  Quality is high
# enough at 200+ (gold-only augmented + gold benchmark + description_sft).
# Override with --min-train-examples or CHATTLA_MIN_TRAIN_EXAMPLES.
MIN_TRAIN_EXAMPLES = int(os.environ.get("CHATTLA_MIN_TRAIN_EXAMPLES", "200"))


def rebuild_and_retrain(cycle_id: int = 0, publish_hf: bool = PUBLISH_HF_DEFAULT) -> RetrainOutcome:
    """Rebuild training dataset, retrain, merge, GGUF, deploy. HF publish is handled by caller."""

    # 1. Rebuild dataset
    log.info("[retrain] Rebuilding training dataset...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.training.dataset_builder",
             "--sany-only", "--include-augmented", "--include-description-sft",
             "--include-gold-benchmark", "--bugfix-oversample", "2"],
            cwd=str(_REPO_ROOT),
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            log.error(f"[retrain] Dataset rebuild failed: {result.stderr[-300:]}")
            return "failed"
        log.info(f"[retrain] Dataset rebuilt. {result.stdout.strip().split(chr(10))[-1]}")
    except subprocess.TimeoutExpired:
        log.error("[retrain] Dataset rebuild timed out")
        return "failed"

    # Count training examples — abort if too few (tiny datasets damage the base model)
    n_train = 0
    if _TRAIN_JSONL.exists():
        with open(_TRAIN_JSONL) as f:
            n_train = sum(1 for line in f if line.strip())
    if n_train < MIN_TRAIN_EXAMPLES:
        log.warning(
            f"[retrain] Only {n_train} training examples (need >= {MIN_TRAIN_EXAMPLES}). "
            "Skipping retrain to avoid damaging the base model. Accumulate more data first."
        )
        return "skipped_min_train"
    num_epochs = SFT_EPOCHS
    log.info(f"[retrain] {n_train} training examples, {num_epochs} epochs (SFT_EPOCHS={SFT_EPOCHS})")

    # Clean up GPU memory from previous phases (inference, benchmarks, etc.)
    # This is critical on shared machines where 35+ GiB may be in use
    try:
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            log.info("[retrain] GPU memory cleaned (empty_cache + gc)")
    except Exception as e:
        log.warning(f"[retrain] Could not clean GPU memory: {e}")

    # 2. Train — use both GPUs (20B model needs ~40GB + activations; single 48GB GPU OOMs)
    # Unload Ollama model to free GPU 0 VRAM (chattla:20b uses ~22GB).
    try:
        import requests as _req
        _req.post("http://localhost:11434/api/generate",
                  json={"model": "chattla:20b", "keep_alive": 0}, timeout=10)
        import time; time.sleep(3)
        log.info("[retrain] Unloaded Ollama model to free GPU VRAM")
    except Exception as e:
        log.warning(f"[retrain] Could not unload Ollama model: {e}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    night = is_nighttime()

    # Measure actual free VRAM per GPU. On a shared machine other users' processes
    # consume GPU memory that hardcoded limits would ignore, causing OOM at step 3.
    # Leave a conservative safety margin: 70% of free (not total) VRAM per GPU.
    try:
        import torch
        n_gpus = torch.cuda.device_count()
        per_gpu_free: list[int] = []
        for d in range(n_gpus):
            try:
                free_b, _ = torch.cuda.mem_get_info(d)
                per_gpu_free.append(int(free_b // (1024 * 1024)))
            except Exception:
                per_gpu_free.append(20_000)
        # Per-GPU cap: 70% of current free memory (conservative for shared systems)
        safety = 0.70 if not night else 0.75
        max_gpu_memory_mb = int(min(per_gpu_free[0] if per_gpu_free else 20_000, 49152) * safety)
        free_mb = sum(per_gpu_free)
        log.info(f"[retrain] Per-GPU free VRAM: {per_gpu_free} MB → cap={max_gpu_memory_mb} MB/GPU")
    except Exception:
        free_mb = total_gpu_memory_free_mb()
        max_gpu_memory_mb = int(20_000 * 0.70)  # conservative fallback
        log.warning(f"[retrain] Could not measure per-GPU VRAM; using fallback cap={max_gpu_memory_mb} MB")

    est_steps = (n_train * num_epochs) // 8

    # Adapt max_length to available VRAM (avoids OOM on shared machines)
    max_length = max_length_for_vram(free_mb)

    # Gradient accumulation: lower = less peak memory, smaller effective batch.
    # Always use batch_size=1 per device to minimise activation memory on shared GPUs.
    grad_accum_steps: int
    if free_mb < 10_000:
        max_length = min(max_length, 512)
        grad_accum_steps = 2
        log.warning(f"[retrain] CRITICAL: Only {free_mb}MB free VRAM. max_length=512 grad_accum=2")
    elif free_mb < 15_000:
        max_length = min(max_length, 768)
        grad_accum_steps = 4
        log.warning(f"[retrain] Tight VRAM ({free_mb}MB free). max_length=768 grad_accum=4")
    elif free_mb < 22_000:
        grad_accum_steps = 6
        log.info(f"[retrain] Moderate VRAM ({free_mb}MB free). grad_accum=6")
    elif free_mb < 40_000:
        grad_accum_steps = 8
        log.info(f"[retrain] VRAM {free_mb}MB free. grad_accum=8")
    else:
        grad_accum_steps = 8
        log.info(f"[retrain] VRAM {free_mb}MB free. grad_accum=8")

    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log.info(
        f"[retrain] Training: max_length={max_length} grad_accum={grad_accum_steps} "
        f"batch=1/device free_mb={free_mb} cap={max_gpu_memory_mb} MB/GPU"
    )
    train_cmd = [
        sys.executable, "-m", "src.training.train",
        "--epochs", str(num_epochs),
        "--lr", "2e-5",                          # 3e-4 caused catastrophic forgetting; safe LoRA range
        "--max-gpu-memory-mb", str(max_gpu_memory_mb),
        "--max-length", str(max_length),
        "--per-device-batch-size", "1",          # minimise activation memory on shared GPUs
        "--gradient-accumulation-steps", str(grad_accum_steps),
    ]
    dpo_n = _count_dpo_gold_pairs()
    if dpo_n >= 2:
        train_cmd.append("--dpo-after")
        log.info(f"[retrain] DPO-after-SFT enabled ({dpo_n} gold pairs in dpo_pairs.jsonl)")
    result = subprocess.run(
        train_cmd,
        cwd=str(_REPO_ROOT), env=env,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error(f"[retrain] Training failed: {result.stderr[-500:]}")
        return "failed"
    log.info("[retrain] Training complete.")

    if _SHUTDOWN:
        return "failed"

    # 3. Merge LoRA — GPU first unless CHATTLA_MERGE_ON_CPU=1 (avoids CUBLAS OOM on merge).
    merge_env = os.environ.copy()
    merge_env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
    merge_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    prefer_cpu_merge = os.environ.get("CHATTLA_MERGE_ON_CPU", "").strip().lower() in (
        "1", "true", "yes",
    )
    merged_ok = False

    def _merge_cpu() -> subprocess.CompletedProcess:
        cpu_env = os.environ.copy()
        cpu_env["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.run(
            [sys.executable, "-m", "src.training.merge_lora", "--device", "cpu"],
            cwd=str(_REPO_ROOT),
            env=cpu_env,
            capture_output=True,
            text=True,
            timeout=7200,
        )

    def _merge_gpu() -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "src.training.merge_lora"],
            cwd=str(_REPO_ROOT),
            env=merge_env,
            capture_output=True,
            text=True,
            timeout=1800,
        )

    try:
        if prefer_cpu_merge:
            log.info("[retrain] Merging LoRA on CPU (CHATTLA_MERGE_ON_CPU)...")
            result = _merge_cpu()
            if result.returncode == 0:
                log.info("[retrain] LoRA merged (CPU).")
                merged_ok = True
            else:
                log.warning(
                    f"[retrain] CPU merge failed; retrying GPU: {(result.stderr or '')[-400:]}"
                )

        if not merged_ok:
            log.info("[retrain] Merging LoRA weights (GPU)...")
            result = _merge_gpu()
            if result.returncode != 0:
                err_tail = (result.stderr or "")[-800:]
                log.warning(
                    f"[retrain] GPU merge failed ({err_tail[-400:]}); retrying on CPU (slow, ~RAM-heavy)..."
                )
                result = _merge_cpu()
                if result.returncode != 0:
                    log.error(f"[retrain] CPU merge also failed: {(result.stderr or '')[-500:]}")
                    return "failed"
                log.info("[retrain] LoRA merged (CPU fallback).")
            else:
                log.info("[retrain] LoRA merged.")
    except subprocess.TimeoutExpired:
        log.error("[retrain] Merge timed out")
        return "failed"

    # 4. Convert to GGUF + register with Ollama
    #    BUT FIRST: eval-gate — quick SANY check on merged model before deploying.
    #    If the new model is worse than the old one, abort deploy and keep the old GGUF.
    log.info("[retrain] Converting to GGUF (temp) for eval gate...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.inference.convert_to_gguf", "--quant", "Q8_0",
             "--no-ollama-register"],
            cwd=str(_REPO_ROOT),
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            log.error(f"[retrain] GGUF conversion failed: {result.stderr[-300:]}")
            return "failed"
    except subprocess.TimeoutExpired:
        log.error("[retrain] GGUF conversion timed out")
        return "failed"

    # Eval gate: register temp model, run quick SANY eval, compare to baseline
    log.info("[retrain] Eval gate: testing new model before deploy...")
    try:
        import requests as _req
        # Register as temp name
        gguf_dir = _REPO_ROOT / "outputs" / "gguf"
        new_gguf = gguf_dir / "chattla-20b-Q8_0.gguf"
        if new_gguf.exists():
            tmp_modelfile = gguf_dir / "Modelfile.evalgate"
            # Read existing Modelfile template, swap to new GGUF
            existing_mf = (gguf_dir / "Modelfile").read_text()
            import re as _re
            tmp_mf_content = _re.sub(
                r'^FROM .*$',
                f'FROM {new_gguf}',
                existing_mf,
                count=1,
                flags=_re.MULTILINE,
            )
            tmp_modelfile.write_text(tmp_mf_content)
            subprocess.run(
                ["ollama", "create", "chattla:20b-candidate", "-f", str(tmp_modelfile)],
                capture_output=True, text=True, timeout=300,
            )
            # Quick SANY eval on 5 benchmarks
            benchmarks = json.load(open(_REPO_ROOT / "data" / "benchmarks" / "benchmark_suite.json"))
            from src.validators.sany_validator import validate_string
            sany_pass = 0
            for bm in benchmarks[:5]:
                try:
                    resp = _req.post("http://localhost:11434/api/generate", json={
                        "model": "chattla:20b-candidate",
                        "prompt": f"Write a TLA+ specification for: {bm['name']}: {bm['description']}",
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 2048},
                    }, timeout=120)
                    spec = resp.json()["response"]
                    spec = _re.sub(r'<analysis>.*?</analysis>', '', spec, flags=_re.DOTALL).strip()
                    mod = _re.search(r'MODULE\s+(\w+)', spec)
                    mod_name = mod.group(1) if mod else "Temp"
                    sr = validate_string(spec, module_name=mod_name)
                    sany_pass += int(sr.valid)
                except Exception:
                    pass
            # Cleanup temp model
            subprocess.run(["ollama", "rm", "chattla:20b-candidate"],
                          capture_output=True, text=True, timeout=30)
            tmp_modelfile.unlink(missing_ok=True)

            log.info(f"[retrain] Eval gate: new model SANY={sany_pass}/5")
            if sany_pass == 0:
                log.error(
                    "[retrain] EVAL GATE FAILED: new model scores 0/5 SANY. "
                    "Aborting deploy — keeping old model. This retrain caused catastrophic forgetting."
                )
                return "failed"
            log.info(f"[retrain] Eval gate passed ({sany_pass}/5 SANY). Deploying...")
    except Exception as e:
        log.warning(f"[retrain] Eval gate error (deploying anyway): {e}")

    # Actually register with Ollama
    log.info("[retrain] Deploying new GGUF to Ollama...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.inference.convert_to_gguf", "--quant", "Q8_0"],
            cwd=str(_REPO_ROOT),
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            log.error(f"[retrain] GGUF deploy failed: {result.stderr[-300:]}")
            return "failed"
        log.info("[retrain] GGUF deployed to Ollama.")
    except subprocess.TimeoutExpired:
        log.error("[retrain] GGUF deploy timed out")
        return "failed"

    # HF publish is now handled by the caller after a full benchmark quality gate.
    return "ok"


def publish_to_hf(cycle_id: int) -> bool:
    """Upload GGUF + Modelfile + README to Hugging Face Hub."""
    if not os.environ.get("HF_TOKEN"):
        log.info("[hf_publish] Skipped (HF_TOKEN not set)")
        return False
    log.info("[hf_publish] Publishing to Hugging Face Hub...")
    try:
        pub_cmd = [
            sys.executable, "-m", "src.training.publish_hf",
            "--repo", _HF_REPO,
            "--quant", "Q8_0",
            "--cycle-id", str(cycle_id),
        ]
        if os.environ.get("CHATTLA_HF_UPLOAD_MERGED", "").strip().lower() in ("1", "true", "yes"):
            pub_cmd.append("--upload-merged-model")
        pub = subprocess.run(
            pub_cmd, cwd=str(_REPO_ROOT), env=os.environ.copy(),
            capture_output=True, text=True, timeout=7200,
        )
        if pub.returncode != 0:
            log.warning(f"[hf_publish] Failed (non-fatal): {(pub.stderr or pub.stdout)[-500:]}")
            return False
        tail = (pub.stdout or "").strip().splitlines()
        log.info(f"[hf_publish] OK. {tail[-1] if tail else ''}")
        return True
    except subprocess.TimeoutExpired:
        log.warning("[hf_publish] Timed out (non-fatal)")
        return False


def best_historical_full_tlc() -> float:
    """Return the best TLC rate from any full benchmark CSV."""
    best = 0.0
    import glob as _glob
    
    # Check both old location (outputs/) and new location (outputs/benchmark_results/)
    for pattern in [
        str(_REPO_ROOT / "outputs" / "benchmark_results_*_full_*.csv"),
        str(_REPO_ROOT / "outputs" / "benchmark_results" / "benchmark_results_*_full_*.csv"),
    ]:
        for path in _glob.glob(pattern):
            try:
                with open(path) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if rows:
                    tlc = sum(1 for r in rows if r.get("tlc_pass") == "1") / len(rows)
                    best = max(best, tlc)
            except Exception:
                pass
    
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Benchmark evaluation
# ─────────────────────────────────────────────────────────────────────────────
def quick_benchmark_timeout_s(limit: int, attempts: int) -> int:
    """Scale subprocess timeout with quick-eval size (default floor matches old 6×2×100s)."""
    return max(1200, limit * attempts * 150)


# Must match `src.inference.benchmark` when run without --limit (full handcrafted suite).
_FULL_BENCHMARK_N = 20


def compute_full_benchmark_timeout(attempts: int, problems: int = _FULL_BENCHMARK_N) -> int:
    """
    Wall-clock cap for the full benchmark subprocess.

    Self-correct + TLC can exceed 2h on a busy GPU after a long retrain; the old
    fixed 7200s caused false (0%, 0%) metrics and skipped HF publish.
    Scales with problems × attempts; floor 4h.
    """
    return max(14_400, problems * attempts * 360)


def run_benchmark(
    cycle_id: int,
    limit: Optional[int] = None,
    attempts: int = 3,
    timeout_s: int = 3600,
    suffix: str = "",
) -> tuple[float, float]:
    """Run benchmark and return (sany_rate, tlc_rate)."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lim = f"_l{limit}" if limit else ""
    suf = f"_{suffix}" if suffix else ""
    
    # Create benchmark_results directory if it doesn't exist
    benchmark_dir = _REPO_ROOT / "outputs" / "benchmark_results"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv = benchmark_dir / f"benchmark_results_rl_c{cycle_id}{lim}{suf}_{timestamp}.csv"

    scope = f"{limit}-problem quick eval" if limit else "full benchmark suite"
    log.info(f"[benchmark] Running {scope} (cycle {cycle_id}, attempts={attempts})...")
    
    # Clean up GPU memory before benchmark to avoid memory fragmentation issues
    try:
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            log.debug("[benchmark] GPU memory cleaned")
    except Exception:
        pass  # Non-critical
    
    try:
        cmd = [
            sys.executable, "-m", "src.inference.benchmark",
            "--model", "chattla:20b",
            "--self-correct",
            "--attempts", str(attempts),
            "--output", str(output_csv),
        ]
        if limit is not None:
            cmd += ["--limit", str(limit)]

        result = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            capture_output=True, text=True, timeout=timeout_s,
        )
        if result.returncode != 0:
            log.error(f"[benchmark] Failed: {result.stderr[-300:]}")
            return 0.0, 0.0
    except subprocess.TimeoutExpired:
        log.error(f"[benchmark] Timed out after {timeout_s}s")
        return 0.0, 0.0

    # Parse results
    if output_csv.exists():
        with open(output_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        n = len(rows) if rows else 1
        sany_rate = sum(1 for r in rows if r.get("sany_pass") == "1") / n
        tlc_rate = sum(1 for r in rows if r.get("tlc_pass") == "1") / n
        log.info(f"[benchmark] Results: sany={sany_rate:.0%} ({int(sany_rate*n)}/{n}) "
                 f"tlc={tlc_rate:.0%} ({int(tlc_rate*n)}/{n})")
        if limit is not None:
            log.info("[benchmark] Quick eval is noisy; full-suite TLC rate is the primary metric (compare across retrains).")
        else:
            log.info("[benchmark] Full suite: prioritize TLC rate for model comparison (SANY is the funnel).")
        return sany_rate, tlc_rate

    return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive difficulty
# ─────────────────────────────────────────────────────────────────────────────
def compute_difficulty_cap(history_path: Path = _RL_HISTORY) -> int:
    """
    Dynamically adjust prompt difficulty from recent rl_history (phase1 stats).

    Uses SANY rate to unlock harder caps, but **gates** with TLC rate so we
    don't pile on hard benchmarks while TLC (the harder objective) is still weak.
    """
    if not history_path.exists():
        return 3  # start with easy/moderate

    recent = []
    with open(history_path) as f:
        for line in f:
            if line.strip():
                try:
                    recent.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if len(recent) < 2:
        return 3

    # Rolling window: phase1 generation/validation (same prompts as training signal)
    last_n = recent[-5:]
    total_specs = sum(
        (c.get("specs_generated") or c.get("prompts_tried") or 0) for c in last_n
    )
    if total_specs < 1:
        return 3
    total_sany = sum(c.get("sany_pass", 0) for c in last_n)
    total_tlc = sum(c.get("tlc_pass", 0) for c in last_n)
    avg_sany = total_sany / total_specs
    avg_tlc = total_tlc / total_specs

    # Base cap from SANY (syntax / static semantics)
    if avg_sany >= 0.8:
        base = 5
    elif avg_sany >= 0.6:
        base = 4
    elif avg_sany >= 0.4:
        base = 3
    else:
        base = 2

    # TLC gate: don't serve hardest prompts while end-to-end TLC is still rare
    if avg_tlc < 0.12:
        base = min(base, 3)
    elif avg_tlc < 0.22:
        base = min(base, 4)

    return base


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def run_cycle(
    cycle_id: int,
    accumulated_new: int,
    gold_prompt_ids: set[str],
    allow_daytime_retrain: bool = False,
    publish_hf: bool = PUBLISH_HF_DEFAULT,
    phase1_max_workers: int = 0,
    benchmark_every_n: int = BENCHMARK_EVERY_N,
    full_benchmark_attempts: int = 2,
    full_benchmark_timeout_override: int = 0,
    quick_eval_limit: int = QUICK_EVAL_LIMIT,
    quick_eval_attempts: int = QUICK_EVAL_ATTEMPTS,
) -> tuple[CycleStats, int, set[str]]:
    """
    Run one full RL cycle: generate → validate → build data → retrain → eval.

    Returns (stats, updated_accumulated_new, gold_prompt_ids).
    """
    global RETRAIN_THRESHOLD
    stats = CycleStats(
        cycle_id=cycle_id,
        timestamp=datetime.datetime.now().isoformat(),
        is_nighttime=is_nighttime(),
    )
    t0 = time.time()

    try:
        # ── Determine schedule parameters ─────────────────────────────────
        night = is_nighttime()
        max_prompts = MAX_PROMPTS_NIGHT if night else MAX_PROMPTS_DAY
        prompt_cooldown = NIGHT_PROMPT_COOLDOWN_S if night else DAY_PROMPT_COOLDOWN_S
        difficulty_cap = compute_difficulty_cap()

        log.info(f"{'='*60}")
        log.info(f"CYCLE {cycle_id} | {'NIGHTTIME' if night else 'DAYTIME'} | "
                 f"difficulty_cap={difficulty_cap} | max_prompts={max_prompts}")
        log.info(f"Pacing: {prompt_cooldown:.1f}s between prompts")
        log.info(f"{'='*60}")

        # ── Phase 1: Generate + Validate ──────────────────────────────────
        log.info("[phase1] Loading prompt bank...")
        all_prompts = load_prompt_bank(difficulty_cap=difficulty_cap)
        
        # ── Stratified Sampling logic ──────────────────────────────────────
        # 1. Anchor set: 20% from prompts that ALREADY produced gold specs (check for regression)
        # 2. Hard set:   30% from difficulty 4-5 prompts
        # 3. Random:     50% from the remaining (mostly unknown or easy-mid)
        
        gold_bank = [p for p in all_prompts if p.get("id") in gold_prompt_ids]
        non_gold_bank = [p for p in all_prompts if p.get("id") not in gold_prompt_ids]
        hard_bank = [p for p in non_gold_bank if p.get("difficulty", 1) >= 4]
        rest_bank = [p for p in non_gold_bank if p.get("difficulty", 1) < 4]

        n_anchor = max(1, int(max_prompts * 0.20))
        n_hard   = max(1, int(max_prompts * 0.30))
        n_random = max_prompts - n_anchor - n_hard

        anchors = random.sample(gold_bank, min(len(gold_bank), n_anchor)) if gold_bank else []
        hards   = random.sample(hard_bank, min(len(hard_bank), n_hard)) if hard_bank else []
        
        # Fill remaining budget from rest_bank, then overflow back into hard_bank if needed
        pool = rest_bank
        random.shuffle(pool)
        selected_random = pool[:n_random]
        
        prompts = anchors + hards + selected_random
        # Ensure we don't exceed max_prompts if we over-sampled anchors/hards
        prompts = prompts[:max_prompts]
        
        log.info(f"[phase1] Stratified: anchors={len(anchors)} hard={len(hards)} random={len(selected_random)}")
        stats.prompts_tried = len(prompts)

        log.info(f"[phase1] Generating and validating {len(prompts)} specs...")
        results = generate_and_validate(
            prompts,
            model="chattla:20b",
            max_attempts=2,
            prompt_cooldown_s=prompt_cooldown,
            phase1_max_workers=phase1_max_workers,
        )
        if results is None:
            log.error("[phase1] generate_and_validate returned None, using empty list.")
            results = []

        stats.specs_generated = len(results)
        stats.sany_pass = sum(1 for r in results if r.sany_pass)
        stats.tlc_pass = sum(1 for r in results if r.tlc_pass)
        stats.gold_count = sum(1 for r in results if r.tier == "gold")
        stats.silver_count = sum(1 for r in results if r.tier == "silver")
        stats.bronze_count = sum(1 for r in results if r.tier == "bronze")

        # ── Error taxonomy ────────────────────────────────────────────────
        error_counts: dict[str, int] = {}
        for r in results:
            if r.error_class:
                error_counts[r.error_class] = error_counts.get(r.error_class, 0) + 1
        stats.errors_syntax = error_counts.get("syntax", 0)
        stats.errors_invariant_violation = error_counts.get("invariant_violation", 0)
        stats.errors_deadlock = error_counts.get("deadlock", 0)
        stats.errors_unbounded_state = error_counts.get("unbounded_state", 0)
        stats.errors_undefined_op = error_counts.get("undefined_op", 0)
        stats.errors_timeout = error_counts.get("timeout", 0)
        stats.errors_other = error_counts.get("other", 0)

        # ── Repair + critique stats ───────────────────────────────────────
        stats.sany_repairs_attempted = sum(1 for r in results if r.repair_type == "sany")
        stats.sany_repairs_succeeded = sum(1 for r in results if r.repair_type == "sany" and r.sany_pass)
        stats.critiques_applied = sum(1 for r in results if r.critique_applied)

        # ── Per-prompt regression tracking ────────────────────────────────
        previous_tiers = load_prompt_tiers()
        regressions, improvements, regr_details = compute_prompt_deltas(results, previous_tiers)
        stats.prompt_regressions = regressions
        stats.prompt_improvements = improvements
        if regressions > 0:
            log.warning(f"[phase1] PROMPT REGRESSIONS: {regressions} prompts degraded: {', '.join(regr_details[:5])}")
        if improvements > 0:
            log.info(f"[phase1] Prompt improvements: {improvements} prompts upgraded tier")

        # Save current per-prompt tiers for next cycle
        current_tiers: dict[str, str] = {}
        tier_rank_map = {"gold": 3, "silver": 2, "bronze": 1}
        for r in results:
            prev = current_tiers.get(r.prompt_id)
            if prev is None or tier_rank_map.get(r.tier, 0) > tier_rank_map.get(prev, 0):
                current_tiers[r.prompt_id] = r.tier
        # Merge with previous (don't lose prompts we didn't evaluate this cycle)
        merged_tiers = {**previous_tiers, **current_tiers}
        save_prompt_tiers(merged_tiers)

        # ── Update gold spec cache for cross-cycle mining ─────────────────
        update_gold_spec_cache(results)

        if error_counts:
            log.info(f"[phase1] Error taxonomy: {error_counts}")

        log.info(f"[phase1] Results: gold={stats.gold_count} silver={stats.silver_count} "
                 f"bronze={stats.bronze_count} sany={stats.sany_pass}/{stats.specs_generated} "
                 f"tlc={stats.tlc_pass}/{stats.specs_generated}")
        if stats.sany_repairs_attempted:
            log.info(f"[phase1] SANY repairs: {stats.sany_repairs_succeeded}/{stats.sany_repairs_attempted} succeeded")
        if stats.critiques_applied:
            log.info(f"[phase1] Self-critiques applied: {stats.critiques_applied}")

        if _SHUTDOWN:
            stats.cycle_duration_s = time.time() - t0
            return stats, accumulated_new, gold_prompt_ids

        # ── Phase 2: Build training data ──────────────────────────────────
        log.info("[phase2] Building training data from results...")
        sft_examples, dpo_pairs = build_training_data(results)
        log.info(f"[phase2] Built {len(sft_examples)} SFT examples, {len(dpo_pairs)} DPO pairs "
                 f"(from {len(results)} results, {len(set(r.prompt_id for r in results))} prompts)")
        n_sft, n_dpo = persist_training_data(sft_examples, dpo_pairs)
        stats.new_train_examples = n_sft
        stats.new_dpo_pairs = n_dpo
        accumulated_new += n_sft

        # Track which prompts produced gold specs for future cycles
        gold_prompt_ids = add_gold_prompts(results, gold_prompt_ids)
        if len(gold_prompt_ids) > 0:
            log.info(f"[phase2] Tracking {len(gold_prompt_ids)} prompts with gold results (skip in future cycles)")

        log.info(f"[phase2] Persisted {n_sft} SFT examples, {n_dpo} DPO pairs. "
                 f"Accumulated: {accumulated_new}")

        if _SHUTDOWN:
            stats.cycle_duration_s = time.time() - t0
            return stats, accumulated_new, gold_prompt_ids

        # ── Phase 3: Retrain if threshold met ─────────────────────────────
        just_retrained = False
        if accumulated_new >= RETRAIN_THRESHOLD:
            if night or allow_daytime_retrain:
                when = "night" if night else "daytime (--allow-daytime-retrain)"
                log.info(f"[phase3] Retrain threshold reached ({when}) ({accumulated_new} >= {RETRAIN_THRESHOLD})")
                
                # Check for free GPU memory before retraining to avoid OOM
                free_mb = total_gpu_memory_free_mb()
                if free_mb < 8000:
                    log.warning(f"[phase3] Critically low VRAM ({free_mb}MB < 8000MB). Deferring retrain to avoid OOM crash.")
                    stats.retrain_deferred_vram = True
                    outcome = "skipped_vram"
                else:
                    outcome = rebuild_and_retrain(cycle_id=cycle_id)
                
                stats.retrained = outcome == "ok"
                stats.deployed = outcome == "ok"
                stats.retrain_skipped_min_data = outcome == "skipped_min_train"
                if outcome == "ok":
                    accumulated_new = 0
                    just_retrained = True
                    log.info("[phase3] Retrain + deploy complete!")
                elif outcome == "skipped_vram":
                    log.info("[phase3] Retrain deferred due to memory; will attempt next cycle.")
                else:
                    log.warning("[phase3] Retrain skipped or failed. Will retry next cycle.")
            else:
                stats.retrain_deferred_to_night = True
                log.info(
                    f"[phase3] Threshold reached ({accumulated_new} >= {RETRAIN_THRESHOLD}) "
                    "but retrain deferred to nighttime. Use --allow-daytime-retrain to force."
                )
        else:
            log.info(f"[phase3] Skipping retrain ({accumulated_new}/{RETRAIN_THRESHOLD} examples accumulated)")

        if _SHUTDOWN:
            stats.cycle_duration_s = time.time() - t0
            return stats, accumulated_new, gold_prompt_ids

        # ── Phase 4: Full benchmark every N cycles; quick eval otherwise ──
        if benchmark_every_n <= 0:
            want_full = False
        elif benchmark_every_n == 1:
            want_full = True
        else:
            want_full = (cycle_id % benchmark_every_n) == 0

        stats.benchmark_run = True
        if want_full:
            stats.benchmark_full_suite = True
            fb_to = (
                full_benchmark_timeout_override
                if full_benchmark_timeout_override > 0
                else compute_full_benchmark_timeout(full_benchmark_attempts)
            )
            log.info(
                f"[benchmark] Full suite timeout {fb_to}s "
                f"({_FULL_BENCHMARK_N} problems × {full_benchmark_attempts} attempts; "
                f"override with --full-benchmark-timeout)"
            )
            full_sany, full_tlc = run_benchmark(
                cycle_id,
                limit=None,
                attempts=full_benchmark_attempts,
                timeout_s=fb_to,
                suffix="full",
            )
        else:
            stats.benchmark_full_suite = False
            q_to = quick_benchmark_timeout_s(quick_eval_limit, quick_eval_attempts)
            full_sany, full_tlc = run_benchmark(
                cycle_id,
                limit=quick_eval_limit,
                attempts=quick_eval_attempts,
                timeout_s=q_to,
                suffix="quick",
            )
        stats.benchmark_sany_rate = full_sany
        stats.benchmark_tlc_rate = full_tlc

        if just_retrained and publish_hf:
            if not stats.benchmark_full_suite:
                log.info(
                    "[phase4] HF publish skipped: quality gate needs a full benchmark "
                    f"(this cycle was quick eval only; full runs every {benchmark_every_n} cycles when >1)."
                )
            else:
                prev_best_tlc = best_historical_full_tlc() or 0.0
                full_tlc_safe = full_tlc if isinstance(full_tlc, float) else 0.0
                if full_tlc_safe >= prev_best_tlc and full_tlc_safe > 0:
                    log.info(
                        f"[phase4] TLC {full_tlc_safe:.0%} >= previous best {prev_best_tlc:.0%} — publishing to HF"
                    )
                    publish_to_hf(cycle_id)
                else:
                    log.warning(
                        f"[phase4] TLC {full_tlc_safe:.0%} < previous best {prev_best_tlc:.0%} (or zero) "
                        "— SKIPPING HF publish to avoid pushing a regression"
                    )
                    # ── Phase 5: Regression Guard ─────────────────────────
                    # If TLC dropped significantly, raise the retrain threshold
                    # so the next retrain requires more/better gold data.
                    if full_tlc_safe < (prev_best_tlc - 0.10):
                        new_thresh = min(RETRAIN_THRESHOLD * 2, 200)
                        log.error(
                            f"[safety] TLC REGRESSION: {full_tlc:.0%} vs best {prev_best_tlc:.0%}. "
                            f"Doubling retrain threshold {RETRAIN_THRESHOLD} → {new_thresh} "
                            "to require more gold data before next retrain."
                        )
                        _write_diag("retrain_regression_guard", {
                            "summary": (
                                f"Post-retrain TLC {full_tlc:.0%} regressed from best {prev_best_tlc:.0%}. "
                                f"Raised threshold {RETRAIN_THRESHOLD}→{new_thresh}."
                            ),
                            "full_tlc": full_tlc,
                            "prev_best_tlc": prev_best_tlc,
                            "old_threshold": RETRAIN_THRESHOLD,
                            "new_threshold": new_thresh,
                            "action": "retrain threshold doubled; more gold examples required",
                        })
                        RETRAIN_THRESHOLD = new_thresh
                    elif full_tlc_safe == 0.0 and prev_best_tlc > 0.0:
                        log.error(
                            f"[safety] Post-retrain TLC=0%. Model likely damaged. "
                            "Skipping next retrain until 50 more gold examples accumulate."
                        )
                        _write_diag("retrain_zero_tlc", {
                            "summary": f"Post-retrain TLC=0% in cycle {cycle_id}",
                            "prev_best_tlc": prev_best_tlc,
                            "action": "threshold raised to prevent further damage",
                        })
                        RETRAIN_THRESHOLD = max(RETRAIN_THRESHOLD, accumulated_new + 50)

    except Exception as e:
        stats.error = f"{type(e).__name__}: {str(e)[:200]}"
        stats.full_traceback = traceback.format_exc()
        log.error(f"[cycle {cycle_id}] Unhandled error: {e}")
        log.error(stats.full_traceback)
        
        # Self-healing hook
        _self_heal(stats.error, stats.full_traceback)

    stats.cycle_duration_s = time.time() - t0
    return stats, accumulated_new, gold_prompt_ids


def main():
    global RETRAIN_THRESHOLD, MIN_TRAIN_EXAMPLES, SFT_EPOCHS

    import argparse

    _check_training_deps()

    _default_min_train = int(os.environ.get("CHATTLA_MIN_TRAIN_EXAMPLES", "200"))

    parser = argparse.ArgumentParser(description="ChatTLA autonomous RL loop")
    parser.add_argument("--cycle-hours", type=float, default=CYCLE_HOURS,
                        help="Hours to wait after each cycle before starting the next (0 = no wait; "
                        f"default: {CYCLE_HOURS})")
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="Max cycles to run (0 = infinite)")
    parser.add_argument("--retrain-threshold", type=int, default=RETRAIN_THRESHOLD,
                        help=f"SFT examples before retrain (default: {RETRAIN_THRESHOLD})")
    parser.add_argument("--min-train-examples", type=int, default=_default_min_train,
                        help="Minimum merged train.jsonl rows before SFT (default: from env or 250)")
    parser.add_argument("--sft-epochs", type=int, default=SFT_EPOCHS,
                        help=f"Training epochs per retrain cycle (default: {SFT_EPOCHS})")
    parser.add_argument("--allow-daytime-retrain", action="store_true",
                        help="Retrain during daytime when threshold met (default: defer to night)")
    parser.add_argument("--model", default="chattla:20b")
    parser.add_argument("--no-publish-hf", action="store_true",
                        help="Skip Hugging Face Hub upload after retrain (requires HF_TOKEN when enabled)")
    parser.add_argument(
        "--benchmark-every",
        type=int,
        default=BENCHMARK_EVERY_N,
        help=f"Run full benchmark every N cycles (1=every cycle, 0=quick eval only; default {BENCHMARK_EVERY_N})",
    )
    parser.add_argument(
        "--full-benchmark-attempts",
        type=int,
        default=2,
        help="Attempts per problem for full benchmark (default: 2; lower is faster)",
    )
    parser.add_argument(
        "--full-benchmark-timeout",
        type=int,
        default=0,
        help="Subprocess timeout seconds for full benchmark (0=auto from suite size; default scales past 2h)",
    )
    parser.add_argument("--quick-eval-limit", type=int, default=QUICK_EVAL_LIMIT,
                        help=f"Problems for quick eval when not a full benchmark cycle (default {QUICK_EVAL_LIMIT})")
    parser.add_argument("--quick-eval-attempts", type=int, default=QUICK_EVAL_ATTEMPTS,
                        help=f"Attempts per problem for quick eval (default {QUICK_EVAL_ATTEMPTS})")
    parser.add_argument(
        "--phase1-workers",
        type=int,
        default=0,
        help="Parallel prompt workers for phase 1 (0=auto: 3 night / 2 day; set 1 to disable)",
    )
    args = parser.parse_args()

    RETRAIN_THRESHOLD = args.retrain_threshold
    MIN_TRAIN_EXAMPLES = args.min_train_examples
    SFT_EPOCHS = args.sft_epochs

    cycle_seconds = max(0.0, args.cycle_hours * 3600)

    log.info("=" * 60)
    log.info("  ChatTLA Autonomous RL Loop")
    log.info(f"  Inter-cycle pause: {args.cycle_hours}h ({'none' if cycle_seconds <= 0 else f'{cycle_seconds/60:.0f} min target'})")
    if args.benchmark_every <= 0:
        log.info("  Eval: quick eval only (full benchmark disabled; use --benchmark-every >=1 for full runs)")
    elif args.benchmark_every == 1:
        log.info(f"  Eval: FULL benchmark every cycle ({args.full_benchmark_attempts} attempts/problem)")
    else:
        log.info(
            f"  Eval: full benchmark every {args.benchmark_every} cycles "
            f"({args.full_benchmark_attempts} attempts); quick eval ({args.quick_eval_limit}×{args.quick_eval_attempts}) otherwise"
        )
    log.info(f"  Retrain threshold: {args.retrain_threshold} | Min training size: {MIN_TRAIN_EXAMPLES}")
    log.info(f"  Phase 1 workers: {args.phase1_workers or 'auto (3 night / 2 day)'}")
    if args.allow_daytime_retrain:
        log.info("  Daytime retrain: ENABLED (--allow-daytime-retrain)")
    if args.no_publish_hf:
        log.info("  Hugging Face publish: DISABLED (--no-publish-hf)")
    elif os.environ.get("HF_TOKEN"):
        log.info(f"  Hugging Face publish: ENABLED → {_HF_REPO} (quality-gated: TLC must match/exceed best)")
    else:
        log.info("  Hugging Face publish: will skip (HF_TOKEN not set)")
    log.info(f"  Max cycles: {'infinite' if args.max_cycles == 0 else args.max_cycles}")
    log.info(f"  PID: {os.getpid()}")
    log.info("=" * 60)

    cycle_id = 0
    accumulated_new = load_accumulated_new()
    
    # Load gold prompt IDs from both persisted state and benchmark CSVs
    gold_prompt_ids = load_gold_prompt_ids()
    gold_from_benchmarks = load_gold_prompt_ids_from_benchmarks()
    gold_prompt_ids.update(gold_from_benchmarks)
    
    if gold_from_benchmarks:
        log.info(f"[startup] Loaded {len(gold_from_benchmarks)} gold prompts from benchmark results")
    if len(gold_prompt_ids) > len(gold_from_benchmarks):
        log.info(f"[startup] Total {len(gold_prompt_ids)} gold prompts (including state cache)")
    if accumulated_new > 0:
        log.info(f"Resuming with accumulated_new={accumulated_new} (from state)")

    # Resume cycle count from history
    if _RL_HISTORY.exists():
        with open(_RL_HISTORY) as f:
            cycle_id = sum(1 for line in f if line.strip())
        log.info(f"Resuming from cycle {cycle_id + 1}")

    while not _SHUTDOWN:
        cycle_id += 1
        if args.max_cycles > 0 and cycle_id > args.max_cycles:
            log.info(f"Reached max cycles ({args.max_cycles}). Exiting.")
            break

        cycle_start = time.time()
        stats, accumulated_new, gold_prompt_ids = run_cycle(
            cycle_id,
            accumulated_new,
            gold_prompt_ids,
            args.allow_daytime_retrain,
            publish_hf=not args.no_publish_hf,
            phase1_max_workers=args.phase1_workers,
            benchmark_every_n=args.benchmark_every,
            full_benchmark_attempts=args.full_benchmark_attempts,
            full_benchmark_timeout_override=args.full_benchmark_timeout,
            quick_eval_limit=args.quick_eval_limit,
            quick_eval_attempts=args.quick_eval_attempts,
        )
        accumulated_new = diagnose_and_fix(stats, accumulated_new)
        save_accumulated_new(accumulated_new, gold_prompt_ids)
        log_history(stats)

        cycle_elapsed = time.time() - cycle_start

        # Print cycle summary
        log.info(f"\n{'─'*60}")
        log.info(f"CYCLE {cycle_id} SUMMARY ({stats.cycle_duration_s/60:.1f} min)")
        log.info(f"  Specs: {stats.specs_generated} | SANY: {stats.sany_pass} | TLC: {stats.tlc_pass}")
        log.info(f"  Gold: {stats.gold_count} | Silver: {stats.silver_count} | Bronze: {stats.bronze_count}")
        log.info(f"  New SFT: {stats.new_train_examples} | New DPO: {stats.new_dpo_pairs}")
        log.info(f"  Retrained: {stats.retrained} | Deployed: {stats.deployed}")
        # Error taxonomy
        err_parts = []
        for cat in ("syntax", "invariant_violation", "deadlock", "unbounded_state", "undefined_op", "timeout", "other"):
            n = getattr(stats, f"errors_{cat}", 0)
            if n > 0:
                err_parts.append(f"{cat}={n}")
        if err_parts:
            log.info(f"  Errors: {' | '.join(err_parts)}")
        # Repair + critique stats
        if stats.sany_repairs_attempted > 0:
            log.info(f"  SANY repairs: {stats.sany_repairs_succeeded}/{stats.sany_repairs_attempted} succeeded")
        if stats.critiques_applied > 0:
            log.info(f"  Self-critiques applied: {stats.critiques_applied}")
        # Prompt regressions
        if stats.prompt_regressions > 0 or stats.prompt_improvements > 0:
            log.info(f"  Prompt deltas: +{stats.prompt_improvements} improved, -{stats.prompt_regressions} regressed")
        if stats.benchmark_run:
            _tag = "FULL" if getattr(stats, "benchmark_full_suite", False) else "quick"
            log.info(
                f"  Benchmark ({_tag}): SANY={stats.benchmark_sany_rate:.0%} TLC={stats.benchmark_tlc_rate:.0%}"
            )
        if stats.error:
            log.info(f"  Error: {stats.error}")
        log.info(f"{'─'*60}\n")

        # Optional pacing: sleep to fill target cycle duration (--cycle-hours > 0)
        remaining = cycle_seconds - cycle_elapsed
        if cycle_seconds <= 0:
            pass  # back-to-back cycles
        elif remaining > 60 and not _SHUTDOWN:
            sleep_time = max(60, remaining)
            log.info(f"Sleeping {sleep_time/60:.1f} min until next cycle (--cycle-hours pacing)...")
            sleep_end = time.time() + sleep_time
            while time.time() < sleep_end and not _SHUTDOWN:
                time.sleep(min(30, sleep_end - time.time()))
        elif not _SHUTDOWN:
            log.info("Cycle exceeded target time. Brief 60s cooldown...")
            time.sleep(60)

    log.info("RL loop terminated gracefully.")

    # Final stats
    if _RL_HISTORY.exists():
        with open(_RL_HISTORY) as f:
            all_cycles = [json.loads(l) for l in f if l.strip()]
        total_gold = sum(c.get("gold_count", 0) for c in all_cycles)
        total_silver = sum(c.get("silver_count", 0) for c in all_cycles)
        total_retrains = sum(1 for c in all_cycles if c.get("retrained"))
        log.info(f"Lifetime: {len(all_cycles)} cycles, {total_gold} gold, "
                 f"{total_silver} silver, {total_retrains} retrains")


if __name__ == "__main__":
    main()
