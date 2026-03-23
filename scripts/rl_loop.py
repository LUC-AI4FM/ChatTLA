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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


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
RETRAIN_THRESHOLD  = 50           # new gold examples before retrain (was 10; too aggressive caused overfitting)
NIGHTTIME_START    = 22           # 10 PM
NIGHTTIME_END      = 6            # 6 AM
GPU_VRAM_CAP_DAY   = 0.75         # 75% VRAM cap during daytime (leave 25%)
GPU_VRAM_CAP_NIGHT = 0.90         # 90% VRAM cap at night
MAX_PROMPTS_DAY    = 25           # fewer prompts during daytime
MAX_PROMPTS_NIGHT  = 40           # full speed at night
BENCHMARK_EVERY_N  = 3            # run full benchmark every N cycles
TEMPERATURE_BASE   = 0.3
TEMPERATURE_RANGE  = (0.1, 0.6)   # diversity range for multi-attempt
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
    benchmark_run: bool = False
    benchmark_sany_rate: float = 0.0
    benchmark_tlc_rate: float = 0.0
    cycle_duration_s: float = 0.0
    error: str = ""


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
                free_bytes, _ = torch.cuda.mem_get_info(d)
                total += int(free_bytes // (1024 * 1024))
        return total if total > 0 else 40000
    except Exception:
        return 40000


def max_length_for_vram(free_mb: int) -> int:
    """Choose max_length based on available VRAM to avoid OOM."""
    if free_mb < 35_000:
        return 1536   # very tight (~1 GPU shared)
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


def save_accumulated_new(accumulated_new: int) -> None:
    """Persist accumulated_new so it survives restarts."""
    _RL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_RL_STATE_FILE, "w") as f:
        json.dump({"accumulated_new": accumulated_new}, f)


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
    if stats.retrained is False and accumulated_new >= RETRAIN_THRESHOLD:
        # Retrain was attempted but failed
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
    if stats.specs_generated >= 10:
        gold_rate = stats.gold_count / stats.specs_generated
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
    {"id": "RL002", "prompt": "A counter that increments from 0 to 5 and stops.", "difficulty": 1, "module_hint": "SimpleCounter"},
    {"id": "RL003", "prompt": "A two-state light switch: ON and OFF.", "difficulty": 1, "module_hint": "LightSwitch"},
    {"id": "RL004", "prompt": "A variable that holds a natural number and can be doubled or reset to 1.", "difficulty": 1, "module_hint": "DoublerReset"},
    # Difficulty 2 — simple protocols
    {"id": "RL005", "prompt": "A ticket dispenser: customers take increasing ticket numbers, and a counter shows the currently-served number.", "difficulty": 2, "module_hint": "TicketDispenser"},
    {"id": "RL006", "prompt": "A simple FIFO queue with enqueue and dequeue operations. The queue has bounded capacity N.", "difficulty": 2, "module_hint": "BoundedFIFO"},
    {"id": "RL007", "prompt": "A readers-writers lock where multiple readers can hold the lock simultaneously but writers are exclusive.", "difficulty": 2, "module_hint": "ReadersWriters"},
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
# Phase 1: Generate + Validate specs
# ─────────────────────────────────────────────────────────────────────────────
def generate_and_validate(
    prompts: list[dict],
    model: str = "chattla:20b",
    max_attempts: int = 3,
    prompt_cooldown_s: float = 0.0,
) -> list[SpecResult]:
    """Generate specs for all prompts, validate each, and return results."""
    from src.inference.ollama_client import ChatTLAClient
    from src.validators.sany_validator import validate_string as sany_validate
    from src.validators.tlc_validator import validate_string as tlc_validate
    from src.training.self_improve import fix_tla_syntax

    client = ChatTLAClient(model=model, reasoning="medium")
    results = []

    for p in prompts:
        if _SHUTDOWN:
            break

        pid = p["id"]
        prompt_text = p["prompt"]
        module_hint = p.get("module_hint", "Spec")
        attempt_results: list[SpecResult] = []  # keep ALL attempts for DPO pairs

        for attempt in range(max_attempts):
            temp = TEMPERATURE_BASE + random.uniform(-0.15, 0.25) if attempt > 0 else TEMPERATURE_BASE
            temp = max(TEMPERATURE_RANGE[0], min(TEMPERATURE_RANGE[1], temp))

            try:
                client._temp_override = temp if attempt > 0 else None
                spec = client.generate_spec(prompt_text, module_name=module_hint, temperature=temp)
            except Exception as e:
                log.warning(f"[{pid}] Generation failed (attempt {attempt+1}): {e}")
                continue

            # Extract module name from generated spec
            m = re.search(r"----\s*MODULE\s+(\w+)", spec)
            module_name = m.group(1) if m else module_hint

            # Try Python fixer
            fix_result = fix_tla_syntax(spec)
            if fix_result.fixes_applied:
                spec = fix_result.fixed_spec

            # SANY check
            sany_result = sany_validate(spec, module_name=module_name)
            sany_ok = sany_result.valid

            # TLC check (only if SANY passes)
            tlc_ok = False
            tlc_violations = []
            tlc_raw = ""
            tier = "bronze"

            if sany_ok:
                tlc_result = tlc_validate(spec, module_name=module_name)
                tier = tlc_result.tier
                tlc_ok = (tier == "gold")
                tlc_violations = tlc_result.tlc_violations
                tlc_raw = tlc_result.raw_output

            # Structural score
            from src.inference.benchmark import score_structural
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
                fixes_applied=fix_result.fixes_applied,
                structural_score=struct_score,
                attempts=attempt + 1,
                temperature=temp,
            )
            attempt_results.append(result)

            # Log TLC failures for failure analysis (invariant violations, deadlocks, etc.)
            if tier in ("silver", "bronze") and (tlc_violations or tlc_raw):
                log_tlc_error(pid, prompt_text, tier, tlc_violations, tlc_raw, spec[:1200])

            # If gold, no need to try more
            if tier == "gold":
                break

            # If SANY failed and we have more attempts, try self-correction
            if not sany_ok and attempt < max_attempts - 1:
                try:
                    corrected_spec, corrected_tier = client.validate_and_generate(
                        prompt_text, max_retries=2
                    )
                    c_sany = sany_validate(corrected_spec, module_name=module_name)
                    if c_sany.valid:
                        c_tlc = tlc_validate(corrected_spec, module_name=module_name)
                        corrected_result = SpecResult(
                            prompt_id=pid,
                            prompt_text=prompt_text,
                            spec=corrected_spec,
                            tier=c_tlc.tier,
                            sany_pass=True,
                            tlc_pass=(c_tlc.tier == "gold"),
                            tlc_violations=c_tlc.tlc_violations,
                            tlc_raw_output=c_tlc.raw_output,
                            fixes_applied=[],
                            structural_score=score_structural(corrected_spec, []),
                            attempts=attempt + 1,
                            temperature=temp,
                        )
                        attempt_results.append(corrected_result)
                        if c_tlc.tier == "gold":
                            break
                except Exception:
                    pass
                break  # self-correction already retries internally

        if attempt_results:
            # Append ALL attempts for DPO pairs (chosen vs rejected)
            tier_rank = {"gold": 3, "silver": 2, "bronze": 1}
            best = max(attempt_results, key=lambda r: tier_rank.get(r.tier, 0))
            for r in attempt_results:
                results.append(r)
            log.info(f"  [{pid}] tier={best.tier} sany={best.sany_pass} "
                     f"tlc={best.tlc_pass} struct={best.structural_score:.2f} "
                     f"attempts={len(attempt_results)} (best of {len(attempt_results)})")

        # Gentle pacing to avoid monopolizing shared GPUs.
        if prompt_cooldown_s > 0 and not _SHUTDOWN:
            time.sleep(prompt_cooldown_s)

    client._temp_override = None
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Build training data from results
# ─────────────────────────────────────────────────────────────────────────────
_DEVELOPER_PROMPT = """\
You are ChatTLA, an expert at writing verified TLA+ formal specifications.
When asked to write a TLA+ spec, follow these rules exactly:
1. Start the module with ---- MODULE <ModuleName> ----
2. End with ====
3. Include EXTENDS, VARIABLES, Init, Next, and Spec operators
4. After the TLA+ module, append a TLC configuration block:
   SPECIFICATION Spec
   INVARIANT TypeOK   (if TypeOK is defined)
5. Output only valid TLA+ code. No markdown fences, no explanation outside the spec.
Reasoning: medium\
"""


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

        # SFT: gold (always) + silver when no gold in group (SANY-valid; TLC may fail).
        # Silver is optional at dataset_builder merge time (--no-silver-augmented to drop).
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
            # DPO: only gold (chosen) vs worse — teaches model to prefer TLC-passing specs
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
        elif best.tier == "silver":
            # Cumulative curriculum: keep SANY-valid specs even when TLC fails (benchmark 30–50% regime).
            sft_examples.append({
                "_tier": "silver",
                "_prompt_id": pid,
                "messages": [
                    {"role": "developer", "content": _DEVELOPER_PROMPT},
                    {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{best.prompt_text}"},
                    {"role": "assistant", "channel": "analysis", "content": "I'll write a well-formed TLA+ specification with proper Init, Next, and invariants."},
                    {"role": "assistant", "channel": "final", "content": best.spec.strip()},
                ],
            })

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

        with open(_DPO_JSONL, "w", encoding="utf-8") as f:
            for p in existing.values():
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

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


MIN_TRAIN_EXAMPLES = 500  # don't retrain on tiny datasets — damages the base model


def rebuild_and_retrain(cycle_id: int = 0, publish_hf: bool = PUBLISH_HF_DEFAULT) -> bool:
    """Rebuild training dataset, retrain, merge, GGUF, deploy. HF publish is handled by caller."""

    # 1. Rebuild dataset
    log.info("[retrain] Rebuilding training dataset...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.training.dataset_builder",
             "--sany-only", "--include-augmented", "--include-description-sft",
             "--bugfix-oversample", "2"],
            cwd=str(_REPO_ROOT),
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            log.error(f"[retrain] Dataset rebuild failed: {result.stderr[-300:]}")
            return False
        log.info(f"[retrain] Dataset rebuilt. {result.stdout.strip().split(chr(10))[-1]}")
    except subprocess.TimeoutExpired:
        log.error("[retrain] Dataset rebuild timed out")
        return False

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
        return False
    num_epochs = 1
    log.info(f"[retrain] {n_train} training examples, {num_epochs} epochs")

    # 2. Train — use both GPUs (20B model needs ~40GB + activations; single 48GB GPU OOMs)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    night = is_nighttime()

    # Keep a safety margin for shared-machine usage.
    cap_ratio = GPU_VRAM_CAP_NIGHT if night else GPU_VRAM_CAP_DAY
    max_gpu_memory_mb = int(49152 * cap_ratio)

    est_steps = (n_train * num_epochs) // 8

    # Adapt max_length to available VRAM (avoids OOM on shared machines)
    free_mb = total_gpu_memory_free_mb()
    max_length = max_length_for_vram(free_mb)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train_cmd = [
        sys.executable, "-m", "src.training.train",
        "--epochs", str(num_epochs),
        "--max-gpu-memory-mb", str(max_gpu_memory_mb),
        "--max-length", str(max_length),
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
        return False
    log.info("[retrain] Training complete.")

    if _SHUTDOWN:
        return False

    # 3. Merge LoRA — use same GPU visibility as training; merge_lora used to
    # default to a single GPU and OOM / CUBLAS_ALLOC_FAILED on 20B + PEFT merge.
    merge_env = os.environ.copy()
    merge_env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
    merge_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log.info("[retrain] Merging LoRA weights (GPU)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.training.merge_lora"],
            cwd=str(_REPO_ROOT),
            env=merge_env,
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            err_tail = (result.stderr or "")[-800:]
            log.warning(
                f"[retrain] GPU merge failed ({err_tail[-400:]}); retrying on CPU (slow, ~RAM-heavy)..."
            )
            cpu_env = os.environ.copy()
            # Hide GPUs so merge runs purely on CPU; avoids CUBLAS init + VRAM.
            cpu_env["CUDA_VISIBLE_DEVICES"] = ""
            result = subprocess.run(
                [
                    sys.executable, "-m", "src.training.merge_lora",
                    "--device", "cpu",
                ],
                cwd=str(_REPO_ROOT),
                env=cpu_env,
                capture_output=True, text=True, timeout=7200,
            )
            if result.returncode != 0:
                log.error(f"[retrain] CPU merge also failed: {(result.stderr or '')[-500:]}")
                return False
            log.info("[retrain] LoRA merged (CPU fallback).")
        else:
            log.info("[retrain] LoRA merged.")
    except subprocess.TimeoutExpired:
        log.error("[retrain] Merge timed out")
        return False

    # 4. Convert to GGUF + register with Ollama
    log.info("[retrain] Converting to GGUF and deploying to Ollama...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.inference.convert_to_gguf", "--quant", "Q8_0"],
            cwd=str(_REPO_ROOT),
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            log.error(f"[retrain] GGUF conversion failed: {result.stderr[-300:]}")
            return False
        log.info("[retrain] GGUF deployed to Ollama.")
    except subprocess.TimeoutExpired:
        log.error("[retrain] GGUF conversion timed out")
        return False

    # HF publish is now handled by the caller after a full benchmark quality gate.
    return True


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
    for path in _glob.glob(str(_REPO_ROOT / "outputs" / "benchmark_results_*_full_*.csv")):
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
    output_csv = _REPO_ROOT / "outputs" / f"benchmark_results_rl_c{cycle_id}{lim}{suf}_{timestamp}.csv"

    scope = f"{limit}-problem quick eval" if limit else "full benchmark suite"
    log.info(f"[benchmark] Running {scope} (cycle {cycle_id}, attempts={attempts})...")
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
    allow_daytime_retrain: bool = False,
    publish_hf: bool = PUBLISH_HF_DEFAULT,
) -> tuple[CycleStats, int]:
    """
    Run one full RL cycle: generate → validate → build data → retrain → eval.

    Returns (stats, updated_accumulated_new).
    """
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
        random.shuffle(all_prompts)
        prompts = all_prompts[:max_prompts]
        stats.prompts_tried = len(prompts)

        log.info(f"[phase1] Generating and validating {len(prompts)} specs...")
        results = generate_and_validate(
            prompts,
            model="chattla:20b",
            max_attempts=2,
            prompt_cooldown_s=prompt_cooldown,
        )
        stats.specs_generated = len(results)
        stats.sany_pass = sum(1 for r in results if r.sany_pass)
        stats.tlc_pass = sum(1 for r in results if r.tlc_pass)
        stats.gold_count = sum(1 for r in results if r.tier == "gold")
        stats.silver_count = sum(1 for r in results if r.tier == "silver")
        stats.bronze_count = sum(1 for r in results if r.tier == "bronze")

        log.info(f"[phase1] Results: gold={stats.gold_count} silver={stats.silver_count} "
                 f"bronze={stats.bronze_count} sany={stats.sany_pass}/{stats.specs_generated} "
                 f"tlc={stats.tlc_pass}/{stats.specs_generated}")

        if _SHUTDOWN:
            stats.cycle_duration_s = time.time() - t0
            return stats, accumulated_new

        # ── Phase 2: Build training data ──────────────────────────────────
        log.info("[phase2] Building training data from results...")
        sft_examples, dpo_pairs = build_training_data(results)
        log.info(f"[phase2] Built {len(sft_examples)} SFT examples, {len(dpo_pairs)} DPO pairs "
                 f"(from {len(results)} results, {len(set(r.prompt_id for r in results))} prompts)")
        n_sft, n_dpo = persist_training_data(sft_examples, dpo_pairs)
        stats.new_train_examples = n_sft
        stats.new_dpo_pairs = n_dpo
        accumulated_new += n_sft

        log.info(f"[phase2] Persisted {n_sft} SFT examples, {n_dpo} DPO pairs. "
                 f"Accumulated: {accumulated_new}")

        if _SHUTDOWN:
            stats.cycle_duration_s = time.time() - t0
            return stats, accumulated_new

        # ── Phase 3: Retrain if threshold met ─────────────────────────────
        just_retrained = False
        if accumulated_new >= RETRAIN_THRESHOLD:
            if night or allow_daytime_retrain:
                when = "night" if night else "daytime (--allow-daytime-retrain)"
                log.info(f"[phase3] Retrain threshold reached ({when}) ({accumulated_new} >= {RETRAIN_THRESHOLD})")
                success = rebuild_and_retrain(cycle_id=cycle_id)
                stats.retrained = success
                stats.deployed = success
                if success:
                    accumulated_new = 0
                    just_retrained = True
                    log.info("[phase3] Retrain + deploy complete!")
                else:
                    log.warning("[phase3] Retrain skipped or failed. Will retry next cycle.")
            else:
                log.info(
                    f"[phase3] Threshold reached ({accumulated_new} >= {RETRAIN_THRESHOLD}) "
                    "but retrain deferred to nighttime. Use --allow-daytime-retrain to force."
                )
        else:
            log.info(f"[phase3] Skipping retrain ({accumulated_new}/{RETRAIN_THRESHOLD} examples accumulated)")

        if _SHUTDOWN:
            stats.cycle_duration_s = time.time() - t0
            return stats, accumulated_new

        # ── Phase 4: Full benchmark every cycle ─────────────────────────
        # Quick eval was pure noise (12 problems → 0-17% TLC when truth is 10%).
        # Full benchmark (20 problems, 3 attempts) is stable and the only
        # meaningful signal.  ~40 min extra per cycle is worth real data.
        full_sany, full_tlc = run_benchmark(
            cycle_id, limit=None, attempts=3, timeout_s=3600, suffix="full",
        )
        stats.benchmark_run = True
        stats.benchmark_sany_rate = full_sany
        stats.benchmark_tlc_rate = full_tlc

        if just_retrained and publish_hf:
            prev_best_tlc = best_historical_full_tlc()
            if full_tlc >= prev_best_tlc and full_tlc > 0:
                log.info(
                    f"[phase4] TLC {full_tlc:.0%} >= previous best {prev_best_tlc:.0%} — publishing to HF"
                )
                publish_to_hf(cycle_id)
            else:
                log.warning(
                    f"[phase4] TLC {full_tlc:.0%} < previous best {prev_best_tlc:.0%} (or zero) "
                    "— SKIPPING HF publish to avoid pushing a regression"
                )

    except Exception as e:
        stats.error = f"{type(e).__name__}: {str(e)[:200]}"
        log.error(f"[cycle {cycle_id}] Unhandled error: {e}")
        log.error(traceback.format_exc())

    stats.cycle_duration_s = time.time() - t0
    return stats, accumulated_new


def main():
    global RETRAIN_THRESHOLD

    import argparse

    _check_training_deps()

    parser = argparse.ArgumentParser(description="ChatTLA autonomous RL loop")
    parser.add_argument("--cycle-hours", type=float, default=CYCLE_HOURS,
                        help="Hours to wait after each cycle before starting the next (0 = no wait; "
                        f"default: {CYCLE_HOURS})")
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="Max cycles to run (0 = infinite)")
    parser.add_argument("--retrain-threshold", type=int, default=RETRAIN_THRESHOLD,
                        help=f"SFT examples before retrain (default: {RETRAIN_THRESHOLD})")
    parser.add_argument("--allow-daytime-retrain", action="store_true",
                        help="Retrain during daytime when threshold met (default: defer to night)")
    parser.add_argument("--model", default="chattla:20b")
    parser.add_argument("--no-publish-hf", action="store_true",
                        help="Skip Hugging Face Hub upload after retrain (requires HF_TOKEN when enabled)")
    # Legacy flags — accepted but ignored (full benchmark runs every cycle now).
    parser.add_argument("--quick-eval-limit", type=int, default=20, help=argparse.SUPPRESS)
    parser.add_argument("--quick-eval-attempts", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-every", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    RETRAIN_THRESHOLD = args.retrain_threshold

    cycle_seconds = max(0.0, args.cycle_hours * 3600)

    log.info("=" * 60)
    log.info("  ChatTLA Autonomous RL Loop")
    log.info(f"  Inter-cycle pause: {args.cycle_hours}h ({'none' if cycle_seconds <= 0 else f'{cycle_seconds/60:.0f} min target'})")
    log.info(f"  Eval: FULL benchmark (20 problems, 3 attempts) every cycle")
    log.info(f"  Retrain threshold: {args.retrain_threshold} | Min training size: {MIN_TRAIN_EXAMPLES}")
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
        stats, accumulated_new = run_cycle(
            cycle_id,
            accumulated_new,
            args.allow_daytime_retrain,
            publish_hf=not args.no_publish_hf,
        )
        accumulated_new = diagnose_and_fix(stats, accumulated_new)
        save_accumulated_new(accumulated_new)
        log_history(stats)

        cycle_elapsed = time.time() - cycle_start

        # Print cycle summary
        log.info(f"\n{'─'*60}")
        log.info(f"CYCLE {cycle_id} SUMMARY ({stats.cycle_duration_s/60:.1f} min)")
        log.info(f"  Specs: {stats.specs_generated} | SANY: {stats.sany_pass} | TLC: {stats.tlc_pass}")
        log.info(f"  Gold: {stats.gold_count} | Silver: {stats.silver_count} | Bronze: {stats.bronze_count}")
        log.info(f"  New SFT: {stats.new_train_examples} | New DPO: {stats.new_dpo_pairs}")
        log.info(f"  Retrained: {stats.retrained} | Deployed: {stats.deployed}")
        if stats.benchmark_run:
            log.info(f"  Benchmark: SANY={stats.benchmark_sany_rate:.0%} TLC={stats.benchmark_tlc_rate:.0%}")
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
