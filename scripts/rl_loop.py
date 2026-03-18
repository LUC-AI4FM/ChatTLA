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

Each cycle targets ~1.5 hours:
  - Generate + Validate: ~20 min
  - Retrain: ~60 min
  - Deploy + Eval: ~10 min
  - Cool-down sleep: remaining time to fill 1.5h

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

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_AUGMENTED_JSONL = _REPO_ROOT / "data" / "processed" / "augmented.jsonl"
_RL_DATA_DIR     = _REPO_ROOT / "data" / "processed" / "rl"
_DPO_JSONL       = _RL_DATA_DIR / "dpo_pairs.jsonl"
_RL_LOG_DIR      = _REPO_ROOT / "outputs" / "logs"
_RL_HISTORY      = _RL_LOG_DIR / "rl_history.jsonl"
_BENCHMARK_JSON  = _REPO_ROOT / "data" / "benchmarks" / "benchmark_suite.json"
_TRAIN_JSONL     = _REPO_ROOT / "data" / "processed" / "train.jsonl"
_EVAL_JSONL      = _REPO_ROOT / "data" / "processed" / "eval.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CYCLE_HOURS        = 1.5          # target hours per full cycle
RETRAIN_THRESHOLD  = 10           # new gold/silver examples before retrain
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
QUICK_EVAL_LIMIT = 6              # mini-eval every cycle
QUICK_EVAL_ATTEMPTS = 2

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


def log_history(stats: CycleStats):
    """Append cycle stats to rl_history.jsonl."""
    _RL_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(_RL_HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(stats), ensure_ascii=False) + "\n")


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

    # Benchmark suite
    if _BENCHMARK_JSON.exists():
        with open(_BENCHMARK_JSON) as f:
            benchmarks = json.load(f)
        for bm in benchmarks:
            if bm.get("difficulty", 1) <= difficulty_cap:
                prompts.append({
                    "id": bm["id"],
                    "prompt": bm["description"] + (f"\n\nHints: {bm['hints']}" if bm.get("hints") else ""),
                    "difficulty": bm.get("difficulty", 3),
                    "module_hint": bm["name"].replace(" ", "").replace("'", ""),
                })

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
def extract_tlc_feedback(tlc_result) -> str:
    """
    Parse TLC output to produce line-by-line, actionable feedback.
    This replaces TLAPS by extracting maximum diagnostic info from TLC.
    """
    raw = tlc_result.raw_output
    violations = tlc_result.tlc_violations
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
        best_result = None

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

            # Keep best result
            tier_rank = {"gold": 3, "silver": 2, "bronze": 1}
            if best_result is None or tier_rank.get(tier, 0) > tier_rank.get(best_result.tier, 0):
                best_result = result

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
                        if tier_rank.get(c_tlc.tier, 0) > tier_rank.get(best_result.tier, 0):
                            best_result = corrected_result
                        if c_tlc.tier == "gold":
                            break
                except Exception:
                    pass
                break  # self-correction already retries internally

        if best_result:
            results.append(best_result)
            log.info(f"  [{pid}] tier={best_result.tier} sany={best_result.sany_pass} "
                     f"tlc={best_result.tlc_pass} struct={best_result.structural_score:.2f} "
                     f"attempts={best_result.attempts}")

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

    SFT examples: gold/silver specs as positive training signal.
    DPO pairs: same prompt with a chosen (gold/silver) and rejected (bronze)
    spec, teaching the model to prefer correct output.
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

        # SFT: add gold and silver specs as positive examples
        if best.tier in ("gold", "silver"):
            sft_examples.append({
                "messages": [
                    {"role": "developer", "content": _DEVELOPER_PROMPT},
                    {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{best.prompt_text}"},
                    {"role": "assistant", "channel": "analysis", "content": "I'll write a well-formed TLA+ specification with proper Init, Next, and invariants."},
                    {"role": "assistant", "channel": "final", "content": best.spec.strip()},
                ],
            })

            # If there's also a bronze version, create a DPO pair
            worst = group[-1]
            if worst.tier == "bronze" and best.tier in ("gold", "silver"):
                # Build feedback from TLC violations
                feedback = ""
                if worst.tlc_violations:
                    feedback = "\n".join(worst.tlc_violations[:5])
                elif not worst.sany_pass:
                    feedback = "SANY parse errors (spec is syntactically invalid)"

                dpo_pairs.append({
                    "prompt": f"Write a TLA+ specification for the following:\n\n{best.prompt_text}",
                    "chosen": best.spec.strip(),
                    "rejected": worst.spec.strip(),
                    "chosen_tier": best.tier,
                    "rejected_tier": worst.tier,
                    "feedback": feedback,
                })

        # Error-conditioned training: for silver specs with TLC violations,
        # create bug_fix examples if we also have the gold version
        if best.tier == "gold":
            for r in group[1:]:
                if r.tier == "silver" and r.tlc_violations:
                    feedback = extract_tlc_feedback(r)
                    if feedback:
                        sft_examples.append({
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


def persist_training_data(sft_examples: list[dict], dpo_pairs: list[dict]) -> tuple[int, int]:
    """Append new training data to augmented.jsonl and dpo_pairs.jsonl."""
    _RL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Dedup against existing augmented data
    existing_keys = set()
    if _AUGMENTED_JSONL.exists():
        with open(_AUGMENTED_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        ex = json.loads(line)
                        msgs = ex.get("messages", [])
                        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                        existing_keys.add(user_msg[:200])
                    except (json.JSONDecodeError, KeyError):
                        pass

    n_sft = 0
    if sft_examples:
        with open(_AUGMENTED_JSONL, "a", encoding="utf-8") as f:
            for ex in sft_examples:
                user_msg = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
                key = user_msg[:200]
                if key not in existing_keys:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    existing_keys.add(key)
                    n_sft += 1

    n_dpo = 0
    if dpo_pairs:
        existing_dpo_keys = set()
        if _DPO_JSONL.exists():
            with open(_DPO_JSONL, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            p = json.loads(line)
                            existing_dpo_keys.add(p.get("prompt", "")[:200])
                        except (json.JSONDecodeError, KeyError):
                            pass

        with open(_DPO_JSONL, "a", encoding="utf-8") as f:
            for pair in dpo_pairs:
                key = pair["prompt"][:200]
                if key not in existing_dpo_keys:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    existing_dpo_keys.add(key)
                    n_dpo += 1

    return n_sft, n_dpo


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Rebuild dataset + Retrain
# ─────────────────────────────────────────────────────────────────────────────
def rebuild_and_retrain() -> bool:
    """Rebuild training dataset, retrain, merge, GGUF, deploy."""

    # 1. Rebuild dataset
    log.info("[retrain] Rebuilding training dataset...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.training.dataset_builder",
             "--sany-only", "--include-augmented"],
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

    # Count training examples for epoch scaling
    n_train = 0
    if _TRAIN_JSONL.exists():
        with open(_TRAIN_JSONL) as f:
            n_train = sum(1 for line in f if line.strip())
    num_epochs = max(3, min(10, 600 // max(n_train, 1)))
    log.info(f"[retrain] {n_train} training examples, {num_epochs} epochs")

    # 2. Train — use both GPUs at night, single GPU during day
    env = os.environ.copy()
    night = is_nighttime()
    if night:
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
    else:
        env["CUDA_VISIBLE_DEVICES"] = "1"

    # Keep a safety margin for shared-machine usage.
    cap_ratio = GPU_VRAM_CAP_NIGHT if night else GPU_VRAM_CAP_DAY
    max_gpu_memory_mb = int(49152 * cap_ratio)

    est_steps = (n_train * num_epochs) // 8
    timeout_s = max(3600, int(est_steps * 45 * 1.5))

    log.info(f"[retrain] Starting training (~{est_steps} steps, timeout={timeout_s}s)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.training.train",
             "--epochs", str(num_epochs),
             "--max-gpu-memory-mb", str(max_gpu_memory_mb)],
            cwd=str(_REPO_ROOT), env=env,
            capture_output=True, text=True, timeout=timeout_s,
        )
        if result.returncode != 0:
            log.error(f"[retrain] Training failed: {result.stderr[-500:]}")
            return False
        log.info("[retrain] Training complete.")
    except subprocess.TimeoutExpired:
        log.error("[retrain] Training timed out")
        return False

    if _SHUTDOWN:
        return False

    # 3. Merge LoRA
    log.info("[retrain] Merging LoRA weights...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.training.merge_lora"],
            cwd=str(_REPO_ROOT),
            capture_output=True, text=True, timeout=900,
        )
        if result.returncode != 0:
            log.error(f"[retrain] Merge failed: {result.stderr[-300:]}")
            return False
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

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Benchmark evaluation
# ─────────────────────────────────────────────────────────────────────────────
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
        return sany_rate, tlc_rate

    return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive difficulty
# ─────────────────────────────────────────────────────────────────────────────
def compute_difficulty_cap(history_path: Path = _RL_HISTORY) -> int:
    """
    Dynamically adjust prompt difficulty based on recent SANY pass rates.
    Start easy, progress to harder prompts as the model improves.
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

    # Average SANY rate over last 5 cycles
    last_n = recent[-5:]
    total_tried = sum(c.get("prompts_tried", 1) for c in last_n)
    total_sany = sum(c.get("sany_pass", 0) for c in last_n)
    avg_sany = total_sany / max(total_tried, 1)

    if avg_sany >= 0.8:
        return 5  # model is strong, give it hard problems
    elif avg_sany >= 0.6:
        return 4
    elif avg_sany >= 0.4:
        return 3
    else:
        return 2  # model struggling, focus on easy wins


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def run_cycle(cycle_id: int, accumulated_new: int) -> tuple[CycleStats, int]:
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
        if accumulated_new >= RETRAIN_THRESHOLD:
            if night:
                log.info(f"[phase3] Retrain threshold reached at night ({accumulated_new} >= {RETRAIN_THRESHOLD})")
                success = rebuild_and_retrain()
                stats.retrained = success
                stats.deployed = success
                if success:
                    accumulated_new = 0
                    log.info("[phase3] Retrain + deploy complete!")
                else:
                    log.warning("[phase3] Retrain failed. Will retry next cycle.")
            else:
                log.info(
                    f"[phase3] Threshold reached ({accumulated_new} >= {RETRAIN_THRESHOLD}) "
                    "but retrain deferred to nighttime to prioritize shared daytime capacity."
                )
        else:
            log.info(f"[phase3] Skipping retrain ({accumulated_new}/{RETRAIN_THRESHOLD} examples accumulated)")

        if _SHUTDOWN:
            stats.cycle_duration_s = time.time() - t0
            return stats, accumulated_new

        # ── Phase 4: Evaluation every cycle + periodic full benchmark ─────
        # Every 1.5h cycle runs a quick eval so quality is continuously tracked.
        quick_sany, quick_tlc = run_benchmark(
            cycle_id,
            limit=QUICK_EVAL_LIMIT,
            attempts=QUICK_EVAL_ATTEMPTS,
            timeout_s=1200,
            suffix="quick",
        )
        stats.benchmark_run = True
        stats.benchmark_sany_rate = quick_sany
        stats.benchmark_tlc_rate = quick_tlc

        # Full 20-problem eval periodically, preferentially at night.
        if cycle_id % BENCHMARK_EVERY_N == 0:
            if night:
                full_sany, full_tlc = run_benchmark(cycle_id, limit=None, attempts=3, timeout_s=3600, suffix="full")
                stats.benchmark_sany_rate = full_sany
                stats.benchmark_tlc_rate = full_tlc
            else:
                log.info("[phase4] Full benchmark deferred to nighttime; quick eval already completed.")

    except Exception as e:
        stats.error = f"{type(e).__name__}: {str(e)[:200]}"
        log.error(f"[cycle {cycle_id}] Unhandled error: {e}")
        log.error(traceback.format_exc())

    stats.cycle_duration_s = time.time() - t0
    return stats, accumulated_new


def main():
    global RETRAIN_THRESHOLD, BENCHMARK_EVERY_N

    import argparse

    parser = argparse.ArgumentParser(description="ChatTLA autonomous RL loop")
    parser.add_argument("--cycle-hours", type=float, default=CYCLE_HOURS,
                        help=f"Target hours per cycle (default: {CYCLE_HOURS})")
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="Max cycles to run (0 = infinite)")
    parser.add_argument("--retrain-threshold", type=int, default=RETRAIN_THRESHOLD,
                        help=f"SFT examples before retrain (default: {RETRAIN_THRESHOLD})")
    parser.add_argument("--model", default="chattla:20b")
    parser.add_argument("--benchmark-every", type=int, default=BENCHMARK_EVERY_N,
                        help=f"Benchmark every N cycles (default: {BENCHMARK_EVERY_N})")
    args = parser.parse_args()

    RETRAIN_THRESHOLD = args.retrain_threshold
    BENCHMARK_EVERY_N = args.benchmark_every

    cycle_seconds = args.cycle_hours * 3600

    log.info("=" * 60)
    log.info("  ChatTLA Autonomous RL Loop")
    log.info(f"  Cycle target: {args.cycle_hours}h | Retrain threshold: {args.retrain_threshold}")
    log.info(f"  Benchmark every {args.benchmark_every} cycles")
    log.info(f"  Max cycles: {'infinite' if args.max_cycles == 0 else args.max_cycles}")
    log.info(f"  PID: {os.getpid()}")
    log.info("=" * 60)

    cycle_id = 0
    accumulated_new = 0

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
        stats, accumulated_new = run_cycle(cycle_id, accumulated_new)
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

        # Sleep to fill the target cycle time
        remaining = cycle_seconds - cycle_elapsed
        if remaining > 60 and not _SHUTDOWN:
            # If it's transitioning to/from nighttime, adjust sleep
            sleep_time = max(60, remaining)
            log.info(f"Sleeping {sleep_time/60:.1f} min until next cycle...")
            # Sleep in chunks to allow graceful shutdown
            sleep_end = time.time() + sleep_time
            while time.time() < sleep_end and not _SHUTDOWN:
                time.sleep(min(30, sleep_end - time.time()))
        elif not _SHUTDOWN:
            # Cycle took longer than target — brief cooldown
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
