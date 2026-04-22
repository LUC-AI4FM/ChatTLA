#!/usr/bin/env python3
"""
rl_tight_loop.py — Fast, reproducible RL loop for TLA+ spec generation.

Key differences from rl_loop.py:
  1. ACTUALLY RETRAINS — low threshold (5 DPO pairs) triggers training
  2. REPRODUCIBLE — explicit seeds, config versioning, git commit tracking
  3. FAST CYCLES — target 1-2 hours per cycle, not 4-6
  4. VERIFIABLE REWARDS — TLC pass = ground truth, no LLM judge

The loop:
  1. Generate: N specs at temperature T from current model
  2. Validate: SANY + TLC (binary reward: gold=1, silver=0.3, bronze=0)
  3. Harvest: Create DPO pairs (gold vs worse for same prompt)
  4. Train: DPO when pairs >= threshold (default 5)
  5. Evaluate: Fixed holdout set, track metrics over time
  6. Loop: Increment seed, repeat

Designed for tmux: `tmux new -s rl_tight && python scripts/rl_tight_loop.py`

Usage:
    python scripts/rl_tight_loop.py --cycles 10          # 10 cycles
    python scripts/rl_tight_loop.py --smoke              # quick test
    python scripts/rl_tight_loop.py --resume             # continue from state
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — EDIT THESE FOR YOUR EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    """Experiment configuration — all hyperparameters in one place."""
    # Experiment identity
    experiment_name: str = "rl_tight_v1"
    base_model_path: str = str(_REPO_ROOT / "outputs" / "merged_model_dpo_piecewise")
    seed_start: int = 42
    
    # Generation
    prompts_per_cycle: int = 15
    attempts_per_prompt: int = 3
    temperature: float = 0.4
    max_tokens: int = 2048
    ollama_model: str = "chattla:20b-fork-a-tlc-v2"  # Latest stable model
    
    # Training thresholds (MUCH LOWER than original)
    dpo_train_threshold: int = 5       # Train after 5 new DPO pairs
    sft_train_threshold: int = 10      # Or 10 new gold specs
    max_dpo_pairs_per_train: int = 50  # Cap to prevent overfitting
    
    # DPO hyperparameters
    dpo_lr: float = 5e-6
    dpo_beta: float = 0.1
    dpo_epochs: int = 1
    dpo_batch_size: int = 1
    dpo_grad_accum: int = 2
    dpo_max_length: int = 2048
    
    # Evaluation
    holdout_size: int = 20
    eval_every_n_cycles: int = 1
    
    # Resource management
    max_cycle_hours: float = 2.0
    cooldown_between_cycles_s: int = 30
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def fingerprint(self) -> str:
        """Hash of config for reproducibility tracking."""
        return hashlib.md5(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_DATA_DIR = _REPO_ROOT / "data" / "processed" / "rl_tight"
_LOG_DIR = _REPO_ROOT / "outputs" / "logs" / "rl_tight"
_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "checkpoints_rl_tight"
_STATE_FILE = _DATA_DIR / "state.json"
_HISTORY_FILE = _LOG_DIR / "history.jsonl"
_DPO_PAIRS_FILE = _DATA_DIR / "dpo_pairs.jsonl"
_GOLD_SPECS_FILE = _DATA_DIR / "gold_specs.jsonl"
_CONFIG_FILE = _DATA_DIR / "config.json"

for d in [_DATA_DIR, _LOG_DIR, _CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
import logging

# Stream only: launcher redirects stdout to rl_tight.log (avoid FileHandler+tee duplicates).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("rl_tight")


# ─────────────────────────────────────────────────────────────────────────────
# State management
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LoopState:
    """Persisted state across restarts."""
    cycle: int = 0
    total_dpo_pairs: int = 0
    total_gold_specs: int = 0
    total_trains: int = 0
    best_tlc_rate: float = 0.0
    best_checkpoint: str = ""
    current_seed: int = 42
    config_fingerprint: str = ""
    git_commit: str = ""
    
    # Accumulated since last train
    new_dpo_pairs: int = 0
    new_gold_specs: int = 0


@dataclass 
class CycleMetrics:
    """Metrics for a single cycle."""
    cycle: int = 0
    timestamp: str = ""
    seed: int = 0
    
    # Generation
    prompts_tried: int = 0
    specs_generated: int = 0
    
    # Validation results
    gold: int = 0
    silver: int = 0
    bronze: int = 0
    sany_rate: float = 0.0
    tlc_rate: float = 0.0
    
    # Training
    dpo_pairs_created: int = 0
    trained: bool = False
    train_loss: float = 0.0
    
    # Evaluation
    holdout_sany_rate: float = 0.0
    holdout_tlc_rate: float = 0.0
    
    # Timing
    duration_s: float = 0.0
    error: str = ""


def load_state() -> LoopState:
    if _STATE_FILE.exists():
        try:
            with open(_STATE_FILE) as f:
                data = json.load(f)
            return LoopState(**data)
        except Exception as e:
            log.warning(f"Could not load state: {e}")
    return LoopState()


def save_state(state: LoopState) -> None:
    with open(_STATE_FILE, "w") as f:
        json.dump(asdict(state), f, indent=2)


def log_metrics(metrics: CycleMetrics) -> None:
    with open(_HISTORY_FILE, "a") as f:
        f.write(json.dumps(asdict(metrics)) + "\n")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ─────────────────────────────────────────────────────────────────────────────
_SHUTDOWN = False

def _signal_handler(signum, frame):
    global _SHUTDOWN
    log.info(f"Received signal {signum}. Will shutdown after current phase.")
    _SHUTDOWN = True

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Generate specs
# ─────────────────────────────────────────────────────────────────────────────
def load_prompts(n: int, seed: int) -> list[dict]:
    """Load N prompts from benchmark suite, deterministically shuffled."""
    from src.rlvr_canary.fullspec_dataset import load_fullspec_prompts
    
    prompts = load_fullspec_prompts(include_topics=True, include_diamond_sft=True)
    if not prompts:
        # Fallback: load from benchmark_suite.json
        bench_file = _REPO_ROOT / "data" / "benchmarks" / "benchmark_suite.json"
        if bench_file.exists():
            with open(bench_file) as f:
                data = json.load(f)
            prompts = [
                type('Prompt', (), {
                    'prompt_id': p.get('benchmark_id', f'b{i}'),
                    'module_name': p.get('module_name', 'Spec'),
                    'nl_description': p.get('nl_description', p.get('description', '')),
                    'domain': p.get('domain', 'general'),
                })()
                for i, p in enumerate(data.get('benchmarks', []))
            ]
    
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:n]


def generate_spec(prompt: str, cfg: Config) -> str:
    """Generate a single spec via Ollama."""
    import urllib.request
    
    system = (
        "You are ChatTLA, an expert at writing verified TLA+ formal specifications.\n"
        "Write a complete, valid TLA+ spec that passes SANY syntax checking and TLC model checking.\n"
        "Output ONLY the TLA+ code — no markdown fences, no explanation, no comments about the spec.\n"
        "Start with ---- MODULE ModuleName ---- and end with ====.\n"
    )
    
    full_prompt = f"{system}\n\nWrite a TLA+ specification for:\n{prompt}"
    
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({
                "model": cfg.ollama_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": cfg.temperature,
                    "num_predict": cfg.max_tokens,
                    "seed": cfg.seed_start,  # Use config seed for reproducibility
                },
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception as e:
        log.warning(f"Generation failed: {e}")
        return ""


def phase_generate(cfg: Config, seed: int) -> list[dict]:
    """Generate specs for prompts."""
    prompts = load_prompts(cfg.prompts_per_cycle, seed)
    log.info(f"Generating specs for {len(prompts)} prompts, {cfg.attempts_per_prompt} attempts each")
    
    results = []
    for pi, p in enumerate(prompts):
        nl = p.nl_description[:1000] if hasattr(p, 'nl_description') else str(p)
        prompt_id = p.prompt_id if hasattr(p, 'prompt_id') else f"p{pi}"
        module_name = p.module_name if hasattr(p, 'module_name') else "Spec"
        
        for attempt in range(cfg.attempts_per_prompt):
            spec = generate_spec(nl, cfg)
            if spec:
                results.append({
                    "prompt_id": prompt_id,
                    "prompt_text": nl,
                    "module_name": module_name,
                    "spec": spec,
                    "attempt": attempt,
                    "seed": seed,
                })
        
        if (pi + 1) % 5 == 0:
            log.info(f"  Generated {pi+1}/{len(prompts)} prompts, {len(results)} specs total")
    
    log.info(f"Phase 1 complete: {len(results)} specs from {len(prompts)} prompts")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Validate specs
# ─────────────────────────────────────────────────────────────────────────────
def validate_spec(spec: str, module_name: str) -> dict:
    """Validate spec with SANY and TLC, return tier and reward."""
    from src.validators.sany_validator import validate_string as sany_validate
    from src.validators.tlc_validator import validate_string as tlc_validate
    
    result = {
        "sany_ok": False,
        "tlc_ok": False,
        "tier": "bronze",
        "reward": 0.0,
        "error": "",
    }
    
    # SANY check
    try:
        sany_result = sany_validate(spec, module_name=module_name)
        result["sany_ok"] = sany_result.valid
        if not sany_result.valid:
            result["error"] = f"SANY: {sany_result.error[:200] if sany_result.error else 'parse error'}"
            return result
    except Exception as e:
        result["error"] = f"SANY exception: {str(e)[:100]}"
        return result
    
    # TLC check
    try:
        tlc_result = tlc_validate(spec, module_name=module_name, timeout=30)
        result["tlc_ok"] = tlc_result.valid
        if tlc_result.valid:
            result["tier"] = "gold"
            result["reward"] = 1.0
        else:
            result["tier"] = "silver"
            result["reward"] = 0.3
            result["error"] = f"TLC: {tlc_result.error[:200] if tlc_result.error else 'model check failed'}"
    except Exception as e:
        result["tier"] = "silver"
        result["reward"] = 0.3
        result["error"] = f"TLC exception: {str(e)[:100]}"
    
    return result


def phase_validate(specs: list[dict]) -> list[dict]:
    """Validate all specs and assign tiers."""
    log.info(f"Validating {len(specs)} specs...")
    
    for i, spec_data in enumerate(specs):
        validation = validate_spec(spec_data["spec"], spec_data["module_name"])
        spec_data.update(validation)
        
        if (i + 1) % 10 == 0:
            gold = sum(1 for s in specs[:i+1] if s.get("tier") == "gold")
            log.info(f"  Validated {i+1}/{len(specs)}, {gold} gold so far")
    
    gold = sum(1 for s in specs if s.get("tier") == "gold")
    silver = sum(1 for s in specs if s.get("tier") == "silver")
    bronze = sum(1 for s in specs if s.get("tier") == "bronze")
    log.info(f"Phase 2 complete: {gold} gold, {silver} silver, {bronze} bronze")
    
    return specs


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Harvest DPO pairs
# ─────────────────────────────────────────────────────────────────────────────
def phase_harvest(specs: list[dict]) -> tuple[int, int]:
    """Create DPO pairs from validated specs and save gold specs."""
    # Group by prompt_id
    by_prompt: dict[str, list[dict]] = {}
    for spec in specs:
        pid = spec["prompt_id"]
        by_prompt.setdefault(pid, []).append(spec)
    
    new_dpo_pairs = 0
    new_gold_specs = 0
    
    # Load existing prompt IDs to avoid duplicates
    existing_prompts = set()
    if _DPO_PAIRS_FILE.exists():
        with open(_DPO_PAIRS_FILE) as f:
            for line in f:
                try:
                    existing_prompts.add(json.loads(line).get("prompt_id", ""))
                except Exception:
                    pass
    
    with open(_DPO_PAIRS_FILE, "a") as dpo_f, open(_GOLD_SPECS_FILE, "a") as gold_f:
        for pid, candidates in by_prompt.items():
            # Sort by tier (gold > silver > bronze) then by reward
            tier_order = {"gold": 3, "silver": 2, "bronze": 1}
            candidates.sort(key=lambda x: (tier_order.get(x.get("tier"), 0), x.get("reward", 0)), reverse=True)
            
            best = candidates[0]
            worst = candidates[-1] if len(candidates) > 1 else None
            
            # Save gold specs
            if best.get("tier") == "gold":
                gold_f.write(json.dumps({
                    "prompt_id": pid,
                    "prompt": best["prompt_text"],
                    "spec": best["spec"],
                    "module_name": best["module_name"],
                }) + "\n")
                new_gold_specs += 1
            
            # Create DPO pair if we have gold vs worse
            if (best.get("tier") == "gold" and worst and 
                worst.get("tier") in ("silver", "bronze") and
                pid not in existing_prompts):
                dpo_f.write(json.dumps({
                    "prompt_id": pid,
                    "prompt": best["prompt_text"],
                    "chosen": best["spec"],
                    "rejected": worst["spec"],
                    "chosen_tier": best["tier"],
                    "rejected_tier": worst["tier"],
                    "feedback": worst.get("error", ""),
                }) + "\n")
                new_dpo_pairs += 1
    
    log.info(f"Phase 3 complete: {new_dpo_pairs} new DPO pairs, {new_gold_specs} new gold specs")
    return new_dpo_pairs, new_gold_specs


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Train
# ─────────────────────────────────────────────────────────────────────────────
def load_dpo_pairs(max_pairs: int) -> list[dict]:
    """Load DPO pairs for training."""
    pairs = []
    if _DPO_PAIRS_FILE.exists():
        with open(_DPO_PAIRS_FILE) as f:
            for line in f:
                try:
                    pairs.append(json.loads(line))
                except Exception:
                    pass
    
    # Take most recent pairs (they're from better model versions)
    return pairs[-max_pairs:]


def phase_train(cfg: Config, state: LoopState) -> tuple[bool, float]:
    """Run DPO training if threshold met."""
    if state.new_dpo_pairs < cfg.dpo_train_threshold:
        log.info(f"Skip training: {state.new_dpo_pairs} < {cfg.dpo_train_threshold} DPO pairs")
        return False, 0.0
    
    pairs = load_dpo_pairs(cfg.max_dpo_pairs_per_train)
    if len(pairs) < 2:
        log.info("Skip training: need at least 2 DPO pairs")
        return False, 0.0
    
    log.info(f"Starting DPO training on {len(pairs)} pairs...")
    
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
        
        # Format pairs for DPO
        formatted = []
        for p in pairs:
            formatted.append({
                "prompt": f"Write a TLA+ specification for:\n{p['prompt']}",
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            })
        ds = Dataset.from_list(formatted)
        
        # Load model
        log.info(f"Loading base model from {cfg.base_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # LoRA config for DPO
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # DPO config
        ckpt_dir = _CHECKPOINT_DIR / f"dpo_cycle_{state.cycle}"
        dpo_config = DPOConfig(
            output_dir=str(ckpt_dir),
            per_device_train_batch_size=cfg.dpo_batch_size,
            gradient_accumulation_steps=cfg.dpo_grad_accum,
            learning_rate=cfg.dpo_lr,
            lr_scheduler_type="constant",
            warmup_ratio=0.0,
            num_train_epochs=cfg.dpo_epochs,
            beta=cfg.dpo_beta,
            max_length=cfg.dpo_max_length,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
        )
        
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config,
            train_dataset=ds,
            processing_class=tokenizer,
            peft_config=lora_config,
        )
        
        train_result = trainer.train()
        loss = train_result.training_loss
        
        # Save checkpoint
        trainer.save_model(str(ckpt_dir / "final"))
        tokenizer.save_pretrained(str(ckpt_dir / "final"))
        
        log.info(f"Phase 4 complete: DPO training done, loss={loss:.4f}")
        return True, loss
        
    except Exception as e:
        log.error(f"Training failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Evaluate
# ─────────────────────────────────────────────────────────────────────────────
def phase_evaluate(cfg: Config) -> tuple[float, float]:
    """Run holdout evaluation."""
    log.info(f"Running holdout evaluation ({cfg.holdout_size} specs)...")
    
    # Load holdout prompts
    holdout_file = _REPO_ROOT / "data" / "processed" / "diamond_eval_holdout.jsonl"
    prompts = []
    if holdout_file.exists():
        with open(holdout_file) as f:
            for line in f:
                try:
                    prompts.append(json.loads(line))
                except Exception:
                    pass
    
    if not prompts:
        log.warning("No holdout prompts found, skipping eval")
        return 0.0, 0.0
    
    prompts = prompts[:cfg.holdout_size]
    
    sany_pass = 0
    tlc_pass = 0
    
    for p in prompts:
        nl = p.get("nl_description", p.get("description", ""))[:800]
        module = p.get("module_name", "Spec")
        
        spec = generate_spec(nl, cfg)
        if not spec:
            continue
        
        validation = validate_spec(spec, module)
        if validation["sany_ok"]:
            sany_pass += 1
        if validation["tlc_ok"]:
            tlc_pass += 1
    
    sany_rate = sany_pass / len(prompts) if prompts else 0.0
    tlc_rate = tlc_pass / len(prompts) if prompts else 0.0
    
    log.info(f"Phase 5 complete: SANY={sany_rate:.1%}, TLC={tlc_rate:.1%}")
    return sany_rate, tlc_rate


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def run_cycle(cfg: Config, state: LoopState) -> CycleMetrics:
    """Run a single RL cycle."""
    start_time = time.time()
    state.cycle += 1
    seed = cfg.seed_start + state.cycle
    state.current_seed = seed
    
    metrics = CycleMetrics(
        cycle=state.cycle,
        timestamp=datetime.datetime.now().isoformat(),
        seed=seed,
    )
    
    log.info(f"═══ Cycle {state.cycle} (seed={seed}) ═══")
    
    try:
        # Phase 1: Generate
        specs = phase_generate(cfg, seed)
        metrics.specs_generated = len(specs)
        metrics.prompts_tried = cfg.prompts_per_cycle
        
        if _SHUTDOWN:
            return metrics
        
        # Phase 2: Validate
        specs = phase_validate(specs)
        metrics.gold = sum(1 for s in specs if s.get("tier") == "gold")
        metrics.silver = sum(1 for s in specs if s.get("tier") == "silver")
        metrics.bronze = sum(1 for s in specs if s.get("tier") == "bronze")
        metrics.sany_rate = (metrics.gold + metrics.silver) / max(1, len(specs))
        metrics.tlc_rate = metrics.gold / max(1, len(specs))
        
        if _SHUTDOWN:
            return metrics
        
        # Phase 3: Harvest
        new_dpo, new_gold = phase_harvest(specs)
        metrics.dpo_pairs_created = new_dpo
        state.new_dpo_pairs += new_dpo
        state.new_gold_specs += new_gold
        state.total_dpo_pairs += new_dpo
        state.total_gold_specs += new_gold
        
        if _SHUTDOWN:
            return metrics
        
        # Phase 4: Train (if threshold met)
        trained, loss = phase_train(cfg, state)
        metrics.trained = trained
        metrics.train_loss = loss
        if trained:
            state.total_trains += 1
            state.new_dpo_pairs = 0  # Reset accumulator
            state.new_gold_specs = 0
        
        if _SHUTDOWN:
            return metrics
        
        # Phase 5: Evaluate
        if state.cycle % cfg.eval_every_n_cycles == 0:
            sany_rate, tlc_rate = phase_evaluate(cfg)
            metrics.holdout_sany_rate = sany_rate
            metrics.holdout_tlc_rate = tlc_rate
            
            if tlc_rate > state.best_tlc_rate:
                state.best_tlc_rate = tlc_rate
                state.best_checkpoint = str(_CHECKPOINT_DIR / f"dpo_cycle_{state.cycle}")
                log.info(f"🎯 New best TLC rate: {tlc_rate:.1%}")
        
    except Exception as e:
        metrics.error = str(e)
        log.error(f"Cycle error: {e}")
        import traceback
        log.error(traceback.format_exc())
    
    metrics.duration_s = time.time() - start_time
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Tight RL loop for TLA+ generation")
    parser.add_argument("--cycles", type=int, default=100, help="Number of cycles to run")
    parser.add_argument("--smoke", action="store_true", help="Quick test (1 cycle, 3 prompts)")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    parser.add_argument("--reset", action="store_true", help="Reset state and start fresh")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()
    
    # Configuration
    cfg = Config()
    if args.smoke:
        cfg.prompts_per_cycle = 3
        cfg.attempts_per_prompt = 2
        cfg.holdout_size = 5
        cfg.dpo_train_threshold = 2
    
    # State
    if args.reset and _STATE_FILE.exists():
        _STATE_FILE.unlink()
        log.info("State reset")
    
    state = load_state() if args.resume or _STATE_FILE.exists() else LoopState()
    state.config_fingerprint = cfg.fingerprint()
    state.git_commit = get_git_commit()
    
    # Save config
    with open(_CONFIG_FILE, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    
    log.info("═══════════════════════════════════════════════════════════════")
    log.info(f"RL Tight Loop — {cfg.experiment_name}")
    log.info(f"Config: {cfg.fingerprint()}, Git: {state.git_commit}")
    log.info(f"Base model: {cfg.base_model_path}")
    log.info(f"Resuming from cycle {state.cycle}, {state.total_trains} trains done")
    log.info("═══════════════════════════════════════════════════════════════")
    
    if args.eval_only:
        sany, tlc = phase_evaluate(cfg)
        print(f"Evaluation: SANY={sany:.1%}, TLC={tlc:.1%}")
        return
    
    cycles_to_run = 1 if args.smoke else args.cycles
    
    for i in range(cycles_to_run):
        if _SHUTDOWN:
            log.info("Shutdown requested, exiting gracefully")
            break
        
        metrics = run_cycle(cfg, state)
        log_metrics(metrics)
        save_state(state)
        
        # Summary
        log.info(f"Cycle {state.cycle} summary: "
                 f"gold={metrics.gold}, trained={metrics.trained}, "
                 f"TLC={metrics.tlc_rate:.1%}, holdout_TLC={metrics.holdout_tlc_rate:.1%}, "
                 f"duration={metrics.duration_s/60:.1f}min")
        
        if i < cycles_to_run - 1 and not _SHUTDOWN:
            log.info(f"Cooldown {cfg.cooldown_between_cycles_s}s before next cycle...")
            time.sleep(cfg.cooldown_between_cycles_s)
    
    log.info("═══════════════════════════════════════════════════════════════")
    log.info(f"Loop complete. {state.cycle} cycles, {state.total_trains} trains")
    log.info(f"Best TLC rate: {state.best_tlc_rate:.1%} at {state.best_checkpoint}")
    log.info("═══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
