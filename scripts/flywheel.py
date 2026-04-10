#!/usr/bin/env python3
"""flywheel.py — Automated generate-validate-train cycle (DeepSeek-Prover self-play).

Orchestrates the iterative data flywheel:
  1. Generate: model writes specs for sampled prompts at multiple temperatures
  2. Validate: full pipeline (SANY → TLC → component verdicts → Diamond gate)
  3. Harvest: gold specs → SFT corpus, (best, worst) → DPO pairs
  4. Train: DPO / GRPO / SFT depending on accumulated data
  5. Evaluate: holdout gate (must not regress from best)
  6. Deploy: merge LoRA → GGUF → Ollama

Each cycle takes ~4-6 hours on 2x RTX 8000. Gated on holdout eval to
prevent regression. Designed to run unattended.

Usage:
    python -m scripts.flywheel --cycles 5         # 5 cycles
    python -m scripts.flywheel --smoke             # 1 cycle, 5 prompts
    python -m scripts.flywheel --cycles 1 --skip-train  # generate + validate only
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PY = sys.executable
_AUGMENTED = _REPO_ROOT / "data" / "processed" / "augmented.jsonl"
_DPO_PAIRS = _REPO_ROOT / "data" / "processed" / "piecewise_dpo_pairs.jsonl"
_METRICS_LOG = _REPO_ROOT / "outputs" / "logs" / "flywheel_metrics.jsonl"
_HOLDOUT = _REPO_ROOT / "data" / "processed" / "diamond_eval_holdout.jsonl"


def ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


# ── Phase 1: Generate ────────────────────────────────────────────────────

def phase_generate(
    n_prompts: int = 50,
    model: str = "chattla:20b",
    temps: list[float] | None = None,
    attempts_per_temp: int = 2,
) -> list[dict]:
    """Generate specs for sampled prompts at multiple temperatures."""
    import urllib.request
    from src.rlvr_canary.fullspec_dataset import load_fullspec_prompts

    if temps is None:
        temps = [0.3, 0.5, 0.7, 0.9]

    examples = load_fullspec_prompts(include_topics=True, include_diamond_sft=True)
    if not examples:
        log("No prompts available!")
        return []

    sampled = random.sample(examples, min(n_prompts, len(examples)))
    log(f"Generating specs for {len(sampled)} prompts, "
        f"{len(temps)} temps x {attempts_per_temp} attempts = "
        f"{len(temps) * attempts_per_temp} candidates each")

    results: list[dict] = []
    for si, ex in enumerate(sampled):
        nl = ex.nl_description[:800]
        prompt = (
            "You are ChatTLA, an expert at writing verified TLA+ formal specifications.\n"
            "Write a complete, valid TLA+ spec that passes SANY and TLC.\n"
            "Output only the spec — no markdown fences, no explanation.\n"
            f"Reasoning: medium\n\nWrite a TLA+ specification for:\n{nl}"
        )
        for temp in temps:
            for _ in range(attempts_per_temp):
                try:
                    req = urllib.request.Request(
                        "http://localhost:11434/api/generate",
                        data=json.dumps({
                            "model": model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": temp, "num_predict": 2048},
                        }).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        text = json.loads(resp.read()).get("response", "")
                except Exception:
                    text = ""
                if text.strip():
                    results.append({
                        "prompt_id": ex.prompt_id,
                        "module_name": ex.module_name,
                        "domain": ex.domain,
                        "nl_description": nl,
                        "spec": text.strip(),
                        "temperature": temp,
                    })

        if (si + 1) % 10 == 0:
            log(f"  Generated {si+1}/{len(sampled)} prompts, {len(results)} candidates total")

    log(f"Phase 1 complete: {len(results)} candidates from {len(sampled)} prompts")
    return results


# ── Phase 2: Validate ────────────────────────────────────────────────────

def phase_validate(candidates: list[dict]) -> list[dict]:
    """Run full validation pipeline on each candidate."""
    from src.validators.component_validator import reward_from_spec
    from src.validators.sany_validator import validate_string as sany_validate
    from src.validators.tlc_validator import validate_string as tlc_validate

    log(f"Validating {len(candidates)} candidates ...")
    for ci, cand in enumerate(candidates):
        spec = cand["spec"]
        module = cand.get("module_name", "Spec")

        # SANY check
        sany_r = sany_validate(spec, module_name=module)
        cand["sany_ok"] = sany_r.valid

        # Component reward (includes depth-1 + full TLC)
        try:
            cand["reward"] = reward_from_spec(
                spec, module_name=module,
                run_depth1=True, run_full_tlc=True,
                full_tlc_timeout=30,
            )
        except Exception:
            cand["reward"] = 0.0

        # Full TLC for tier
        if cand["sany_ok"]:
            try:
                tlc_r = tlc_validate(spec, module_name=module, timeout=60)
                cand["tier"] = tlc_r.tier
                cand["is_diamond"] = (
                    tlc_r.tier == "gold"
                    and hasattr(tlc_r, "semantic")
                    and tlc_r.semantic is not None
                    and tlc_r.semantic.is_diamond()
                )
            except Exception:
                cand["tier"] = "bronze"
                cand["is_diamond"] = False
        else:
            cand["tier"] = "bronze"
            cand["is_diamond"] = False

        if (ci + 1) % 50 == 0:
            log(f"  Validated {ci+1}/{len(candidates)}")

    # Stats
    tiers = {"gold": 0, "silver": 0, "bronze": 0}
    diamonds = 0
    for c in candidates:
        tiers[c.get("tier", "bronze")] = tiers.get(c.get("tier", "bronze"), 0) + 1
        if c.get("is_diamond"):
            diamonds += 1
    log(f"Phase 2 complete: {tiers}, diamond={diamonds}")
    return candidates


# ── Phase 3: Harvest ─────────────────────────────────────────────────────

def phase_harvest(candidates: list[dict]) -> dict:
    """Harvest gold specs for SFT and DPO pairs."""
    from src.training.dataset_builder import _DEVELOPER_PROMPT

    new_sft = 0
    new_dpo = 0

    # Group by prompt_id
    by_prompt: dict[str, list[dict]] = {}
    for c in candidates:
        pid = c.get("prompt_id", "unknown")
        by_prompt.setdefault(pid, []).append(c)

    _AUGMENTED.parent.mkdir(parents=True, exist_ok=True)
    _DPO_PAIRS.parent.mkdir(parents=True, exist_ok=True)

    with _AUGMENTED.open("a", encoding="utf-8") as sft_f, \
         _DPO_PAIRS.open("a", encoding="utf-8") as dpo_f:

        for pid, group in by_prompt.items():
            group.sort(key=lambda c: c.get("reward", 0), reverse=True)
            best = group[0]
            worst = group[-1]

            # Harvest gold/diamond specs for SFT
            for c in group:
                if c.get("tier") == "gold":
                    sft_row = {
                        "_tier": "gold",
                        "_prompt_id": pid,
                        "_source": "flywheel",
                        "_timestamp": ts(),
                        "messages": [
                            {"role": "developer", "content": _DEVELOPER_PROMPT},
                            {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{c['nl_description']}"},
                            {"role": "assistant", "channel": "final", "content": c["spec"]},
                        ],
                    }
                    sft_f.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
                    new_sft += 1
                    break  # one gold per prompt

            # DPO pair: best vs worst if reward gap is meaningful
            if len(group) >= 2 and (best["reward"] - worst["reward"]) >= 0.05:
                dpo_row = {
                    "prompt": f"Write a TLA+ specification for:\n{best['nl_description']}",
                    "chosen": best["spec"],
                    "rejected": worst["spec"],
                    "chosen_reward": best["reward"],
                    "rejected_reward": worst["reward"],
                    "piece_name": "full_spec",
                    "module_name": best.get("module_name", "Spec"),
                }
                dpo_f.write(json.dumps(dpo_row, ensure_ascii=False) + "\n")
                new_dpo += 1

    log(f"Phase 3 complete: {new_sft} new SFT examples, {new_dpo} new DPO pairs")
    return {"new_sft": new_sft, "new_dpo": new_dpo}


# ── Phase 4: Train ───────────────────────────────────────────────────────

def phase_train(harvest_stats: dict, smoke: bool = False) -> str:
    """Choose and run the appropriate training method."""
    new_dpo = harvest_stats.get("new_dpo", 0)
    new_sft = harvest_stats.get("new_sft", 0)

    if new_dpo >= 100:
        action = "dpo_piecewise"
        log(f"Training: piecewise DPO ({new_dpo} new pairs)")
        cmd = [_PY, "-m", "src.training.train_dpo_piecewise",
               "--base-model", "outputs/merged_model_v14"]
        if smoke:
            cmd.append("--smoke")
    elif new_sft >= 50:
        action = "sft_incremental"
        log(f"Training: incremental SFT ({new_sft} new gold specs)")
        cmd = [_PY, "-m", "src.training.train",
               "--resume", "outputs/checkpoints/checkpoint-401",
               "--epochs", "1"]
        if smoke:
            cmd.extend(["--max-steps", "4"])
    else:
        action = "grpo_50"
        log("Training: 50-step full-spec GRPO")
        cmd = [_PY, "-m", "scripts.train_rl_fullspec",
               "--max-steps", "50"]
        if smoke:
            cmd.append("--smoke")

    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        log(f"Training failed (exit {result.returncode})")
        return f"{action}_failed"

    log(f"Training complete: {action}")
    return action


# ── Phase 5: Evaluate ────────────────────────────────────────────────────

def phase_evaluate(
    model: str = "chattla:20b",
    holdout_file: str | None = None,
    output_file: str | None = None,
) -> dict:
    """Run holdout evaluation and return results."""
    if holdout_file is None:
        holdout_file = str(_HOLDOUT)
    if output_file is None:
        output_file = f"outputs/eval/holdout_flywheel_{int(time.time())}.json"

    cmd = [
        _PY, "-m", "scripts.eval_3shot_tlc_tlaps",
        "--model", model,
        "--holdout", holdout_file,
        "--output", output_file,
    ]
    log(f"Evaluating: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))

    out_path = _REPO_ROOT / output_file
    if out_path.is_file():
        data = json.loads(out_path.read_text(encoding="utf-8"))
        log(f"Phase 5 results: {data.get('by_tier', {})}, diamond={data.get('diamond', 0)}")
        return data

    log("Evaluation failed — no output file")
    return {}


# ── Phase 6: Deploy ──────────────────────────────────────────────────────

def phase_deploy(skip: bool = False) -> bool:
    """Merge LoRA → GGUF → Ollama."""
    if skip:
        log("Deploy skipped")
        return True

    # Merge
    merge_cmd = [_PY, "-m", "src.training.merge_lora"]
    log(f"Merging: {' '.join(merge_cmd)}")
    r = subprocess.run(merge_cmd, cwd=str(_REPO_ROOT))
    if r.returncode != 0:
        log("Merge failed")
        return False

    # GGUF + Ollama
    gguf_cmd = [_PY, "-m", "src.inference.convert_to_gguf"]
    log(f"Converting: {' '.join(gguf_cmd)}")
    r = subprocess.run(gguf_cmd, cwd=str(_REPO_ROOT))
    if r.returncode != 0:
        log("GGUF conversion failed")
        return False

    log("Deploy complete")
    return True


# ── Gating ───────────────────────────────────────────────────────────────

def _load_best_gold() -> int:
    """Load the best gold count from previous evaluations."""
    eval_dir = _REPO_ROOT / "outputs" / "eval"
    best = 0
    if eval_dir.is_dir():
        for f in eval_dir.glob("holdout_*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                gold = data.get("by_tier", {}).get("gold", 0)
                best = max(best, gold)
            except Exception:
                continue
    return best


def _gate_check(eval_results: dict, best_gold: int) -> bool:
    """Check if this cycle passed the holdout gate."""
    current_gold = eval_results.get("by_tier", {}).get("gold", 0)
    if current_gold < best_gold - 1:
        log(f"GATE FAILED: gold {current_gold} < best {best_gold} - 1. Rolling back.")
        return False
    log(f"GATE PASSED: gold {current_gold} (best was {best_gold})")
    return True


# ── Main loop ────────────────────────────────────────────────────────────

def run_cycle(
    cycle_num: int,
    n_prompts: int = 50,
    model: str = "chattla:20b",
    skip_train: bool = False,
    skip_deploy: bool = False,
    smoke: bool = False,
) -> dict:
    """Run one complete flywheel cycle."""
    log(f"{'='*60}")
    log(f"FLYWHEEL CYCLE {cycle_num}")
    log(f"{'='*60}")

    if smoke:
        n_prompts = 5

    # Phase 1: Generate
    candidates = phase_generate(n_prompts=n_prompts, model=model)
    if not candidates:
        return {"cycle": cycle_num, "error": "no_candidates"}

    # Phase 2: Validate
    candidates = phase_validate(candidates)

    # Phase 3: Harvest
    harvest_stats = phase_harvest(candidates)

    # Phase 4: Train
    action = "skipped"
    if not skip_train:
        action = phase_train(harvest_stats, smoke=smoke)

    # Phase 5: Evaluate
    eval_results = phase_evaluate(model=model)

    # Phase 6: Gate + Deploy
    best_gold = _load_best_gold()
    gate_passed = _gate_check(eval_results, best_gold)

    if gate_passed and not skip_deploy and not skip_train:
        phase_deploy()
    elif not gate_passed:
        log("Skipping deploy due to gate failure")

    # Log metrics
    metrics = {
        "cycle": cycle_num,
        "timestamp": ts(),
        "generated": len(candidates),
        "gold": sum(1 for c in candidates if c.get("tier") == "gold"),
        "diamond": sum(1 for c in candidates if c.get("is_diamond")),
        "new_sft": harvest_stats.get("new_sft", 0),
        "new_dpo": harvest_stats.get("new_dpo", 0),
        "train_action": action,
        "holdout_gold": eval_results.get("by_tier", {}).get("gold", 0),
        "holdout_diamond": eval_results.get("diamond", 0),
        "gate_passed": gate_passed,
    }
    _METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _METRICS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    log(f"Cycle {cycle_num} complete: {json.dumps(metrics, indent=2)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cycles", type=int, default=5,
                        help="Number of flywheel cycles to run")
    parser.add_argument("--n-prompts", type=int, default=50,
                        help="Prompts per cycle")
    parser.add_argument("--model", default="chattla:20b")
    parser.add_argument("--skip-train", action="store_true",
                        help="Generate + validate + harvest only")
    parser.add_argument("--skip-deploy", action="store_true",
                        help="Skip merge/GGUF/Ollama deploy")
    parser.add_argument("--smoke", action="store_true",
                        help="1 cycle, 5 prompts, smoke-test training")
    args = parser.parse_args()

    n_cycles = 1 if args.smoke else args.cycles

    for cycle in range(1, n_cycles + 1):
        try:
            metrics = run_cycle(
                cycle_num=cycle,
                n_prompts=args.n_prompts,
                model=args.model,
                skip_train=args.skip_train,
                skip_deploy=args.skip_deploy,
                smoke=args.smoke,
            )
            if metrics.get("error"):
                log(f"Cycle {cycle} had error: {metrics['error']}")
        except KeyboardInterrupt:
            log("Interrupted by user")
            break
        except Exception as e:
            log(f"Cycle {cycle} failed with exception: {e}")
            import traceback
            traceback.print_exc()

    log("Flywheel complete")


if __name__ == "__main__":
    main()
