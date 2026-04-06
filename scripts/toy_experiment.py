#!/usr/bin/env python3
"""
toy_experiment.py — Fast convergence test for ChatTLA training pipeline.

Purpose: Answer the question "does SFT on verified TLA+ specs teach the
model to produce valid TLA+?" in minutes, not hours.

Design:
  1. Build a tiny dataset of 5 known-gold TLA+ specs (shortest ones)
  2. Train for N steps with aggressive LR, logging loss every step
  3. Generate specs for:
     (a) MEMORIZATION: the exact training prompts (should be easy)
     (b) GENERALIZATION: similar but unseen prompts (the real test)
  4. Validate every generated spec through SANY + TLC
  5. Print a clear pass/fail verdict

Usage:
    # Quick smoke (3 training steps, ~5 min):
    CUDA_VISIBLE_DEVICES=0 python scripts/toy_experiment.py --steps 3

    # Short convergence test (20 steps, ~15 min):
    CUDA_VISIBLE_DEVICES=0 python scripts/toy_experiment.py --steps 20

    # Full overfit test (50 steps, ~30 min):
    CUDA_VISIBLE_DEVICES=0 python scripts/toy_experiment.py --steps 50

    # Skip training, just evaluate current model:
    python scripts/toy_experiment.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Tiny gold dataset: hand-picked short, verified TLA+ specs
# ---------------------------------------------------------------------------

GOLD_SPECS = {
    "Counter": {
        "description": "A simple counter that increments from 0 up to a maximum value N, then stops.",
        "hints": "Use a single variable count, CONSTANT N, Init sets count=0, Next increments if count < N.",
        "spec": r"""---- MODULE Counter ----
EXTENDS Naturals

CONSTANT N

VARIABLE count

Init == count = 0

Next ==
    IF count < N
    THEN count' = count + 1
    ELSE UNCHANGED count

Spec == Init /\ [][Next]_count

TypeOK == count \in 0..N

====
""",
        "invariants": ["TypeOK"],
    },
    "OneBitClock": {
        "description": "A one-bit clock that alternates between 0 and 1 forever.",
        "hints": "Single variable bit, Init sets bit=0, Next flips the bit.",
        "spec": r"""---- MODULE OneBitClock ----
EXTENDS Naturals

VARIABLE bit

Init == bit = 0

Next == bit' = 1 - bit

Spec == Init /\ [][Next]_bit

TypeOK == bit \in {0, 1}

====
""",
        "invariants": ["TypeOK"],
    },
    "TokenPass": {
        "description": "A token passing protocol between 3 nodes arranged in a ring. The token moves clockwise from node 1 to 2 to 3 and back to 1.",
        "hints": "Variable token in 1..3, Next moves token clockwise modulo 3.",
        "spec": r"""---- MODULE TokenPass ----
EXTENDS Naturals

VARIABLE token

Init == token = 1

Next ==
    token' = IF token = 3 THEN 1 ELSE token + 1

Spec == Init /\ [][Next]_token

TypeOK == token \in 1..3

====
""",
        "invariants": ["TypeOK"],
    },
    "BoundedBuffer": {
        "description": "A bounded buffer of capacity 2. A producer can add an item if the buffer is not full, a consumer can remove an item if the buffer is not empty.",
        "hints": "Variable buf_size in 0..2, producer increments, consumer decrements.",
        "spec": r"""---- MODULE BoundedBuffer ----
EXTENDS Naturals

VARIABLE buf_size

Init == buf_size = 0

Produce == buf_size < 2 /\ buf_size' = buf_size + 1

Consume == buf_size > 0 /\ buf_size' = buf_size - 1

Next == Produce \/ Consume

Spec == Init /\ [][Next]_buf_size

TypeOK == buf_size \in 0..2

====
""",
        "invariants": ["TypeOK"],
    },
    "Mutex": {
        "description": "Two processes that take turns entering a critical section. At most one process is in the critical section at any time.",
        "hints": "Variable turn in {1,2} and pc as a function from {1,2} to {idle, critical}.",
        "spec": r"""---- MODULE Mutex ----
EXTENDS Naturals, FiniteSets

VARIABLES turn, pc

Procs == {1, 2}

Init ==
    /\ turn = 1
    /\ pc = [p \in Procs |-> "idle"]

Enter(p) ==
    /\ pc[p] = "idle"
    /\ turn = p
    /\ pc' = [pc EXCEPT ![p] = "critical"]
    /\ UNCHANGED turn

Exit(p) ==
    /\ pc[p] = "critical"
    /\ pc' = [pc EXCEPT ![p] = "idle"]
    /\ turn' = IF p = 1 THEN 2 ELSE 1

Next == \E p \in Procs : Enter(p) \/ Exit(p)

vars == <<turn, pc>>

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ turn \in Procs
    /\ pc \in [Procs -> {"idle", "critical"}]

MutualExclusion == Cardinality({p \in Procs : pc[p] = "critical"}) <= 1

====
""",
        "invariants": ["TypeOK", "MutualExclusion"],
    },
}

# Unseen prompts for generalization testing (structurally similar to training)
GENERALIZATION_PROMPTS = [
    {
        "name": "UpDownCounter",
        "description": "A counter that can both increment and decrement. It starts at 0 and stays within the range 0 to 5.",
        "hints": "Variable count in 0..5, two actions: increment if < 5, decrement if > 0.",
        "expected_invariants": ["TypeOK"],
    },
    {
        "name": "TrafficLight",
        "description": "A traffic light that cycles through three states: red, green, yellow, then back to red.",
        "hints": "Variable light in {red, green, yellow}. Red -> Green -> Yellow -> Red.",
        "expected_invariants": ["TypeOK"],
    },
    {
        "name": "SimpleLock",
        "description": "A simple lock with two processes. A process can acquire the lock if it is free, and release it when done.",
        "hints": "Variables lock_holder (0 means free, 1 or 2 means held), two actions: Acquire and Release.",
        "expected_invariants": ["TypeOK"],
    },
]

# ---------------------------------------------------------------------------
# Harmony prompt format helpers
# ---------------------------------------------------------------------------

from src.training.dataset_builder import _DEVELOPER_PROMPT as DEVELOPER_PROMPT  # single source of truth


def make_training_example(name: str, info: dict) -> dict:
    """Build a harmony-format training example."""
    desc = info["description"]
    if info.get("hints"):
        desc += f"\n\nHints: {info['hints']}"
    return {
        "messages": [
            {"role": "developer", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{desc}"},
            {"role": "assistant", "content": "I'll write a well-formed TLA+ specification with proper Init, Next, and invariants."},
            {"role": "assistant", "content": info["spec"].strip()},
        ],
        "_tier": "gold",
        "_toy": True,
    }


def build_toy_dataset(out_dir: Path):
    """Write tiny train/eval JSONL files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"

    examples = [make_training_example(n, s) for n, s in GOLD_SPECS.items()]

    with train_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Eval = first 2 examples
    with eval_path.open("w") as f:
        for ex in examples[:2]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[toy] Wrote {len(examples)} train, 2 eval examples to {out_dir}")
    return train_path, eval_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(train_path: Path, eval_path: Path, steps: int, lr: float):
    """Run SFT training on the toy dataset.

    Swaps the real train/eval JSONL with our toy data, runs training with
    --max-steps and --lr, then restores the originals.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 1: TRAINING ({steps} steps, lr={lr})")
    print(f"{'='*60}\n")

    data_dir = REPO_ROOT / "data" / "processed"
    real_train = data_dir / "train.jsonl"
    real_eval = data_dir / "eval.jsonl"
    backup_train = data_dir / "train.jsonl.bak"
    backup_eval = data_dir / "eval.jsonl.bak"

    # Backup real data
    import shutil
    if real_train.exists():
        shutil.copy2(real_train, backup_train)
    if real_eval.exists():
        shutil.copy2(real_eval, backup_eval)

    try:
        # Swap in toy data
        shutil.copy2(train_path, real_train)
        shutil.copy2(eval_path, real_eval)

        cmd = [
            sys.executable, "-m", "src.training.train",
            "--epochs", "100",           # high ceiling; --max-steps is the real limit
            "--max-steps", str(steps),
            "--lr", str(lr),
            "--per-device-batch-size", "1",
            "--gradient-accumulation-steps", "1",
            "--max-length", "2048",      # toy specs are short
        ]

        print(f"[toy] Running: {' '.join(cmd)}")
        t0 = time.time()

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(REPO_ROOT),
            timeout=3600,  # 1h hard cap
        )

        elapsed = time.time() - t0
        print(f"[toy] Training took {elapsed:.0f}s ({elapsed/60:.1f}m)")

        if result.returncode != 0:
            print(f"[toy] TRAINING FAILED (exit {result.returncode})")
            # Show last 2000 chars of stderr
            err = result.stderr or ""
            print(err[-2000:])
            return False

        # Extract loss values from stdout
        losses = []
        for line in (result.stdout or "").splitlines():
            if "'loss'" in line:
                try:
                    d = eval(line)
                    losses.append(float(d["loss"]))
                except Exception:
                    pass

        if losses:
            print(f"\n[toy] Loss curve ({len(losses)} logged points):")
            print(f"  Start: {losses[0]:.4f}")
            print(f"  End:   {losses[-1]:.4f}")
            print(f"  Delta: {losses[-1] - losses[0]:+.4f}")
            if losses[-1] < losses[0]:
                print("  >>> LOSS DECREASING (good)")
            else:
                print("  >>> WARNING: loss not decreasing!")
        else:
            print("[toy] No loss values captured (logging_steps may be > max_steps)")

        return True

    finally:
        # Restore originals
        if backup_train.exists():
            if real_train.exists():
                real_train.unlink()
            backup_train.rename(real_train)
        if backup_eval.exists():
            if real_eval.exists():
                real_eval.unlink()
            backup_eval.rename(real_eval)


# ---------------------------------------------------------------------------
# Evaluation via SANY/TLC (direct, no Ollama needed for validation)
# ---------------------------------------------------------------------------

def validate_spec(spec_text: str, module_name: str = "Temp") -> dict:
    """Run SANY + TLC on a spec string. Returns tier + details."""
    from src.validators.sany_validator import validate_string as sany_check
    from src.validators.tlc_validator import validate_string as tlc_check

    result = {"spec": spec_text, "sany": False, "tlc": False, "tier": "bronze", "errors": []}

    try:
        sany = sany_check(spec_text, module_name=module_name)
        result["sany"] = sany.valid
        if not sany.valid:
            result["errors"] = sany.errors[:3]
            return result
    except Exception as e:
        result["errors"] = [str(e)]
        return result

    try:
        tlc = tlc_check(spec_text, module_name=module_name, timeout=30)
        result["tier"] = tlc.tier
        result["tlc"] = tlc.tier == "gold"
        if tlc.tlc_violations:
            result["errors"] = tlc.tlc_violations[:3]
    except Exception as e:
        result["tier"] = "silver"
        result["errors"] = [str(e)]

    return result


def generate_with_ollama(prompt: str, model: str = "chattla:20b") -> str:
    """Generate a spec via Ollama."""
    import subprocess
    # Use the ollama client directly
    from src.inference.ollama_client import ChatTLAClient
    client = ChatTLAClient(model=model)
    return client.generate_spec(prompt, temperature=0.05)


def run_evaluation(model: str, skip_memorization: bool = False):
    """Test memorization and generalization."""
    print(f"\n{'='*60}")
    print(f"  PHASE 2: EVALUATION (model={model})")
    print(f"{'='*60}\n")

    results = {"memorization": [], "generalization": []}

    # --- Memorization test ---
    if not skip_memorization:
        print("--- MEMORIZATION TEST (can it reproduce training specs?) ---\n")
        for name, info in GOLD_SPECS.items():
            desc = info["description"]
            if info.get("hints"):
                desc += f"\n\nHints: {info['hints']}"

            print(f"  [{name}] Generating...", end=" ", flush=True)
            t0 = time.time()
            try:
                spec = generate_with_ollama(desc, model=model)
                elapsed = time.time() - t0

                import re
                m = re.search(r"----\s*MODULE\s+(\w+)", spec)
                mod_name = m.group(1) if m else name

                v = validate_spec(spec, module_name=mod_name)
                tier = v["tier"]
                emoji_map = {"gold": "GOLD", "silver": "SILVER", "bronze": "BRONZE"}
                status = emoji_map.get(tier, tier)
                print(f"{status} (sany={'OK' if v['sany'] else 'FAIL'} tlc={'OK' if v['tlc'] else 'FAIL'}) {elapsed:.1f}s")
                if v["errors"]:
                    print(f"    errors: {v['errors'][0][:100]}")
                results["memorization"].append({"name": name, **v, "time": elapsed})
            except Exception as e:
                print(f"ERROR: {e}")
                results["memorization"].append({"name": name, "sany": False, "tlc": False, "tier": "error", "time": 0})

    # --- Generalization test ---
    print("\n--- GENERALIZATION TEST (unseen but similar problems) ---\n")
    for prompt_info in GENERALIZATION_PROMPTS:
        name = prompt_info["name"]
        desc = prompt_info["description"]
        if prompt_info.get("hints"):
            desc += f"\n\nHints: {prompt_info['hints']}"

        print(f"  [{name}] Generating...", end=" ", flush=True)
        t0 = time.time()
        try:
            spec = generate_with_ollama(desc, model=model)
            elapsed = time.time() - t0

            import re
            m = re.search(r"----\s*MODULE\s+(\w+)", spec)
            mod_name = m.group(1) if m else name

            v = validate_spec(spec, module_name=mod_name)
            tier = v["tier"]
            emoji_map = {"gold": "GOLD", "silver": "SILVER", "bronze": "BRONZE"}
            status = emoji_map.get(tier, tier)
            print(f"{status} (sany={'OK' if v['sany'] else 'FAIL'} tlc={'OK' if v['tlc'] else 'FAIL'}) {elapsed:.1f}s")
            if v["errors"]:
                print(f"    errors: {v['errors'][0][:100]}")
            results["generalization"].append({"name": name, **v, "time": elapsed})
        except Exception as e:
            print(f"ERROR: {e}")
            results["generalization"].append({"name": name, "sany": False, "tlc": False, "tier": "error", "time": 0})

    return results


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def print_verdict(results: dict, trained: bool):
    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}\n")

    for phase in ["memorization", "generalization"]:
        items = results.get(phase, [])
        if not items:
            continue
        total = len(items)
        sany_pass = sum(1 for r in items if r.get("sany"))
        tlc_pass = sum(1 for r in items if r.get("tlc"))
        gold = sum(1 for r in items if r.get("tier") == "gold")
        silver = sum(1 for r in items if r.get("tier") == "silver")
        bronze = sum(1 for r in items if r.get("tier") == "bronze")

        print(f"  {phase.upper()}:")
        print(f"    SANY: {sany_pass}/{total} ({100*sany_pass/total:.0f}%)")
        print(f"    TLC:  {tlc_pass}/{total} ({100*tlc_pass/total:.0f}%)")
        print(f"    Tiers: {gold} gold, {silver} silver, {bronze} bronze")
        print()

    # Overall assessment
    mem = results.get("memorization", [])
    gen = results.get("generalization", [])

    mem_sany = sum(1 for r in mem if r.get("sany")) if mem else 0
    mem_tlc = sum(1 for r in mem if r.get("tlc")) if mem else 0
    gen_sany = sum(1 for r in gen if r.get("sany")) if gen else 0
    gen_tlc = sum(1 for r in gen if r.get("tlc")) if gen else 0

    print("  ASSESSMENT:")
    if mem:
        if mem_tlc == len(mem):
            print("    Memorization: PERFECT - model can reproduce all training specs")
        elif mem_sany == len(mem):
            print("    Memorization: PARTIAL - specs parse but TLC fails (logic errors)")
        elif mem_sany > 0:
            print("    Memorization: WEAK - some specs parse, training signal exists")
        else:
            print("    Memorization: FAILED - model cannot reproduce training specs")
            if trained:
                print("    >>> TRAINING PIPELINE MAY BE BROKEN <<<")

    if gen_tlc > 0:
        print(f"    Generalization: STRONG - {gen_tlc}/{len(gen)} unseen specs pass TLC")
        print("    >>> SFT approach is working! Scale up. <<<")
    elif gen_sany > 0:
        print(f"    Generalization: PARTIAL - {gen_sany}/{len(gen)} parse but TLC fails")
        print("    >>> Model learns syntax but not semantics. More gold data needed. <<<")
    else:
        print("    Generalization: NONE - no unseen specs parse")
        if trained:
            print("    >>> Consider: more data, longer training, or different approach <<<")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ChatTLA toy convergence experiment")
    parser.add_argument("--steps", type=int, default=30,
                        help="Training steps (default: 30, ~25min)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4, aggressive for overfit test)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, just evaluate the current chattla:20b model")
    parser.add_argument("--model", default="chattla:20b",
                        help="Model to evaluate (default: chattla:20b)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also evaluate base model for comparison")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  ChatTLA Toy Convergence Experiment")
    print(f"  Steps: {args.steps} | LR: {args.lr} | Eval-only: {args.eval_only}")
    print(f"{'='*60}")

    # Build toy dataset
    toy_dir = REPO_ROOT / "data" / "toy"
    train_path, eval_path = build_toy_dataset(toy_dir)

    trained = False
    if not args.eval_only:
        trained = run_training(train_path, eval_path, args.steps, args.lr)
        if trained:
            # Merge + GGUF + deploy (the fast path)
            print(f"\n{'='*60}")
            print(f"  DEPLOYING TOY MODEL")
            print(f"{'='*60}\n")

            print("[toy] Merging LoRA...")
            r = subprocess.run(
                [sys.executable, "-m", "src.training.merge_lora"],
                capture_output=True, text=True, cwd=str(REPO_ROOT),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")},
            )
            if r.returncode != 0:
                print(f"[toy] Merge failed: {r.stderr[-500:]}")
                return

            print("[toy] Converting to GGUF + registering with Ollama...")
            r = subprocess.run(
                [sys.executable, "-m", "src.inference.convert_to_gguf", "--quant", "q8_0"],
                capture_output=True, text=True, cwd=str(REPO_ROOT),
            )
            if r.returncode != 0:
                print(f"[toy] GGUF conversion failed: {r.stderr[-500:]}")
                return

            print("[toy] Model deployed as chattla:20b")

    # Baseline comparison
    if args.baseline:
        print("\n--- BASE MODEL (gpt-oss:20b) ---")
        base_results = run_evaluation("gpt-oss:20b", skip_memorization=True)
        print_verdict(base_results, trained=False)
        print("\n--- FINE-TUNED MODEL ---")

    # Evaluate
    results = run_evaluation(args.model)
    print_verdict(results, trained=trained)


if __name__ == "__main__":
    main()
