#!/usr/bin/env python3
"""Collect Ralph-style repair trajectories for GRPO training.

Runs iterative repair loops over NL problem descriptions:
  1. Generate initial spec via Ollama
  2. Verify with SANY/TLC/TLAPS/Apalache (Ralph's pipeline)
  3. Score with component_validator partial_credit
  4. Build repair prompt with line-annotated errors
  5. Generate repair attempt
  6. Repeat

Outputs:
  - ralph_trajectories.jsonl: full trajectory per problem
  - ralph_repair_pairs.jsonl: flattened (broken, errors, repaired, delta) pairs
    ready for GRPO repair training

Reuses Ralph's verify pipeline and repair_prompt format directly.
Reuses component_validator for scoring.

Run:
    python -m scripts.collect_ralph_trajectories --smoke          # 5 topics, 2 iters
    python -m scripts.collect_ralph_trajectories --max-iters 6    # full run (~8h)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Ralph lives at ../ralph-tla relative to ChatTLA
_RALPH_ROOT = Path(os.getenv(
    "RALPH_PATH",
    _REPO_ROOT.parent / "ralph-tla",
))
if str(_RALPH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RALPH_ROOT))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class RepairStep:
    iteration: int
    spec: str
    module_name: str
    verify_summary: str
    sany_ok: bool
    partial_credit: float
    errors_rendered: str
    is_diamond: bool
    elapsed: float = 0.0


@dataclass
class Trajectory:
    prompt_id: str
    nl_description: str
    steps: list[RepairStep] = field(default_factory=list)
    achieved_diamond: bool = False
    best_partial_credit: float = 0.0


# ---------------------------------------------------------------------------
# Core collection logic
# ---------------------------------------------------------------------------
def collect_one(
    nl: str,
    prompt_id: str,
    model: str,
    max_iters: int = 6,
    verbose: bool = True,
) -> Trajectory:
    """Run one Ralph repair loop, recording every step."""
    # Lazy imports so multiprocessing forks cleanly
    from ralph_tla import (
        _ollama_generate,
        extract_module,
        initial_prompt,
        repair_prompt,
        verify,
    )
    from src.validators.component_validator import reward_from_spec

    traj = Trajectory(prompt_id=prompt_id, nl_description=nl)

    current_src = None
    prev_report = None

    for i in range(1, max_iters + 1):
        t0 = time.monotonic()

        # Generate
        if current_src is None:
            prompt = initial_prompt(nl)
        else:
            prompt = repair_prompt(nl, current_src, prev_report)

        try:
            raw = _ollama_generate(model, prompt)
        except Exception as e:
            if verbose:
                print(f"  [{prompt_id}] ollama error at iter {i}: {e}",
                      file=sys.stderr)
            break

        module_name, spec = extract_module(raw)

        # Verify (Ralph pipeline: SANY→TLC→TLAPS→Apalache)
        report, _tla_path = verify(spec, module_name)

        # Score (component_validator partial_credit)
        try:
            score = reward_from_spec(
                spec, run_depth1=True, run_full_tlc=True, full_tlc_timeout=30,
            )
        except Exception:
            score = 0.0

        elapsed = time.monotonic() - t0

        # Render errors for the repair prompt
        src_lines = spec.splitlines()
        errors = report.all_errors
        rendered = "\n".join(
            e.render(src_lines) for e in errors[:25]
        ) or "(no parsed errors; see tier summary)"

        step = RepairStep(
            iteration=i,
            spec=spec,
            module_name=module_name,
            verify_summary=report.summary(),
            sany_ok=report.sany.ok,
            partial_credit=score,
            errors_rendered=rendered,
            is_diamond=report.is_diamond(),
            elapsed=elapsed,
        )
        traj.steps.append(step)
        traj.best_partial_credit = max(traj.best_partial_credit, score)

        if verbose:
            print(f"  [{prompt_id}] iter {i}/{max_iters}: "
                  f"score={score:.3f} {report.summary()} ({elapsed:.1f}s)")

        if report.is_diamond():
            traj.achieved_diamond = True
            if verbose:
                print(f"  [{prompt_id}] DIAMOND at iteration {i}!")
            break

        # Set up for next repair iteration
        current_src = spec
        prev_report = report

    return traj


def flatten_to_repair_pairs(traj: Trajectory) -> list[dict]:
    """Extract repair pairs from consecutive steps.

    Includes ALL consecutive pairs (not just improvements) so the reward
    function can learn from regressions too. The repair_reward function
    handles shaping based on delta.
    """
    pairs = []
    for i in range(len(traj.steps) - 1):
        before = traj.steps[i]
        after = traj.steps[i + 1]

        pairs.append({
            "repair_id": f"{traj.prompt_id}_step{i + 1}to{i + 2}",
            "nl": traj.nl_description,
            "broken_spec": before.spec,
            "errors_rendered": before.errors_rendered,
            "verify_summary": before.verify_summary,
            "before_score": before.partial_credit,
            "repaired_spec": after.spec,
            "after_score": after.partial_credit,
            "before_diamond": before.is_diamond,
            "after_diamond": after.is_diamond,
        })

    return pairs


# ---------------------------------------------------------------------------
# Worker for ProcessPoolExecutor
# ---------------------------------------------------------------------------
def _worker(args: tuple) -> tuple[dict, list[dict]]:
    """Process one topic. Returns (trajectory_dict, repair_pairs)."""
    nl, prompt_id, model, max_iters = args
    traj = collect_one(nl, prompt_id, model, max_iters)
    pairs = flatten_to_repair_pairs(traj)
    # Convert to serializable dict
    traj_dict = {
        "prompt_id": traj.prompt_id,
        "nl_description": traj.nl_description,
        "achieved_diamond": traj.achieved_diamond,
        "best_partial_credit": traj.best_partial_credit,
        "num_steps": len(traj.steps),
        "steps": [asdict(s) for s in traj.steps],
    }
    return traj_dict, pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="chattla:20b",
                        help="Ollama model tag (default: chattla:20b)")
    parser.add_argument("--max-iters", type=int, default=6,
                        help="Max repair iterations per problem (default: 6)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel topic workers (default: 4)")
    parser.add_argument("--out-trajectories",
                        default="data/processed/ralph_trajectories.jsonl")
    parser.add_argument("--out-pairs",
                        default="data/processed/ralph_repair_pairs.jsonl")
    parser.add_argument("--smoke", action="store_true",
                        help="5 topics, 2 iters for quick sanity check")
    args = parser.parse_args()

    if args.smoke:
        args.max_iters = 2
        args.workers = 1

    # Load NL descriptions
    from src.rlvr_canary.fullspec_dataset import load_fullspec_prompts

    examples = load_fullspec_prompts(
        include_topics=True,
        include_diamond_sft=True,
        include_train=False,
        max_per_source=5 if args.smoke else None,
    )
    if not examples:
        print("No prompts found.", file=sys.stderr)
        return 1
    print(f"[trajectories] {len(examples)} problems to process")

    # Build work items
    work = [
        (ex.nl_description, ex.prompt_id, args.model, args.max_iters)
        for ex in examples
    ]

    # Output paths
    traj_path = _REPO_ROOT / args.out_trajectories
    pairs_path = _REPO_ROOT / args.out_pairs
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)

    total_trajectories = 0
    total_pairs = 0
    total_diamonds = 0
    t0 = time.monotonic()

    # Process — use serial execution if workers=1 (easier debugging)
    with open(traj_path, "w") as tf, open(pairs_path, "w") as pf:
        if args.workers <= 1:
            for item in work:
                traj_dict, pairs = _worker(item)
                tf.write(json.dumps(traj_dict) + "\n")
                for p in pairs:
                    pf.write(json.dumps(p) + "\n")
                total_trajectories += 1
                total_pairs += len(pairs)
                if traj_dict["achieved_diamond"]:
                    total_diamonds += 1
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(_worker, item): item[1]
                    for item in work
                }
                for fut in as_completed(futures):
                    prompt_id = futures[fut]
                    try:
                        traj_dict, pairs = fut.result()
                    except Exception as e:
                        print(f"[trajectories] FAILED {prompt_id}: {e}",
                              file=sys.stderr)
                        continue
                    tf.write(json.dumps(traj_dict) + "\n")
                    for p in pairs:
                        pf.write(json.dumps(p) + "\n")
                    total_trajectories += 1
                    total_pairs += len(pairs)
                    if traj_dict["achieved_diamond"]:
                        total_diamonds += 1

    elapsed = time.monotonic() - t0
    print(f"\n[trajectories] done in {elapsed:.0f}s")
    print(f"  trajectories: {total_trajectories} -> {traj_path}")
    print(f"  repair pairs: {total_pairs} -> {pairs_path}")
    print(f"  diamonds:     {total_diamonds}/{total_trajectories}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
