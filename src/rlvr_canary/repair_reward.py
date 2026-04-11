"""Improvement-based reward for GRPO repair training on TLA+.

Unlike fullspec_component_reward which grades absolute spec quality,
this reward measures IMPROVEMENT: how much better is the repaired spec
compared to the broken input?

This solves the sparse reward problem: even partial fixes (reducing
SANY errors, adding missing Init) produce positive signal, giving
GRPO the variance it needs to learn.

Reward shaping:
  - Regression (got worse):     0.0–0.1  (harsh penalty)
  - No change:                  0.15     (small baseline)
  - Improvement:                0.2–0.8  (proportional to delta)
  - Tier transition bonuses:    +0.1 each (structure, SANY, TLC)
  - Full TLC pass:              bonus +0.1

The before_score for each repair prompt is registered at training
startup via register_before_scores(). The repair_id is embedded
in the prompt string as <!-- repair:ID --> and extracted at reward time.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.validators.component_validator import reward_from_spec


_REWARD_WORKERS = int(os.environ.get("CHATTLA_REWARD_WORKERS", "4"))
_FULL_TLC_TIMEOUT = int(os.environ.get("CHATTLA_REWARD_TLC_TIMEOUT", "30"))

# Module-level state: prompt repair_id -> before_score
_BEFORE_SCORES: dict[str, float] = {}

# Regex to extract repair_id from prompt string
_REPAIR_ID_RE = re.compile(r"<!-- repair:([^ ]+) -->")


def register_before_scores(mapping: dict[str, float]) -> None:
    """Called by train_rl_repair.py at startup to register before_scores."""
    _BEFORE_SCORES.update(mapping)
    print(f"[repair_reward] registered {len(mapping)} before_scores "
          f"(mean={sum(mapping.values()) / max(len(mapping), 1):.3f})")


def _extract_repair_id(prompt: Any) -> str | None:
    """Extract repair_id from the <!-- repair:ID --> tag in the prompt."""
    text = str(prompt) if not isinstance(prompt, str) else prompt
    m = _REPAIR_ID_RE.search(text)
    return m.group(1) if m else None


def _shape_reward(before: float, after: float) -> float:
    """Compute shaped reward from score delta.

    Returns a value in [0, 1] suitable for GRPO.
    """
    delta = after - before

    if delta < -0.01:
        # Regression: strong negative signal
        return max(0.0, 0.1 + delta)

    if delta < 0.01:
        # No change: small baseline reward
        return 0.15

    # Improvement: proportional to delta
    reward = 0.2 + delta * 0.6  # maps [0.01, 1.0] -> [0.206, 0.80]

    # Tier transition bonuses (component_validator weight thresholds)
    if before < 0.10 and after >= 0.10:
        reward += 0.1  # gained basic structure (Init + Next present)
    if before < 0.40 and after >= 0.40:
        reward += 0.1  # SANY clean + invariants declared
    if after >= 1.0:
        reward += 0.1  # full TLC pass

    return min(1.0, reward)


def _completion_text(comp: Any) -> str:
    """Normalize TRL's two completion shapes (string vs message list)."""
    if isinstance(comp, list):
        return "".join(m.get("content", "") for m in comp if isinstance(m, dict))
    return str(comp or "")


def _grade_one(text: str) -> float:
    """Grade a single completion via the component validator."""
    if not text or not text.strip():
        return 0.0
    try:
        return reward_from_spec(
            text,
            run_depth1=True,
            run_full_tlc=True,
            full_tlc_timeout=_FULL_TLC_TIMEOUT,
        )
    except Exception:
        return 0.0


def repair_reward(
    prompts: list[Any] | None = None,
    completions: list[Any] | None = None,
    **_: Any,
) -> list[float]:
    """TRL GRPO reward function for repair training.

    For each completion:
      1. Extract repair_id from prompt -> look up before_score
      2. Run reward_from_spec on completion -> after_score
      3. Return shaped reward based on improvement delta
    """
    completions = completions or []
    prompts = prompts or []
    if not completions:
        return []

    n = len(completions)
    texts = [_completion_text(c) for c in completions]

    # Grade all completions in parallel
    after_scores: list[float] = [0.0] * n
    with ThreadPoolExecutor(max_workers=_REWARD_WORKERS) as pool:
        futures = {
            pool.submit(_grade_one, texts[i]): i
            for i in range(n)
        }
        for fut in futures:
            i = futures[fut]
            try:
                after_scores[i] = fut.result(timeout=_FULL_TLC_TIMEOUT + 30)
            except Exception:
                after_scores[i] = 0.0

    # Look up before_scores and compute shaped rewards
    rewards: list[float] = []
    for i in range(n):
        # GRPOTrainer replicates prompts for num_generations, so prompt
        # index maps to i // num_generations... but TRL passes the
        # already-replicated list, so prompts[i] is correct.
        prompt_text = str(prompts[i]) if i < len(prompts) else ""
        repair_id = _extract_repair_id(prompt_text)

        if repair_id and repair_id in _BEFORE_SCORES:
            before = _BEFORE_SCORES[repair_id]
        else:
            # Fallback: treat as absolute reward (no shaping)
            before = 0.0

        rewards.append(_shape_reward(before, after_scores[i]))

    return rewards
