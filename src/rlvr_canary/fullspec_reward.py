"""Full-spec component-weighted reward for GRPO training on TLA+.

Unlike the per-action reward (tla_reward.py) which grades only the Next
operator fragment, this reward function evaluates complete TLA+ specs against
the 7-component partial credit signal from component_validator:

  init_present        0.05
  next_present        0.05
  init_level_ok       0.10
  next_level_ok       0.10
  invariants_declared 0.10
  tlc_depth1_ok       0.25
  tlc_full_ok         0.35

This gives ~10 distinct reward levels in [0, 1], solving the zero-variance
problem that killed the per-action 20B GRPO run (where all 8 completions
got the same tier → zero GRPO advantage → zero gradient).

Speed: ~55s worst case per completion (SANY 10s + depth-1 15s + full TLC 30s).
With ThreadPoolExecutor(4), 4 completions take ~55s wall time per GRPO step.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.validators.component_validator import reward_from_spec


_REWARD_WORKERS = int(os.environ.get("CHATTLA_REWARD_WORKERS", "4"))
_FULL_TLC_TIMEOUT = int(os.environ.get("CHATTLA_REWARD_TLC_TIMEOUT", "30"))


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


def fullspec_component_reward(
    prompts: list[Any] | None = None,
    completions: list[Any] | None = None,
    **_: Any,
) -> list[float]:
    """TRL GRPO reward function. One float per completion.

    Unlike per_action_tlc_reward, this function does NOT require harness
    columns — it evaluates complete specs end-to-end. The GRPOTrainer
    dataset only needs a `prompt` column.
    """
    completions = completions or []
    if not completions:
        return []

    texts = [_completion_text(c) for c in completions]
    n = len(texts)
    rewards: list[float] = [0.0] * n

    with ThreadPoolExecutor(max_workers=_REWARD_WORKERS) as pool:
        futures = {
            pool.submit(_grade_one, texts[i]): i
            for i in range(n)
        }
        for fut in futures:
            i = futures[fut]
            try:
                rewards[i] = fut.result(timeout=_FULL_TLC_TIMEOUT + 30)
            except Exception:
                rewards[i] = 0.0

    return rewards
