"""Binary verifier reward for the RLVR canary.

The reward function is intentionally trivial:

  +1.0  if the model's <answer> matches the gold answer (numeric equality)
  +0.1  if the model's output parses as a number but is wrong  (format reward)
   0.0  if the output can't be parsed at all

The 0.1 format-shaping bonus stops the model from collapsing to no-answer
output during early RL when the verifier reward is sparse. This is the same
trick the published TRL/GRPO GSM8K recipes use; if you remove it, the
canary needs many more steps to lift off, which makes a stack regression
look like a hyperparameter problem.

The reward signature matches TRL's GRPOTrainer expectation:
    reward_fn(prompts, completions, **kwargs) -> list[float]
where `kwargs` contains any extra columns from the dataset (we pass `answer`).
"""

from __future__ import annotations

import re
from typing import Any

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(-?\d[\d,]*(?:\.\d+)?)\s*</answer>", re.IGNORECASE)
_FALLBACK_NUM_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)")


def extract_model_answer(text: str) -> str | None:
    """Pull the boxed answer out of a model completion.

    First try the canonical <answer>X</answer> form. If the model forgot the
    tag, fall back to the **last** numeric token in the output (matches the
    convention in published GSM8K eval scripts)."""
    if not text:
        return None
    m = _ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = _FALLBACK_NUM_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def _numeric_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def binary_correctness_reward(
    prompts: list[Any] | None = None,
    completions: list[Any] | None = None,
    answer: list[str] | None = None,
    **_: Any,
) -> list[float]:
    """TRL-compatible reward function. Returns one float per completion.

    Both `completions` and `prompts` may arrive as either plain strings or
    list-of-message dicts depending on TRL version; we handle both.
    """
    completions = completions or []
    answer = answer or []
    rewards: list[float] = []
    for i, comp in enumerate(completions):
        if isinstance(comp, list):
            text = "".join(m.get("content", "") for m in comp if isinstance(m, dict))
        else:
            text = str(comp)
        gold = answer[i] if i < len(answer) else None
        pred = extract_model_answer(text)
        if pred is None:
            rewards.append(0.0)
        elif gold is not None and _numeric_eq(pred, gold):
            rewards.append(1.0)
        else:
            rewards.append(0.1)  # parsed but wrong → small format bonus
    return rewards
