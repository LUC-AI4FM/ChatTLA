"""GSM8K loader for the RLVR canary.

We use the standard `gsm8k/main` HF dataset. Gold answers always live after
the literal "#### " marker in the `answer` field — that's how the dataset
authors flag the final integer for parsing.

The prompt format is deliberately minimal: a system message asking for
chain-of-thought followed by a final boxed answer. This matches what most
public GSM8K + GRPO recipes use, so our results are comparable.
"""

from __future__ import annotations

import re
from typing import Any

# Public so reward.py and tests use the same regex.
_GOLD_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")


CANARY_SYSTEM_PROMPT = """\
You solve grade-school math word problems. Think step by step inside <think>
tags, then put the final numerical answer inside <answer> tags. The answer
must be a single number with no units and no commas.

Example:
<think>Anna had 5 apples and bought 3 more, so she has 5 + 3 = 8.</think>
<answer>8</answer>\
"""


def extract_gold_answer(answer_field: str) -> str | None:
    """Pull the gold integer/decimal answer out of a GSM8K `answer` field."""
    m = _GOLD_ANSWER_RE.search(answer_field or "")
    if not m:
        return None
    return m.group(1).replace(",", "")


def _format_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CANARY_SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]


def load_gsm8k_prompts(split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
    """Return a list of records ready for TRL's GRPOTrainer.

    Each record has:
        prompt:  list[dict] in chat format (passed via tokenizer.apply_chat_template)
        answer:  str       gold answer ("72", "10", ...)
        question: str      raw NL question (kept for logging)
    """
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)
    out: list[dict[str, Any]] = []
    for ex in ds:
        gold = extract_gold_answer(ex["answer"])
        if gold is None:
            continue
        out.append({
            "prompt": _format_prompt(ex["question"]),
            "answer": gold,
            "question": ex["question"],
        })
        if limit is not None and len(out) >= limit:
            break
    return out
