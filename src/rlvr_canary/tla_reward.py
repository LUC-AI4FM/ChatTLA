"""Per-action TLC reward for GRPO training on TLA+.

Reward shaping (mirrors the GSM8K canary's three-tier structure so the same
TRL trainer can drive both):

  +1.0  TLC accepts the spliced module — gold (TypeOK + Spec hold)
  +0.5  SANY parses but TLC fails or times out — silver
  +0.2  the model emitted a normalizable `Next == ...` body but SANY rejected it
  +0.05 the model emitted *something* and the canonical normalizer cleaned it
   0.0  unparseable / refused / empty

Why the staircase: the dominant TLA+ failure mode is "model emits something
that almost looks right but trips SANY on a tiny mistake." Without partial
credit, every such attempt is indistinguishable from "model wrote `def foo():`."
The staircase lets policy gradient prefer "almost correct" over "complete
garbage" even before any TLC pass arrives.

Speed:
  - Hard cap: 30s per TLC call (set in validate_action). With num_generations=8
    and 16 prompts/step, worst-case wall is ~64 minutes per step. The harness
    state space is small (CONSTANTS bounded by the gold spec), so most actions
    finish in <2s. Slow specs are caught by the cap.
  - Concurrency: TLC sub-processes are CPU-bound and don't conflict with vLLM.
    A pool of N workers (default 4) checks N completions in parallel.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.postprocess import strip_reasoning_artifacts
from src.postprocess.normalize import (
    NormalizationReport,
    UNICODE_OP_TABLE,
)
from src.validators.per_action_tlc import ActionHarness, validate_action


_NEXT_RE = re.compile(r"(Next\s*==.*?)(?=^[A-Z]\w*\s*==|\Z)", re.MULTILINE | re.DOTALL)
_TLC_TIMEOUT_S = int(os.environ.get("CHATTLA_REWARD_TLC_TIMEOUT", "20"))
_REWARD_WORKERS = int(os.environ.get("CHATTLA_REWARD_WORKERS", "4"))


def _completion_text(comp: Any) -> str:
    """Normalize TRL's two completion shapes (string vs message list)."""
    if isinstance(comp, list):
        return "".join(m.get("content", "") for m in comp if isinstance(m, dict))
    return str(comp or "")


def _extract_next_body(text: str) -> str | None:
    """Pull a `Next == ...` block out of the model output, tolerating fences,
    `<think>` blocks, and prose. Returns None if the model emitted nothing
    that looks like a Next operator.

    We deliberately do NOT use the full module-level normalizer here. That
    function appends a `====` terminator and synthesizes module headers when
    it can't find one — both wrong for a *fragment* that's about to be
    spliced into a harness. Instead we strip reasoning artifacts (harmony
    tags, <think>, fences), apply Unicode→ASCII operator translation
    inline, and then carve the operator body."""
    if not text:
        return None
    rep = NormalizationReport()
    cleaned = strip_reasoning_artifacts(text, rep)
    # Inline Unicode op replacement (no whitespace collapse — junction
    # alignment is column-sensitive in TLA+).
    for src, dst in UNICODE_OP_TABLE:
        if src in cleaned:
            cleaned = cleaned.replace(src, dst)
    # If the model emitted a full module by accident, drop the terminator
    # and any trailing junk so it can splice cleanly.
    cleaned = re.sub(r"\n=+\s*\Z", "", cleaned).rstrip()
    m = _NEXT_RE.search(cleaned)
    if m:
        body = m.group(1).rstrip()
        # Drop any leftover module terminator that snuck in.
        body = re.sub(r"\n=+\s*\Z", "", body).rstrip()
        return body
    # Some models emit just the body without `Next ==`. If the cleaned text
    # parses as a sequence of disjuncts, accept it as the body.
    body = cleaned.strip()
    if body.startswith("\\/") or body.startswith("/\\"):
        return body
    return None


def _grade_one(harness: ActionHarness, completion_text: str) -> float:
    body = _extract_next_body(completion_text)
    if body is None:
        return 0.0
    # Even if the body never reaches TLC, the model gets a small credit for
    # producing something the normalizer could clean — that's the format
    # rung that prevents collapse-to-nothing during early RL.
    base = 0.05
    try:
        result = validate_action(harness, body, timeout=_TLC_TIMEOUT_S)
    except Exception:
        return base
    if result.tier == "gold":
        return 1.0
    if result.tier == "silver":
        return 0.5
    # bronze: SANY rejected the spliced module. Sub-tier by violation count
    # to give GRPO continuous signal within the dominant reward band.
    # Fewer SANY errors = closer to silver = higher reward.
    n_violations = len(result.violations) if result.violations else 5
    if n_violations == 0:
        return 0.35  # SANY passed but TLC found issues — almost silver
    elif n_violations <= 2:
        return 0.25  # minor parse errors
    elif n_violations <= 5:
        return 0.15  # moderate errors
    else:
        return 0.10  # many errors — barely above base


def per_action_tlc_reward(
    prompts: list[Any] | None = None,
    completions: list[Any] | None = None,
    harness_prefix: list[str] | None = None,
    harness_suffix: list[str] | None = None,
    harness_module: list[str] | None = None,
    **_: Any,
) -> list[float]:
    """TRL GRPO reward function. One float per completion.

    The harness fields are passed as separate string columns from the
    dataset (HF Datasets / Arrow can't always serialize dataclass objects,
    but strings always work). We reconstruct an ActionHarness on the fly
    here.
    """
    completions = completions or []
    if not completions:
        return []
    n = len(completions)

    def _col(col: list[str] | None) -> list[str]:
        if not col:
            return [""] * n
        if len(col) == n:
            return col
        # TRL passes 1 column entry per prompt; the trainer fans out
        # num_generations completions, so the column is broadcast.
        if len(col) < n and n % len(col) == 0:
            return [col[i % len(col)] for i in range(n)]
        return (col * ((n // len(col)) + 1))[:n]

    prefixes = _col(harness_prefix)
    suffixes = _col(harness_suffix)
    modules  = _col(harness_module)

    harnesses = [
        ActionHarness(module_name=m, prefix=p, suffix=s, gold_next="")
        for p, s, m in zip(prefixes, suffixes, modules)
    ]
    texts = [_completion_text(c) for c in completions]

    rewards: list[float] = [0.0] * n
    with ThreadPoolExecutor(max_workers=_REWARD_WORKERS) as pool:
        futures = {
            pool.submit(_grade_one, harnesses[i], texts[i]): i
            for i in range(n)
        }
        for fut in futures:
            i = futures[fut]
            try:
                rewards[i] = fut.result(timeout=_TLC_TIMEOUT_S + 10)
            except Exception:
                rewards[i] = 0.0
    return rewards
