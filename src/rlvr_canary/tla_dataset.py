"""TLA+ training dataset for GRPO using the per-action TLC reward.

We don't try to teach the model to write a whole module from scratch under
RL — that's the FormaLLM "single-pass fails" failure mode. Instead, every
training prompt asks the model to produce **only the `Next` action** for a
spec whose Init / VARIABLES / TypeOK / Spec are already given (taken from a
verified Diamond-tier reference). The reward then runs `Init + candidate Next +
TypeOK` through TLC and grades the model on whether the action keeps TypeOK
invariant.

Why this works as an RL signal:

  * The action-level state space is bounded by the harness (CONSTANTS were
    fixed in the gold spec), so TLC terminates fast and reliably.
  * The reward is dense at the spec level (any candidate that parses gets
    graded) which is much better than 0/1 on whole-module emission.
  * Each training prompt gives the model the Init + variables + intended
    behavior in plain English, so it doesn't have to invent the scaffolding —
    it only has to write the action body. This matches the FormaLLM
    progressive-prompting recipe that produced the only TLC passes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.validators.per_action_tlc import (
    ActionExample,
    iter_action_examples,
)


_TLA_DEVELOPER_PROMPT = """\
You are ChatTLA, an expert at writing TLA+ specifications. The user will give
you an existing TLA+ module that defines the variables, the initial state, the
type invariant, and a natural-language description of the system's intended
behavior. Your job is to write **only the `Next` operator** so that the
resulting module type-checks and TLC accepts it as a valid step relation.

Output rules:
  - Emit ONLY a `Next ==` operator definition (and any helper sub-actions it
    directly references). Nothing else: no markdown fences, no prose, no
    `<think>` blocks, no `MODULE` header, no `====`.
  - Every disjunct in `Next` must specify ALL variables: either prime them
    (x' = ...) or use UNCHANGED <<x>>.
  - Use ASCII operators only (`/\\`, `\\/`, `\\in`, `->`, `|->`), never
    Unicode.
Reasoning: medium\
"""


@dataclass
class TLATrainExample:
    prompt_id: str
    prompt: list[dict[str, str]]
    harness: Any   # ActionHarness — kept opaque so the reward fn can splice
    nl: str


def _build_user_message(ex: ActionExample) -> str:
    h = ex.harness
    nl_clean = re.sub(r"\s+", " ", (ex.nl_description or "")).strip()[:600]
    return (
        f"System description:\n{nl_clean}\n\n"
        f"Existing module (everything up to but not including `Next`):\n"
        f"```tla\n{h.prefix}\n```\n\n"
        f"After your `Next ==` block, the module continues with:\n"
        f"```tla\n{h.suffix.strip()}\n```\n\n"
        f"Write ONLY the `Next ==` operator. Make sure every disjunct "
        f"specifies all of the module's variables (either primed or via "
        f"UNCHANGED)."
    )


def load_tla_action_prompts(
    corpus_path: str | Path = "data/processed/diamond_curated.jsonl",
) -> list[TLATrainExample]:
    """Build a list of GRPO-ready TLA+ training examples.

    Each example carries an `ActionHarness` we'll need at reward time —
    GRPOTrainer's `remove_unused_columns=False` lets us pass the harness
    through as a dataset column.
    """
    out: list[TLATrainExample] = []
    for ex in iter_action_examples(corpus_path):
        prompt = [
            {"role": "system", "content": _TLA_DEVELOPER_PROMPT},
            {"role": "user",   "content": _build_user_message(ex)},
        ]
        out.append(TLATrainExample(
            prompt_id=ex.prompt_id,
            prompt=prompt,
            harness=ex.harness,
            nl=ex.nl_description,
        ))
    return out
