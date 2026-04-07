"""Per-action TLC reward — fast, localized semantic feedback.

Why this exists
---------------
The eval-callback bottleneck right now is full-spec TLC: we generate an entire
module, run SANY, run TLC end-to-end, and only learn one bit per spec ("did the
whole thing pass?"). The FormaLLM analysis (docs/formallm.md §4.4) shows that
~half of failures are mid- or late-file structural errors in the Next action,
not in the surrounding scaffolding. That means the reward signal we want is:

  *Given a known-good Init / TypeOK / VARIABLES, does the model's Next action
   keep TypeOK invariant?*

Splicing a generated Next into a verified harness:
  - Removes the noise from header / EXTENDS / variable-naming hallucinations
    (which the canonical normalizer already cleans up).
  - Localizes the credit assignment problem to the part being trained.
  - Runs ~10x faster than full-spec TLC because the state space is bounded
    by the harness, not by the model's free-form variable choices.
  - Lets us harvest *training* signal at every eval step, not just post-hoc.

Public API
----------
  build_harness(gold_record)  -> ActionHarness
  validate_action(harness, candidate_next, timeout) -> ActionResult
  iter_action_examples(corpus_path) -> Iterator[ActionExample]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


_OPERATOR_RE = re.compile(r"^([A-Z]\w*)\s*==", re.MULTILINE)
_NEXT_RE = re.compile(r"^Next\s*==.*?(?=^[A-Z]\w*\s*==|\Z)", re.MULTILINE | re.DOTALL)
_HEADER_RE = re.compile(r"^-{2,}\s*MODULE\s+\w+\s*-{2,}\s*$", re.MULTILINE)
_TERMINATOR_RE = re.compile(r"^={3,}\s*$", re.MULTILINE)


@dataclass
class ActionHarness:
    """A verified spec with the `Next` action removed.

    `prefix` ends just before `Next ==`. `suffix` starts just after the next
    operator definition. Splicing a candidate `Next` body in between yields
    a fresh module ready for TLC."""
    module_name: str
    prefix: str
    suffix: str
    gold_next: str
    cfg_text: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def assemble(self, candidate_next: str) -> str:
        """Return a complete module string with `candidate_next` spliced in.

        `candidate_next` may either start with `Next ==` or just be the body
        — both are normalized."""
        body = candidate_next.strip()
        if not body.startswith("Next"):
            body = "Next ==\n" + body
        return f"{self.prefix.rstrip()}\n\n{body}\n\n{self.suffix.lstrip()}"


@dataclass
class ActionResult:
    tier: str                  # "gold" | "silver" | "bronze"
    sany_ok: bool
    tlc_ok: bool
    violations: list[str] = field(default_factory=list)
    runtime_seconds: float = 0.0

    @property
    def reward(self) -> float:
        """Dense reward for RL: 1.0 gold, 0.5 silver (parses but TLC inconclusive),
        0.0 bronze. Designed so policy gradient prefers TLC-clean over
        parse-only over broken."""
        return {"gold": 1.0, "silver": 0.5, "bronze": 0.0}.get(self.tier, 0.0)


@dataclass
class ActionExample:
    prompt_id: str
    harness: ActionHarness
    nl_description: str


def build_harness(gold_spec: str, cfg_text: Optional[str] = None) -> Optional[ActionHarness]:
    """Carve a verified spec into (prefix, removed Next, suffix).

    Returns None if the spec doesn't have a clearly delimited `Next == ...`
    block."""
    spec = gold_spec.strip()
    header_m = _HEADER_RE.search(spec)
    if not header_m:
        return None
    mod_match = re.search(r"MODULE\s+(\w+)", spec)
    module_name = mod_match.group(1) if mod_match else "Harness"

    next_match = _NEXT_RE.search(spec)
    if not next_match:
        return None

    prefix = spec[: next_match.start()].rstrip()
    gold_next = next_match.group(0).rstrip()
    suffix = spec[next_match.end():].lstrip()

    # Make sure the suffix still has a terminator; if not, add one.
    if not _TERMINATOR_RE.search(suffix):
        suffix = suffix.rstrip() + "\n" + ("=" * 78) + "\n"

    return ActionHarness(
        module_name=module_name,
        prefix=prefix,
        suffix=suffix,
        gold_next=gold_next,
        cfg_text=cfg_text,
    )


def validate_action(harness: ActionHarness, candidate_next: str,
                    timeout: int = 30) -> ActionResult:
    """Splice `candidate_next` into `harness` and run TLC.

    The result reuses the same SANY/TLC pipeline as full-spec validation
    (src.validators.tlc_validator) so reward semantics are consistent across
    eval modes."""
    from src.validators.tlc_validator import validate_string

    spliced = harness.assemble(candidate_next)
    res = validate_string(
        spliced,
        cfg_content=harness.cfg_text,
        module_name=harness.module_name,
        timeout=timeout,
    )
    return ActionResult(
        tier=res.tier,
        sany_ok=res.tier in ("gold", "silver"),
        tlc_ok=res.tier == "gold",
        violations=list(res.tlc_violations),
        runtime_seconds=res.runtime_seconds,
    )


def iter_action_examples(corpus_path: str | Path) -> Iterator[ActionExample]:
    """Walk a curated jsonl and yield (harness, NL description) tuples for
    every spec we can carve."""
    import json
    from src.postprocess import normalize_spec

    p = Path(corpus_path)
    if not p.exists():
        return
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        nl = ""
        spec = ""
        for m in rec.get("messages", []):
            if m.get("role") == "user":
                nl = m.get("content", "")
            if m.get("role") == "assistant" and m.get("channel") == "final":
                spec = m.get("content", "")
        if not spec:
            continue
        cleaned, _ = normalize_spec(spec)
        harness = build_harness(cleaned)
        if harness is None:
            continue
        yield ActionExample(
            prompt_id=rec.get("_prompt_id", "?"),
            harness=harness,
            nl_description=nl,
        )
