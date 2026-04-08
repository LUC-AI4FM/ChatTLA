"""
spec_plan.py — Structured plan for plan-then-spec generation.

A SpecPlan is a high-level structural decomposition of a TLA+ module that the
model emits BEFORE writing the spec body. It is the ChatTLA analogue of the
"detailed proof plan" prompt that DeepSeek-Prover-V2 puts in front of every
Lean 4 generation: forcing the model to commit to a structure first reduces
the per-token entropy of the body and gives us a checkable artifact (the plan
itself) independent of whether the body parses.

Plans flow through the pipeline in two directions:

1. **Forward (inference)**: model emits a plan as JSON in the harmony `final`
   channel; we parse it, then condition a second generation pass on the plan
   to produce the spec body.

2. **Reverse (training data)**: for gold specs we already have, we mechanically
   extract a plan from the SANY AST (see component_validator.plan_from_ast)
   and inject it into the analysis channel of the training record. This teaches
   the model the plan→spec mapping without any additional curation pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional


InvariantKind = Literal["safety", "type", "liveness"]


@dataclass
class NextAction:
    """One disjunct of the Next-state relation."""
    name: str                # action operator name (e.g. "Acquire", "Release")
    guard: str = ""          # natural-language description of the precondition
    effect: str = ""         # natural-language description of the post-state


@dataclass
class PlannedInvariant:
    name: str
    statement: str = ""      # natural-language statement (NOT TLA+ syntax)
    kind: InvariantKind = "safety"


@dataclass
class SpecPlan:
    """Structured plan for a TLA+ module, emitted before the spec body."""
    module_name: str
    extends: list[str] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    init_sketch: str = ""                                    # NL description of initial state
    next_actions: list[NextAction] = field(default_factory=list)
    invariants: list[PlannedInvariant] = field(default_factory=list)
    fairness: str = ""                                       # "none" | "WF on X" | "SF on X" | free text
    notes: str = ""

    # ──────────────────────────── serialization ────────────────────────────

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpecPlan":
        actions = [NextAction(**a) for a in d.get("next_actions", []) if isinstance(a, dict)]
        invs = [PlannedInvariant(**i) for i in d.get("invariants", []) if isinstance(i, dict)]
        return cls(
            module_name=d.get("module_name", ""),
            extends=list(d.get("extends", [])),
            constants=list(d.get("constants", [])),
            variables=list(d.get("variables", [])),
            init_sketch=d.get("init_sketch", ""),
            next_actions=actions,
            invariants=invs,
            fairness=d.get("fairness", ""),
            notes=d.get("notes", ""),
        )

    # ──────────────────────────── rendering ───────────────────────────────

    def render_markdown(self) -> str:
        """Human-readable rendering for the analysis channel of training data."""
        lines = [f"## Plan for module {self.module_name}"]
        if self.extends:
            lines.append(f"**EXTENDS**: {', '.join(self.extends)}")
        if self.constants:
            lines.append(f"**CONSTANTS**: {', '.join(self.constants)}")
        if self.variables:
            lines.append(f"**VARIABLES**: {', '.join(self.variables)}")
        if self.init_sketch:
            lines.append(f"**Init**: {self.init_sketch}")
        if self.next_actions:
            lines.append("**Next actions**:")
            for a in self.next_actions:
                bits = [f"- `{a.name}`"]
                if a.guard:
                    bits.append(f"when {a.guard}")
                if a.effect:
                    bits.append(f"→ {a.effect}")
                lines.append(" ".join(bits))
        if self.invariants:
            lines.append("**Invariants**:")
            for inv in self.invariants:
                lines.append(f"- `{inv.name}` ({inv.kind}): {inv.statement}")
        if self.fairness:
            lines.append(f"**Fairness**: {self.fairness}")
        if self.notes:
            lines.append(f"**Notes**: {self.notes}")
        return "\n".join(lines)


# ──────────────────────────── tolerant parser ────────────────────────────

# Match a fenced JSON block ```json ... ``` or ```{...}```. The model is told
# to emit raw JSON but we accept either form so a fenced response still works.
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_plan(text: str) -> Optional[SpecPlan]:
    """Extract a SpecPlan from raw model output. Returns None on failure.

    The model is prompted to emit a JSON object directly, but real outputs vary:
    sometimes a ```json fence, sometimes prose around the JSON, sometimes a
    truncated trailing brace. This parser tries (in order):

      1. Whole-text json.loads
      2. First fenced ```json``` block
      3. Greedy {...} substring scan with brace balancing

    Returning None is a soft failure — the caller falls back to single-shot
    generation rather than crashing the pipeline.
    """
    if not text:
        return None

    # 1. whole-text
    try:
        return SpecPlan.from_dict(json.loads(text))
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    # 2. fenced
    m = _FENCED_JSON_RE.search(text)
    if m:
        try:
            return SpecPlan.from_dict(json.loads(m.group(1)))
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    # 3. brace-balanced scan
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        d = json.loads(candidate)
                        if isinstance(d, dict) and "module_name" in d:
                            return SpecPlan.from_dict(d)
                    except (json.JSONDecodeError, TypeError, KeyError):
                        pass
                    break
        start = text.find("{", start + 1)

    return None
