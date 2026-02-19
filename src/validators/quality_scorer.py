"""
quality_scorer.py — Heuristic richness scoring for TLA+ specs.

Produces a QualityScore (1–5 overall) that is used as a soft filter
during data collection.  High-quality specs (score >= 3) are preferred
for training and for few-shot selection during annotation.

Scoring rubric (each criterion adds points, max raw = 10 → normalised to 1-5):
  +2   : line_count 50–500 (not trivially small, not an uncommented wall)
  +1   : has_comments (any line with \\* or (* ... *))
  +2   : has_invariants (TypeOK, Safety, or named Invariant operators)
  +1   : has_liveness (ENABLED, <>, [], ~>, WF_, SF_ patterns)
  +2   : operator_richness >= 5 distinct TLA+ operators beyond /\\ and =
  +1   : has_primed_variables (Next-step reasoning, not just state definition)
  +1   : has_quantifiers (\\E, \\A — existential / universal quantification)

Research note
-------------
This scoring is intentionally heuristic.  It correlates with spec richness
but is not a ground truth quality metric.  The TLC validation tier is the
canonical quality signal.  Score is used for:
  1. Oversampling rich specs during few-shot prompt construction
  2. Prioritising which specs to send to the Ollama annotation pass first
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QualityScore:
    overall: int = 1              # 1–5
    line_count: int = 0
    has_comments: bool = False
    has_invariants: bool = False
    has_liveness: bool = False
    operator_richness: int = 0    # count of distinct TLA+ ops (beyond /\ and =)
    has_primed_variables: bool = False
    has_quantifiers: bool = False
    raw_score: int = 0            # unnormalised (0–10), for debugging


# Patterns for various TLA+ constructs
_COMMENT_RE = re.compile(r"(\\\*|^\s*\(\*)", re.MULTILINE)
_INVARIANT_RE = re.compile(
    r"^(TypeOK|Safety|Inv|.*Invariant)\s*==", re.MULTILINE | re.IGNORECASE
)
_LIVENESS_RE = re.compile(r"(ENABLED|<>|^\[\]|~>|WF_|SF_)", re.MULTILINE)
_PRIMED_RE = re.compile(r"[a-zA-Z_]\w*'")
_QUANTIFIER_RE = re.compile(r"(\\E\s|\\A\s|∃|∀)")

# TLA+ operators beyond basic /\ and = (proxy for spec sophistication)
_RICH_OPS = [
    r"\\E\s", r"\\A\s",           # quantifiers
    r"<>", r"\[\]", r"~>",        # temporal operators
    r"WF_", r"SF_",               # fairness
    r"ENABLED",
    r"UNCHANGED",
    r"CHOOSE",
    r"\\in\b", r"\\notin\b",
    r"SUBSET",
    r"UNION",
    r"\\times\b",
    r"Seq\s*\(",
    r"INSTANCE",
    r"MODULE\s+\w+\s+WITH",       # parameterised instantiation
]


def score(tla_content: str) -> QualityScore:
    """
    Score a TLA+ spec string on a 1–5 heuristic richness scale.

    Parameters
    ----------
    tla_content : str   Full .tla file text.

    Returns
    -------
    QualityScore with `overall` in [1, 5].
    """
    lines = tla_content.splitlines()
    line_count = len(lines)
    raw = 0

    # Line count heuristic
    if 50 <= line_count <= 500:
        raw += 2
    elif 20 <= line_count < 50 or 500 < line_count <= 1000:
        raw += 1

    has_comments = bool(_COMMENT_RE.search(tla_content))
    if has_comments:
        raw += 1

    has_invariants = bool(_INVARIANT_RE.search(tla_content))
    if has_invariants:
        raw += 2

    has_liveness = bool(_LIVENESS_RE.search(tla_content))
    if has_liveness:
        raw += 1

    op_count = sum(1 for pat in _RICH_OPS if re.search(pat, tla_content))
    if op_count >= 5:
        raw += 2
    elif op_count >= 2:
        raw += 1

    has_primed = bool(_PRIMED_RE.search(tla_content))
    if has_primed:
        raw += 1

    has_quant = bool(_QUANTIFIER_RE.search(tla_content))
    if has_quant:
        raw += 1

    # Normalise 0–10 → 1–5
    overall = max(1, min(5, round(raw / 2)))

    return QualityScore(
        overall=overall,
        line_count=line_count,
        has_comments=has_comments,
        has_invariants=has_invariants,
        has_liveness=has_liveness,
        operator_richness=op_count,
        has_primed_variables=has_primed,
        has_quantifiers=has_quant,
        raw_score=raw,
    )
