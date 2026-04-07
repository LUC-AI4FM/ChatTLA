"""Deterministic normalization of LLM-generated TLA+ output.

Closes the five hallucination categories from FormaLLM (docs/formallm.md §RQ4):

  1. Unicode operator substitution    (∧ → /\, ∈ → \in, ...)
  2. Cross-language syntax injection  (semicolons, backticks, END keyword)
  3. Reasoning / formatting leakage   (<think> blocks, markdown fences, prose)
  4. Generation length miscalibration (handled at the caller via length budgets;
                                       here we just stop greedy continuation past ====)
  5. Structural errors                (missing ====, missing/duplicate MODULE)

This module is intentionally narrow: rules that always apply, never modify
intent, and are safe to run on every spec. Heavier semantic fixers
(PlusCal removal, CONSTANT/VARIABLE rewriting, etc.) live in
src/inference/ollama_client._sanitize_spec and src/training/self_improve.fix_tla_syntax
and run on top of the output of this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Unicode → ASCII operator table
# ---------------------------------------------------------------------------
# Source: SANY rejects all of these. Built from the FormaLLM hallucination
# table plus the broader Unicode TLA+ rendering set used in tla2tools docs.

UNICODE_OP_TABLE: List[tuple[str, str]] = [
    # Logical
    ("\u2227", " /\\ "),   # ∧
    ("\u2228", " \\/ "),   # ∨
    ("\u00ac", " ~"),      # ¬
    ("\u21d2", " => "),    # ⇒
    ("\u21d4", " <=> "),   # ⇔
    # Set theory
    ("\u2208", " \\in "),       # ∈
    ("\u2209", " \\notin "),    # ∉
    ("\u2286", " \\subseteq "), # ⊆
    ("\u222a", " \\union "),    # ∪
    ("\u2229", " \\intersect "),# ∩
    ("\u00d7", " \\X "),        # ×
    ("\u2205", " {} "),         # ∅
    # Quantifiers
    ("\u2200", " \\A "),   # ∀
    ("\u2203", " \\E "),   # ∃
    # Relations / arithmetic
    ("\u2260", " # "),     # ≠
    ("\u2264", " <= "),    # ≤
    ("\u2265", " >= "),    # ≥
    ("\u00f7", " \\div "), # ÷
    # Arrows / temporal
    ("\u2192", " -> "),    # →
    ("\u21a6", " |-> "),   # ↦
    ("\u25fb", " []"),     # ◻ (always)
    ("\u25c7", " <>"),     # ◇ (eventually)
    ("\u22a2", " |- "),    # ⊢
    # ASCII alternates that some tokenizers emit
    ("\u00b7", " \\cdot "),# ·
]


_FENCE_RE = re.compile(r"^\s*```[\w+-]*\s*\n?|\n?\s*```\s*$", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_ONLY_RE = re.compile(r"<think>.*?(?=----\s*MODULE)", re.DOTALL | re.IGNORECASE)
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>", re.IGNORECASE
)
_HARMONY_ANY_TAG_RE = re.compile(r"<\|[a-z_]+\|>", re.IGNORECASE)
_MODULE_HEADER_RE = re.compile(r"^-{2,}\s*MODULE\s+(\w+)\s*-{2,}\s*$", re.MULTILINE)
_TERMINATOR_RE = re.compile(r"^={3,}\s*$", re.MULTILINE)
_LINE_COMMENT_RE = re.compile(r"\\\*.*$")
_BLOCK_COMMENT_RE = re.compile(r"\(\*.*?\*\)", re.DOTALL)


@dataclass
class NormalizationReport:
    """What the normalizer did. Useful as a reward-shaping signal: any non-empty
    field means the model emitted output that needed fixing, and policy gradient
    should be discouraged from relying on the cleanup."""

    unicode_ops_replaced: int = 0
    semicolons_stripped: int = 0
    backticks_stripped: int = 0
    fences_stripped: int = 0
    think_blocks_stripped: int = 0
    harmony_tags_stripped: int = 0
    module_header_added: bool = False
    terminator_added: bool = False
    duplicate_modules_removed: int = 0
    end_keyword_stripped: int = 0
    fixes: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return (
            self.unicode_ops_replaced == 0
            and self.semicolons_stripped == 0
            and self.backticks_stripped == 0
            and self.fences_stripped == 0
            and self.think_blocks_stripped == 0
            and self.harmony_tags_stripped == 0
            and not self.module_header_added
            and not self.terminator_added
            and self.duplicate_modules_removed == 0
            and self.end_keyword_stripped == 0
        )

    def merge(self, other: "NormalizationReport") -> "NormalizationReport":
        return NormalizationReport(
            unicode_ops_replaced=self.unicode_ops_replaced + other.unicode_ops_replaced,
            semicolons_stripped=self.semicolons_stripped + other.semicolons_stripped,
            backticks_stripped=self.backticks_stripped + other.backticks_stripped,
            fences_stripped=self.fences_stripped + other.fences_stripped,
            think_blocks_stripped=self.think_blocks_stripped + other.think_blocks_stripped,
            harmony_tags_stripped=self.harmony_tags_stripped + other.harmony_tags_stripped,
            module_header_added=self.module_header_added or other.module_header_added,
            terminator_added=self.terminator_added or other.terminator_added,
            duplicate_modules_removed=self.duplicate_modules_removed + other.duplicate_modules_removed,
            end_keyword_stripped=self.end_keyword_stripped + other.end_keyword_stripped,
            fixes=self.fixes + other.fixes,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def strip_reasoning_artifacts(text: str, report: NormalizationReport | None = None) -> str:
    """Remove harmony channel markers, <think> blocks, and markdown fences.

    Always safe: never touches anything inside the ---- MODULE ... ==== block."""
    rep = report or NormalizationReport()

    # Harmony: keep only what comes after the last "<|channel|>final<|message|>".
    final_matches = list(_HARMONY_FINAL_RE.finditer(text))
    if final_matches:
        text = text[final_matches[-1].end():]
        rep.fixes.append("harmony_final_extracted")

    # Strip any remaining harmony tags.
    new_text, n = _HARMONY_ANY_TAG_RE.subn("", text)
    if n:
        rep.harmony_tags_stripped += n
        text = new_text

    # Strip <think>...</think> pairs.
    new_text, n = _THINK_RE.subn("", text)
    if n:
        rep.think_blocks_stripped += n
        text = new_text

    # Open-only <think> followed eventually by ---- MODULE: drop the prefix.
    if "<think>" in text.lower() and re.search(r"----\s*MODULE", text):
        new_text, n = _THINK_OPEN_ONLY_RE.subn("", text)
        if n:
            rep.think_blocks_stripped += n
            text = new_text

    # Strip markdown code fences.
    new_text, n = _FENCE_RE.subn("", text)
    if n:
        rep.fences_stripped += n
        text = new_text

    return text


def extract_module_block(text: str) -> str | None:
    """Return the first ---- MODULE ... ==== block.

    If a MODULE header is present but the ==== terminator is missing,
    take everything from the header to end-of-text (the terminator will be
    added later by `_ensure_module_terminator`)."""
    m = re.search(r"(-{2,}\s*MODULE\b.*?={3,})", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"-{2,}\s*MODULE\b", text)
    if m:
        return text[m.start():].strip()
    return None


def _strip_semicolons(spec: str, rep: NormalizationReport) -> str:
    """Remove stray semicolons that aren't legal TLA+. Skip comments and strings."""
    out_lines: List[str] = []
    for line in spec.split("\n"):
        # Preserve comment text as-is (semicolons inside comments are fine).
        comment_idx = line.find("\\*")
        head, tail = (line[:comment_idx], line[comment_idx:]) if comment_idx >= 0 else (line, "")
        # Don't strip if this is a PlusCal block comment line (handled elsewhere).
        if "(*" in head and "*)" not in head:
            out_lines.append(line)
            continue
        new_head = head.replace(";", "")
        if new_head != head:
            rep.semicolons_stripped += head.count(";")
        out_lines.append(new_head + tail)
    return "\n".join(out_lines)


def _strip_backticks(spec: str, rep: NormalizationReport) -> str:
    """Backticks aren't valid TLA+ except inside comments."""
    if "`" not in spec:
        return spec
    out: List[str] = []
    for line in spec.split("\n"):
        if "\\*" in line or "(*" in line:
            out.append(line)
            continue
        if "`" in line:
            rep.backticks_stripped += line.count("`")
            line = line.replace("`", "")
        out.append(line)
    return "\n".join(out)


def _normalize_unicode_ops(spec: str, rep: NormalizationReport) -> str:
    """Replace Unicode TLA+ operators with their ASCII equivalents.

    Note: we do NOT collapse multi-whitespace afterwards even though the
    replacement table inserts padding spaces around each operator. The reason
    is that TLA+ junction-list parsing is **column-sensitive**: the parser
    requires the `/\\` (or `\\/`) items in a junction list to share an
    indentation column. Collapsing `  ` → ` ` shifts those columns and
    silently breaks otherwise-correct specs (the failure looks like a parse
    error inside the action body). The extra space inside operator boundaries
    is harmless to SANY."""
    for src, dst in UNICODE_OP_TABLE:
        if src in spec:
            count = spec.count(src)
            rep.unicode_ops_replaced += count
            spec = spec.replace(src, dst)
    return spec


def _strip_end_keyword(spec: str, rep: NormalizationReport) -> str:
    """Pascal/SQL-style `END` as a standalone token isn't valid TLA+.
    PlusCal `end algorithm` etc. are handled by ollama_client._sanitize_spec.
    Only strip a bare uppercase `END` line that isn't followed by an identifier."""
    new_spec, n = re.subn(r"^\s*END\s*$", "", spec, flags=re.MULTILINE)
    if n:
        rep.end_keyword_stripped += n
    return new_spec


def _ensure_module_terminator(spec: str, rep: NormalizationReport) -> str:
    if not _TERMINATOR_RE.search(spec):
        spec = spec.rstrip() + "\n" + ("=" * 78) + "\n"
        rep.terminator_added = True
        rep.fixes.append("terminator_added")
    return spec


def _dedupe_module_headers(spec: str, rep: NormalizationReport) -> str:
    """If the model emitted multiple ---- MODULE Foo ---- headers, keep the first."""
    headers = list(_MODULE_HEADER_RE.finditer(spec))
    if len(headers) <= 1:
        return spec
    keep = headers[0]
    extras = headers[1:]
    rep.duplicate_modules_removed = len(extras)
    # Remove extras line by line.
    new_spec = spec
    for h in reversed(extras):
        new_spec = new_spec[: h.start()] + new_spec[h.end():]
    return new_spec


def normalize_spec(text: str) -> tuple[str, NormalizationReport]:
    """Run the full deterministic normalizer.

    Args:
        text: Raw model output. May include harmony tags, <think>, fences, and prose.

    Returns:
        (cleaned_spec, report). cleaned_spec is *just* the ---- MODULE ... ====
        block when one can be found, otherwise the cleaned text. Report enumerates
        every fix applied — useful for reward shaping (rep.clean is True only if
        the model emitted normalizer-clean output).
    """
    report = NormalizationReport()
    if not text:
        return "", report

    text = strip_reasoning_artifacts(text, report)

    # Try to localize on the module block as early as possible — once we have it,
    # we never operate on prose around it again.
    block = extract_module_block(text)
    if block is not None:
        spec = block
    else:
        spec = text.strip()
        # Likely missing the header. If we can find a `Init ==` and a `VARIABLES`
        # line we'll synthesize a header at write time; otherwise leave it alone.
        if re.search(r"^VARIABLES?\b", spec, re.MULTILINE) or re.search(r"^Init\s*==", spec, re.MULTILINE):
            report.module_header_added = True
            report.fixes.append("module_header_synthesized")
            name = "GeneratedSpec"
            spec = f"---- MODULE {name} ----\n" + spec

    spec = _normalize_unicode_ops(spec, report)
    spec = _strip_backticks(spec, report)
    spec = _strip_semicolons(spec, report)
    spec = _strip_end_keyword(spec, report)
    spec = _dedupe_module_headers(spec, report)
    spec = _ensure_module_terminator(spec, report)

    # Final whitespace tidy: collapse runs of blank lines.
    spec = re.sub(r"\n{3,}", "\n\n", spec).strip() + "\n"
    return spec, report
