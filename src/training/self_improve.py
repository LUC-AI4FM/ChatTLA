"""
self_improve.py — Autonomous self-improvement loop for ChatTLA.

Architecture
------------
    ┌─────────────┐     ┌──────────┐     ┌──────────────┐
    │  Generate   │────▸│  SANY    │────▸│ Python Fixer │
    │  (Ollama)   │     │ Validate │     │  (rule-based)│
    └─────────────┘     └──────────┘     └──────────────┘
          ▲                                       │
          │                    ┌──────────────────┘
          │                    ▼
    ┌─────────────┐     ┌──────────────┐
    │  Retrain    │◂────│  Augmented   │
    │  + Deploy   │     │  Dataset     │
    └─────────────┘     └──────────────┘

Loop phases
-----------
1. **Generate**: ChatTLA generates specs for prompts from the benchmark suite
   and synthetic prompts.
2. **Validate**: Each spec is run through SANY.  Passing specs become
   `spec_generation` training examples.
3. **Fix**: Failing specs go through a Python rule-engine that applies
   deterministic syntax fixes (the "Python engine between inference").
   If the fixed version passes SANY, we create a `bug_fix` training triple:
   (buggy_spec, sany_error, fixed_spec).
4. **Augment**: New examples are appended to `data/processed/augmented.jsonl`.
5. **Rebuild**: When enough examples accumulate, re-run `dataset_builder`
   (which merges the augmented data into the training set).
6. **Retrain**: Launch a short fine-tuning run on the enriched dataset.
7. **Deploy**: Merge LoRA → GGUF → Ollama.
8. **Repeat**: Jump to step 1 with the improved model.

The loop is designed to run unattended (tmux) for hours.  Each iteration
takes ~5-10 min for generation + ~80 min for retraining.

Usage
-----
    # Run one iteration (generate + fix + augment, no retrain)
    python -m src.training.self_improve --iterations 1 --no-retrain

    # Full autonomous loop with retraining every 20 new examples
    python -m src.training.self_improve --iterations 10 --retrain-threshold 20

    # Quick dry run
    python -m src.training.self_improve --iterations 1 --prompts 3 --no-retrain
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_REPO_ROOT       = Path(__file__).resolve().parents[2]
_AUGMENTED_JSONL = _REPO_ROOT / "data" / "processed" / "augmented.jsonl"
_BENCHMARK_JSON  = _REPO_ROOT / "data" / "benchmarks" / "benchmark_suite.json"
_TRAIN_JSONL     = _REPO_ROOT / "data" / "processed" / "train.jsonl"
_EVAL_JSONL      = _REPO_ROOT / "data" / "processed" / "eval.jsonl"

# Developer prompt matching dataset_builder.py format
_DEVELOPER_PROMPT = """\
You are ChatTLA, an expert at writing verified TLA+ formal specifications.
When asked to write a TLA+ spec, follow these rules exactly:
1. Start the module with ---- MODULE <ModuleName> ----
2. End with ====
3. Include EXTENDS, VARIABLES, Init, Next, and Spec operators
4. After the TLA+ module, append a TLC configuration block:
   SPECIFICATION Spec
   INVARIANT TypeOK   (if TypeOK is defined)
5. Output only valid TLA+ code. No markdown fences, no explanation outside the spec.
Reasoning: medium\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FixResult:
    """Result of attempting to fix a TLA+ spec."""
    original_spec: str
    fixed_spec: str
    sany_errors: str
    fixes_applied: list[str] = field(default_factory=list)
    passed_sany: bool = False


@dataclass
class IterationStats:
    """Statistics for one self-improvement iteration."""
    prompts_tried: int = 0
    specs_generated: int = 0
    sany_pass_raw: int = 0
    sany_pass_fixed: int = 0
    bug_fix_examples: int = 0
    spec_gen_examples: int = 0
    total_new_examples: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Python syntax fixer — deterministic rule engine
# ─────────────────────────────────────────────────────────────────────────────

def fix_tla_syntax(spec: str, sany_errors: str = "") -> FixResult:
    """
    Apply deterministic Python-based fixes for common TLA+ syntax errors.

    This is the "Python engine between inference" — it catches patterns that
    the LLM consistently gets wrong and applies mechanical corrections.

    Returns a FixResult with the fixed spec and list of applied fixes.
    """
    result = FixResult(
        original_spec=spec,
        fixed_spec=spec,
        sany_errors=sany_errors,
    )
    fixed = spec

    # ── Fix 1: Remove PlusCal blocks ──────────────────────────────────────
    # Pattern A: full block  (* --algorithm ... end algorithm; *)
    pluscal_pat = r"\(\*\s*--(?:fair\s+)?algorithm\b.*?end\s+algorithm\s*;?\s*\*\)"
    if re.search(pluscal_pat, fixed, re.DOTALL | re.IGNORECASE):
        fixed = re.sub(pluscal_pat, "", fixed, flags=re.DOTALL | re.IGNORECASE)
        result.fixes_applied.append("removed PlusCal block")
    else:
        # Pattern B: incomplete PlusCal — model starts (* --algorithm but
        # never closes with end algorithm; *).  Strip from the opening (* --algorithm
        # to the ==== (keep everything before it and the ==== itself).
        incomplete_pluscal = re.search(
            r"\(\*\s*--(?:fair\s+)?algorithm\b", fixed, re.IGNORECASE
        )
        if incomplete_pluscal:
            # Keep the spec up to right before the PlusCal block
            before = fixed[:incomplete_pluscal.start()].rstrip()
            # Ensure we still have the ==== delimiter
            if "====" not in before:
                before += "\n\n===="
            fixed = before
            result.fixes_applied.append("removed incomplete PlusCal block")

        # Pattern C: bare --algorithm (no (* wrapper)
        bare_pluscal = re.search(
            r"^--(?:fair\s+)?algorithm\b", fixed, re.MULTILINE | re.IGNORECASE
        )
        if bare_pluscal:
            before = fixed[:bare_pluscal.start()].rstrip()
            if "====" not in before:
                before += "\n\n===="
            fixed = before
            result.fixes_applied.append("removed bare PlusCal algorithm block")

    # If the spec has a BEGIN TRANSLATION / END TRANSLATION block (PlusCal
    # translator output), extract the header + translation + footer.
    trans_match = re.search(
        r"\\?\*?\s*BEGIN TRANSLATION.*?\n(.*?)\\?\*?\s*END TRANSLATION",
        fixed, re.DOTALL,
    )
    if trans_match:
        header_match = re.search(
            r"(----.*?MODULE.*?)(?:\(\*\s*--(?:fair\s+)?algorithm|--(?:fair\s+)?algorithm)",
            fixed, re.DOTALL,
        )
        footer_match = re.search(r"\\?\*?\s*END TRANSLATION.*?\n(.*)", fixed, re.DOTALL)
        header = header_match.group(1).strip() if header_match else ""
        translation = trans_match.group(1).strip()
        footer = footer_match.group(1).strip() if footer_match else "===="
        fixed = f"{header}\n\n{translation}\n\n{footer}"
        if "====" not in fixed:
            fixed += "\n===="
        result.fixes_applied.append("extracted TLA+ translation from PlusCal")

    # Remove standalone PlusCal keywords
    pluscal_kw = re.findall(r"^\s*(begin|end\s+algorithm|macro|procedure)\b.*$", fixed, re.MULTILINE | re.IGNORECASE)
    if pluscal_kw:
        fixed = re.sub(r"^\s*(begin|end\s+algorithm|macro|procedure)\b.*$", "", fixed, flags=re.MULTILINE | re.IGNORECASE)
        result.fixes_applied.append("removed PlusCal keywords")

    # ── Fix 2: Remove markdown fences ─────────────────────────────────────
    if re.search(r"^```", fixed, re.MULTILINE):
        fixed = re.sub(r"^```\w*\s*$", "", fixed, flags=re.MULTILINE)
        result.fixes_applied.append("removed markdown fences")

    # ── Fix 3: Fix VARIABLES declaration (missing commas) ─────────────────
    # Pattern: VARIABLES followed by identifiers on separate lines without commas
    var_block = re.search(r"(VARIABLES?\s*\n)((?:\s+\w+\s*(?:\\.*)?(?:\n|$))+)", fixed)
    if var_block:
        var_lines = var_block.group(2).strip().splitlines()
        # Check if lines lack commas between variables
        needs_fix = False
        for i, line in enumerate(var_lines[:-1]):
            stripped = line.rstrip()
            if stripped and not stripped.endswith(",") and not stripped.endswith("\\*"):
                needs_fix = True
                break
        if needs_fix:
            new_lines = []
            for i, line in enumerate(var_lines):
                stripped = line.rstrip()
                # Remove trailing comments for processing
                comment = ""
                if "\\*" in stripped:
                    parts = stripped.split("\\*", 1)
                    stripped = parts[0].rstrip()
                    comment = " \\*" + parts[1]
                # Add comma if not last line and doesn't have one
                if i < len(var_lines) - 1 and not stripped.endswith(","):
                    stripped = stripped + ","
                new_lines.append(stripped + comment)
            new_var_block = var_block.group(1) + "\n".join("    " + l.strip() for l in new_lines) + "\n"
            fixed = fixed[:var_block.start()] + new_var_block + fixed[var_block.end():]
            result.fixes_applied.append("added commas to VARIABLES declaration")

    # ── Fix 3b: Fix CONSTANTS with inline constraints ─────────────────────
    # Models produce: CONSTANTS N \in Nat, FORK \in 1..N
    # TLA+ requires: CONSTANT N, FORK   (no constraints — use ASSUME instead)
    const_block = re.search(r"^(CONSTANTS?)\s+(.+?)(?=\n\n|\n[A-Z])", fixed, re.MULTILINE | re.DOTALL)
    if const_block:
        keyword = const_block.group(1)
        body = const_block.group(2).strip()
        # Check if it has \in or \setminus or other constraint operators
        if re.search(r"\\in\b|\\setminus|\\subseteq|\\cup|\\cap", body):
            # Extract just variable names (strip constraints)
            names = []
            for part in re.split(r",\s*(?=\w)", body):
                m = re.match(r"(\w+)", part.strip())
                if m:
                    names.append(m.group(1))
            if names:
                new_const = f"{keyword} {', '.join(names)}"
                fixed = fixed[:const_block.start()] + new_const + fixed[const_block.end():]
                result.fixes_applied.append("stripped inline constraints from CONSTANTS")

    # ── Fix 4: Fix double-prime (x'' → x') ───────────────────────────────
    if re.search(r"\w''", fixed):
        fixed = re.sub(r"(\w)''", r"\1'", fixed)
        result.fixes_applied.append("fixed double-prime to single-prime")

    # ── Fix 5: Fix vars == {...} → vars == <<...>> ────────────────────────
    vars_set = re.search(r"(vars\s*==\s*)\{([^}]+)\}", fixed)
    if vars_set:
        fixed = fixed[:vars_set.start()] + vars_set.group(1) + "<<" + vars_set.group(2) + ">>" + fixed[vars_set.end():]
        result.fixes_applied.append("fixed vars set to tuple")

    # ── Fix 6: Fix UNCHANGED single variable (no <<>>) ───────────────────
    # UNCHANGED x, y → UNCHANGED <<x, y>>
    unchanged_multi = re.findall(r"UNCHANGED\s+(\w+\s*,\s*\w+(?:\s*,\s*\w+)*)", fixed)
    for match in unchanged_multi:
        old = f"UNCHANGED {match}"
        new = f"UNCHANGED <<{match}>>"
        fixed = fixed.replace(old, new, 1)
        result.fixes_applied.append("wrapped UNCHANGED multi-variable in tuple")

    # ── Fix 7: Fix CONSTANT with value assignment ─────────────────────────
    # CONSTANT N = 5 → CONSTANT N
    const_assigns = re.findall(r"^(\s*CONSTANTS?\s+\w+)\s*=\s*\S+", fixed, re.MULTILINE)
    if const_assigns:
        for old in const_assigns:
            fixed = re.sub(
                r"^(\s*CONSTANTS?\s+\w+)\s*=\s*\S+",
                r"\1",
                fixed, count=1, flags=re.MULTILINE,
            )
        result.fixes_applied.append("removed CONSTANT value assignment")

    # ── Fix 8: Fix quoted strings (TLA+ uses double quotes only) ─────────
    # Some models produce "idle" with escaped quotes
    if '\\"' in fixed:
        # Convert escaped quotes to regular quotes
        fixed = fixed.replace('\\"', '"')
        result.fixes_applied.append("unescaped quotes")

    # ── Fix 8b: Replace \notin (not standard in SANY) with ~(\in) ────────
    if "\\notin" in fixed:
        fixed = re.sub(r"(\w+)\s*\\notin\s*", r"~(\1 \\in ", fixed)
        # Close the parens — heuristic: add ) at end of line
        fixed = re.sub(r"(~\([^)\n]+)$", r"\1)", fixed, flags=re.MULTILINE)
        result.fixes_applied.append("replaced \\notin with ~(\\in)")

    # ── Fix 8c: Remove ASSUME blocks that reference VARIABLEs ────────────
    # SANY requires ASSUME to be level 0 (constants only). Models often
    # put variables in ASSUME which gives "Level error".
    # Also remove ASSUMEs that use non-standard operators.
    if "ASSUME" in fixed:
        # Extract variable names
        var_match = re.search(r"VARIABLES?\s+(.+?)(?:\n\n|\n[A-Z])", fixed, re.DOTALL)
        var_names = set()
        if var_match:
            var_names = {v.strip().rstrip(",") for v in re.split(r"[,\n]", var_match.group(1)) if v.strip().rstrip(",")}

        # Check each ASSUME line
        assume_lines_to_remove = []
        for m in re.finditer(r"^(\s*ASSUME\b.*)$", fixed, re.MULTILINE):
            assume_text = m.group(1)
            # Remove if it contains a variable name or if SANY reported level error
            should_remove = "level error" in sany_errors.lower() or "otin" in sany_errors
            if var_names:
                for vn in var_names:
                    if vn and re.search(rf"\b{re.escape(vn)}\b", assume_text):
                        should_remove = True
                        break
            if should_remove:
                assume_lines_to_remove.append(m.group(0))

        for line in assume_lines_to_remove:
            fixed = fixed.replace(line, "", 1)
            result.fixes_applied.append("removed ASSUME referencing variable")

        # Also remove multi-line ASSUME blocks
        if "otin" in sany_errors or "ASSUME" in sany_errors:
            fixed = re.sub(r"^\s*ASSUME\s*\n(?:\s+.*\n)*", "\n", fixed, flags=re.MULTILINE)
            result.fixes_applied.append("removed problematic ASSUME block")


    # ── Fix 9: Fix missing ==== ───────────────────────────────────────────
    if "====" not in fixed:
        fixed = fixed.rstrip() + "\n\n===="
        result.fixes_applied.append("added missing ==== delimiter")

    # ── Fix 10: Fix MODULE header (needs at least 4 dashes each side) ─────
    header = re.search(r"^(-+)\s*MODULE\s+(\w+)\s*(-+)", fixed, re.MULTILINE)
    if header:
        if len(header.group(1)) < 4 or len(header.group(3)) < 4:
            new_header = f"---- MODULE {header.group(2)} ----"
            fixed = fixed[:header.start()] + new_header + fixed[header.end():]
            result.fixes_applied.append("fixed MODULE header dashes")

    # ── Fix 11: (merged into Fix 15 — unconditional alignment) ─────────── 

    # ── Fix 12: Remove trailing garbage after ==== ────────────────────────
    m = re.search(r"(----\s*MODULE\b.*?====)", fixed, re.DOTALL)
    if m:
        fixed = m.group(1)
        if fixed != result.fixed_spec:
            result.fixes_applied.append("truncated after ====")

    # ── Fix 13: Fix empty-line-separated conjunctions ─────────────────────
    # Some models put blank lines between /\ conjuncts, which breaks SANY
    # e.g.:
    #   Init ==
    #     /\ x = 0
    #
    #     /\ y = 0
    fixed = re.sub(r"(\n\s*/\\[^\n]+)\n\s*\n(\s*/\\)", r"\1\n\2", fixed)

    # ── Fix 14: Fix \\in vs \in (double-backslash in non-raw context) ─────
    # Models sometimes produce \\in, \\cup, \\subseteq etc. with double backslash
    for op in ["in", "cup", "cap", "subseteq", "union", "notin", "times",
               "leq", "geq", "div", "o", "circ", "land", "lor", "lnot",
               "equiv", "neg", "A", "E"]:
        fixed = re.sub(rf"\\\\{op}\b", rf"\\{op}", fixed)

    # ── Fix 15: Conjunction/disjunction indent normalization ─────────────
    # Only run when SANY reports indent issues — the fixer can break
    # quantifier scoping if applied blindly.
    if "not properly indented" in sany_errors.lower() or "alignment" in sany_errors.lower():
        old_fixed = fixed
        fixed = _fix_conjunction_indent(fixed)
        if fixed != old_fixed:
            result.fixes_applied.append("normalized conjunction/disjunction alignment")

    # ── Fix 16: Remove \notin if still present after Fix 8b ──────────────
    # Fallback: just replace \notin X with ~(var \in X) more aggressively
    _notin_safety = 0
    while "\\notin" in fixed and _notin_safety < 10:
        old_fixed = fixed
        fixed = re.sub(
            r"(\b\w+(?:\[\w+\])?)\s*\\notin\s+(\w+(?:\([^)]*\))?)",
            r"~(\1 \\in \2)",
            fixed, count=1,
        )
        if fixed == old_fixed:
            # regex didn't match — forcibly remove remaining \notin
            fixed = fixed.replace("\\notin", "\\in", 1)
            result.fixes_applied.append("force-replaced \\notin with \\in")
            break
        result.fixes_applied.append("replaced remaining \\notin")
        _notin_safety += 1

    # ── Fix 17: Fix EXTENDS with non-existent modules ────────────────────
    # SANY only knows: Naturals, Integers, Reals, Sequences, FiniteSets,
    # TLC, Bags, RealTime, Toolbox. Remove unknown modules.
    known_modules = {"Naturals", "Integers", "Reals", "Sequences",
                     "FiniteSets", "TLC", "Bags", "RealTime", "Toolbox",
                     "TLAPS", "Apalache"}
    extends_match = re.search(r"^EXTENDS\s+(.+)$", fixed, re.MULTILINE)
    if extends_match:
        mods = [m.strip() for m in extends_match.group(1).split(",")]
        valid_mods = [m for m in mods if m in known_modules]
        if valid_mods and len(valid_mods) < len(mods):
            new_extends = "EXTENDS " + ", ".join(valid_mods)
            fixed = fixed[:extends_match.start()] + new_extends + fixed[extends_match.end():]
            result.fixes_applied.append(f"removed unknown EXTENDS modules")
        elif not valid_mods and len(mods) > 0 and mods != [""]:
            # All modules unknown — remove EXTENDS line entirely
            fixed = fixed[:extends_match.start()] + fixed[extends_match.end():]
            result.fixes_applied.append("removed EXTENDS with all unknown modules")

    # ── Fix 18: Fix single-line ASSUME with \notin cleanup artifacts ─────
    # Remove ASSUME lines that are now syntactically broken
    fixed = re.sub(r"^\s*ASSUME\s*~?\s*\(\s*\)\s*$", "", fixed, flags=re.MULTILINE)

    # ── Fix 19: Auto-define `vars` if referenced but not defined ──────────
    # Models often use WF_vars(Next) or SF_vars(Next) without defining vars.
    if re.search(r"\bvars\b", fixed) and not re.search(r"^\s*vars\s*==", fixed, re.MULTILINE):
        # Extract variable names from VARIABLES declaration
        var_match = re.search(r"VARIABLES?\s+(.+?)(?=\n\n|\n[A-Z])", fixed, re.DOTALL)
        if var_match:
            var_names = [v.strip().rstrip(",") for v in re.split(r"[,\n]", var_match.group(1))
                         if v.strip().rstrip(",")]
            if var_names:
                vars_tuple = "<<" + ", ".join(var_names) + ">>"
                # Insert vars definition after the VARIABLES block
                # Find the end of the VARIABLES block (next blank line)
                var_end = var_match.end()
                # Look for the next blank line after VARIABLES
                next_blank = fixed.find("\n\n", var_end)
                if next_blank != -1:
                    insert_pos = next_blank
                else:
                    insert_pos = var_end
                vars_def = f"\n\nvars == {vars_tuple}\n"
                fixed = fixed[:insert_pos] + vars_def + fixed[insert_pos:]
                result.fixes_applied.append(f"auto-defined vars == {vars_tuple}")

    # ── Fix 20: Replace inline WF_vars/SF_vars if vars still undefined ────
    # Fallback: directly substitute the variable tuple into WF/SF expressions
    if re.search(r"[WS]F_vars\b", fixed) and not re.search(r"^\s*vars\s*==", fixed, re.MULTILINE):
        var_match = re.search(r"VARIABLES?\s+(.+?)(?=\n\n|\n[A-Z])", fixed, re.DOTALL)
        if var_match:
            var_names = [v.strip().rstrip(",") for v in re.split(r"[,\n]", var_match.group(1))
                         if v.strip().rstrip(",")]
            if var_names:
                vars_tuple = "<<" + ", ".join(var_names) + ">>"
                fixed = re.sub(r"WF_vars\b", f"WF_{vars_tuple}", fixed)
                fixed = re.sub(r"SF_vars\b", f"SF_{vars_tuple}", fixed)
                result.fixes_applied.append("inlined vars tuple in WF/SF")

    # ── Fix 21: Remove bare module-level expressions (BM019 pattern) ──────
    # Models sometimes emit bare `x \in Type` lines directly at module level
    # (e.g., after a "(* Type definitions *)" comment) instead of inside
    # TypeOK ==.  These cause parse errors because TLA+ doesn't allow bare
    # expressions at module level.
    # Strategy: if such lines exist, collect them into a TypeOK definition
    # if TypeOK isn't already defined, otherwise just remove them.
    bare_expr_pat = re.compile(
        r"^(\w[\w\[\]]*(?:\[[\w, ]+\])?)\s*\\in\s+[^\n]+$", re.MULTILINE
    )
    # Only apply outside operator bodies: find lines that are at column 0 or
    # preceded only by comments and appear before the first == definition
    def _is_bare_module_level(spec_text: str) -> list[tuple[str,str]]:
        """Return (full_line, variable_part) for bare-level \in expressions."""
        found = []
        in_body = False
        for line in spec_text.splitlines():
            stripped = line.strip()
            if re.match(r"^\w+\s*(==|\(==\))", stripped):
                in_body = True
            if stripped == "" or stripped.startswith("\\*") or stripped.startswith("(*"):
                continue
            if not in_body and bare_expr_pat.match(line.strip()):
                found.append((line, line.strip()))
        return found

    bare_lines = _is_bare_module_level(fixed)
    if bare_lines:
        # Remove them from the spec — only try the fallback if the first
        # replacement didn't work to avoid accidentally removing matching
        # substrings inside operator bodies.
        for full_line, _ in bare_lines:
            new_fixed = fixed.replace(full_line + "\n", "", 1)
            if new_fixed == fixed:
                # No exact newline match — try without newline
                new_fixed = fixed.replace(full_line, "", 1)
            fixed = new_fixed
        result.fixes_applied.append("removed bare module-level \\in expressions")

    # ── Fix 22: Detect and patch truncated specs ──────────────────────────
    # If the spec has no ==== ending, it was likely truncated.  We try to
    # close any open operator definitions and add a minimal terminator.
    if "====" not in fixed:
        # Count unclosed lines — if the last meaningful line doesn't end an
        # operator (doesn't look like a complete expression), add minimal close.
        stripped_lines = [l.rstrip() for l in fixed.splitlines() if l.strip()]
        if stripped_lines:
            last = stripped_lines[-1]
            # If last line looks like a truncated operator call / expression,
            # try to close the current operator with a simple TRUE and move on.
            if not re.match(r"^\s*(====|Spec\s*==|THEOREM|PROPERTY)", last):
                # Add ==== to close the module — Fix 9 might add it too, but
                # we also want to try to make the preceding incomplete
                # operator syntactically valid by appending TRUE if needed.
                if last.endswith("==") or last.endswith("/\\") or last.endswith("\\/"):
                    fixed = fixed.rstrip() + "\n    TRUE\n\n===="
                else:
                    fixed = fixed.rstrip() + "\n\n===="
                result.fixes_applied.append("closed truncated spec")

    result.fixed_spec = fixed.strip()
    return result


def _fix_conjunction_indent(spec: str) -> str:
    """
    Normalize indentation of /\\ and \\/ lines within operator definitions.

    SANY requires that all conjuncts/disjuncts in a list are at the same
    indentation level.  This function finds contiguous blocks of /\\ or \\/
    lines and aligns them to the first /\\ (or \\/) in each block.
    """
    lines = spec.splitlines()
    out: list[str] = []
    i = 0
    conj_re = re.compile(r"(\s*)(?:/\\|\\/)")

    while i < len(lines):
        line = lines[i]
        # Detect start of a conjunction/disjunction block:
        # a line that is followed by /\ or \/ lines
        if i + 1 < len(lines) and conj_re.match(lines[i + 1]):
            # Find the indentation of the first /\ or \/ line
            m = conj_re.match(lines[i + 1])
            target_indent = m.group(1) if m else "    "
            out.append(line)
            i += 1
            # Align all subsequent /\ and \/ lines to the same indent
            while i < len(lines):
                if conj_re.match(lines[i]):
                    stripped = lines[i].lstrip()
                    out.append(target_indent + stripped)
                    i += 1
                elif lines[i].strip() == "":
                    # blank line breaks the block
                    break
                elif re.match(r"\s+\S", lines[i]) and not re.match(r"\s*(ELSE|IN|THEN|LET|IF|CASE|OTHER)\b", lines[i]):
                    # continuation line within a conjunct — keep relative indent
                    out.append(lines[i])
                    i += 1
                else:
                    break
            continue

        # Handle standalone \/ block (disjunction not preceded by /\)
        if conj_re.match(line):
            m = conj_re.match(line)
            target_indent = m.group(1) if m else "    "
            stripped = line.lstrip()
            out.append(target_indent + stripped)
            i += 1
            while i < len(lines) and conj_re.match(lines[i]):
                stripped = lines[i].lstrip()
                out.append(target_indent + stripped)
                i += 1
            continue

        out.append(line)
        i += 1
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# SANY validation wrapper
# ─────────────────────────────────────────────────────────────────────────────

def validate_with_sany(spec: str) -> tuple[bool, str]:
    """Run SANY on a spec and return (is_valid, error_text)."""
    from src.validators.sany_validator import validate_string

    m = re.search(r"----\s*MODULE\s+(\w+)", spec)
    module_name = m.group(1) if m else "Spec"

    result = validate_string(spec, module_name=module_name)
    errors = "\n".join(result.errors) if result.errors else result.raw_output[-500:]
    return result.valid, errors


# ─────────────────────────────────────────────────────────────────────────────
# Training example builders (ChatML format for DeepSeek R1)
# ─────────────────────────────────────────────────────────────────────────────

def build_spec_gen_example(prompt: str, spec: str) -> dict:
    """Build a spec_generation training example in ChatML format."""
    return {"messages": [
        {"role": "system",    "content": _DEVELOPER_PROMPT},
        {"role": "user",      "content": f"Write a TLA+ specification for the following:\n\n{prompt}"},
        {"role": "assistant", "content": spec.strip()},
    ]}


def build_bug_fix_example(prompt: str, buggy_spec: str, sany_errors: str, fixed_spec: str) -> dict:
    """Build a bug_fix training example in ChatML format."""
    return {"messages": [
        {"role": "system", "content": _DEVELOPER_PROMPT},
        {
            "role": "user",
            "content": (
                f"This TLA+ spec has syntax errors:\n\n"
                f"SANY errors:\n{sany_errors}\n\n"
                f"Buggy spec:\n{buggy_spec.strip()}\n\n"
                f"Fix the spec."
            ),
        },
        {"role": "assistant", "content": fixed_spec.strip()},
    ]}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt generation
# ─────────────────────────────────────────────────────────────────────────────

def load_prompts(limit: Optional[int] = None) -> list[dict]:
    """Load benchmark prompts + generate synthetic variations."""
    prompts = []

    # Load benchmark suite
    if _BENCHMARK_JSON.exists():
        with open(_BENCHMARK_JSON) as f:
            benchmarks = json.load(f)
        for bm in benchmarks:
            prompts.append({
                "id": bm["id"],
                "prompt": bm["description"],
                "module_hint": bm["name"].replace(" ", "").replace("'", ""),
            })

    # Synthetic prompt variations (rewordings and new problems)
    synthetic = [
        {"id": "SYN001", "prompt": "A token ring protocol where N nodes pass a token in a circle. Only the node holding the token may enter the critical section.", "module_hint": "TokenRing"},
        {"id": "SYN002", "prompt": "A simple clock that counts from 0 to MAX and wraps around.", "module_hint": "Clock"},
        {"id": "SYN003", "prompt": "A traffic light controller that cycles through Red, Green, Yellow states.", "module_hint": "TrafficLight"},
        {"id": "SYN004", "prompt": "A bank account with deposit and withdraw operations. Balance must never go negative.", "module_hint": "BankAccount"},
        {"id": "SYN005", "prompt": "A simple stack (LIFO) data structure with push and pop operations. The stack has a maximum capacity.", "module_hint": "BoundedStack"},
        {"id": "SYN006", "prompt": "Peterson's algorithm for mutual exclusion between two processes.", "module_hint": "Peterson"},
        {"id": "SYN007", "prompt": "A leader election algorithm in a ring of N processes where each process has a unique ID. The process with the highest ID becomes the leader.", "module_hint": "RingLeader"},
        {"id": "SYN008", "prompt": "A simple key-value store supporting Get, Put, and Delete operations.", "module_hint": "KVStore"},
        {"id": "SYN009", "prompt": "A state machine for a vending machine that accepts coins and dispenses items when enough money is inserted.", "module_hint": "VendingMachine"},
        {"id": "SYN010", "prompt": "An elevator controller for a building with N floors. The elevator moves up and down, opening doors at requested floors.", "module_hint": "Elevator"},
        # Simple specs — high success probability
        {"id": "SYN011", "prompt": "A counter that starts at 0, can increment by 1, and has a maximum value MAX.", "module_hint": "Counter"},
        {"id": "SYN012", "prompt": "A toggle switch that alternates between ON and OFF states.", "module_hint": "Toggle"},
        {"id": "SYN013", "prompt": "A simple semaphore with initial count N. Processes can acquire (decrement) or release (increment) the semaphore. Count must be non-negative.", "module_hint": "Semaphore"},
        {"id": "SYN014", "prompt": "A register that stores a natural number. It can be read, written, or reset to zero.", "module_hint": "Register"},
        {"id": "SYN015", "prompt": "A two-bit binary counter that counts from 0 to 3 and wraps.", "module_hint": "BinaryCounter"},
        {"id": "SYN016", "prompt": "A simple buffer with one slot. A producer writes to it, a consumer reads from it. The buffer is either empty or full.", "module_hint": "SingleBuffer"},
        {"id": "SYN017", "prompt": "A state machine that models a door: it can be Open, Closed, or Locked. Transitions follow physical constraints.", "module_hint": "Door"},
        {"id": "SYN018", "prompt": "A resource allocator for a single resource. Processes can request, acquire, and release the resource.", "module_hint": "ResourceAlloc"},
        {"id": "SYN019", "prompt": "A min/max tracker that keeps track of the minimum and maximum values seen from a stream of naturals.", "module_hint": "MinMaxTracker"},
        {"id": "SYN020", "prompt": "A simple DAG node with status: Pending, Running, Done. It transitions from Pending to Running to Done.", "module_hint": "TaskNode"},
    ]
    prompts.extend(synthetic)

    if limit:
        prompts = prompts[:limit]

    return prompts


# ─────────────────────────────────────────────────────────────────────────────
# Core loop: one iteration
# ─────────────────────────────────────────────────────────────────────────────

def run_iteration(
    model: str = "deepseek-r1:8b",
    num_prompts: Optional[int] = None,
    temperature: float = 0.4,
) -> IterationStats:
    """
    Run one generate → validate → fix → augment iteration.

    Returns IterationStats with counts of what happened.
    """
    from src.inference.ollama_client import ChatTLAClient

    stats = IterationStats()
    client = ChatTLAClient(model=model, reasoning="medium")
    prompts = load_prompts(limit=num_prompts)
    new_examples: list[dict] = []

    log.info(f"[self_improve] Starting iteration with {len(prompts)} prompts, model={model}")

    for p in prompts:
        prompt_id = p["id"]
        prompt_text = p["prompt"]
        module_hint = p.get("module_hint")
        stats.prompts_tried += 1

        # ── Phase 1: Generate ─────────────────────────────────────────────
        log.info(f"  [{prompt_id}] Generating spec for: {prompt_text[:60]}...")
        try:
            spec = client.generate_spec(prompt_text, module_name=module_hint, temperature=temperature)
        except Exception as e:
            log.warning(f"  [{prompt_id}] Generation failed: {e}")
            continue

        stats.specs_generated += 1

        # ── Phase 2: SANY validate ────────────────────────────────────────
        is_valid, sany_errors = validate_with_sany(spec)

        if is_valid:
            # Spec passes SANY! Add as spec_generation example
            log.info(f"  [{prompt_id}] ✓ SANY pass (raw). Adding as spec_gen example.")
            stats.sany_pass_raw += 1
            stats.spec_gen_examples += 1
            new_examples.append(build_spec_gen_example(prompt_text, spec))
            continue

        # ── Phase 3: Python fixer ─────────────────────────────────────────
        log.info(f"  [{prompt_id}] ✗ SANY fail. Attempting Python fixes...")
        fix_result = fix_tla_syntax(spec, sany_errors)

        if fix_result.fixes_applied:
            log.info(f"  [{prompt_id}]   Applied {len(fix_result.fixes_applied)} fixes: {', '.join(fix_result.fixes_applied)}")

            # Re-validate after Python fixes
            is_valid_fixed, sany_errors_fixed = validate_with_sany(fix_result.fixed_spec)

            if is_valid_fixed:
                log.info(f"  [{prompt_id}] ✓ SANY pass after fixes. Adding spec_gen example.")
                stats.sany_pass_fixed += 1

                # Only create spec_gen example with the corrected spec.
                # Bug-fix examples are deliberately excluded: they contain
                # buggy TLA+ in the user message which the model memorises,
                # causing it to reproduce syntax errors during generation.
                new_examples.append(build_spec_gen_example(prompt_text, fix_result.fixed_spec))
                stats.spec_gen_examples += 1
                continue

        # ── Phase 4: Model self-correction attempt ────────────────────────
        log.info(f"  [{prompt_id}] Python fixes insufficient. Trying model self-correction...")
        try:
            corrected_spec, tier = client.validate_and_generate(prompt_text)
            is_valid_corrected, _ = validate_with_sany(corrected_spec)

            if is_valid_corrected:
                log.info(f"  [{prompt_id}] ✓ Self-correction succeeded (tier={tier}).")
                stats.sany_pass_fixed += 1

                # Only spec_gen — no bug_fix examples (see note above).
                new_examples.append(build_spec_gen_example(prompt_text, corrected_spec))
                stats.spec_gen_examples += 1
            else:
                log.info(f"  [{prompt_id}] ✗ Self-correction also failed. Skipping.")
        except Exception as e:
            log.warning(f"  [{prompt_id}] Self-correction error: {e}")

    # ── Phase 5: Persist new examples (with dedup) ────────────────────────
    stats.total_new_examples = len(new_examples)
    if new_examples:
        # Load existing augmented examples to dedup against
        existing_prompts = set()
        if _AUGMENTED_JSONL.exists():
            with open(_AUGMENTED_JSONL, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            ex = json.loads(line)
                            user_msg = [m for m in ex["messages"] if m["role"] == "user"][0]["content"]
                            # Hash first 200 chars of prompt as dedup key
                            existing_prompts.add(user_msg[:200])
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass

        # Filter out duplicates
        deduped_examples = []
        for ex in new_examples:
            user_msg = [m for m in ex["messages"] if m["role"] == "user"][0]["content"]
            key = user_msg[:200]
            if key not in existing_prompts:
                deduped_examples.append(ex)
                existing_prompts.add(key)

        stats.total_new_examples = len(deduped_examples)
        if deduped_examples:
            _AUGMENTED_JSONL.parent.mkdir(parents=True, exist_ok=True)
            with open(_AUGMENTED_JSONL, "a", encoding="utf-8") as f:
                for ex in deduped_examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            log.info(f"[self_improve] Appended {len(deduped_examples)} new examples "
                     f"({len(new_examples) - len(deduped_examples)} deduped) to {_AUGMENTED_JSONL}")
        else:
            log.info(f"[self_improve] All {len(new_examples)} examples were duplicates — nothing appended.")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Dataset rebuild
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_dataset() -> tuple[int, int]:
    """Rebuild train/eval JSONL from combined.jsonl + augmented.jsonl."""
    from src.training.dataset_builder import build

    n_train, n_eval = build(
        include_augmented=True,
        include_description_sft=True,
        bugfix_oversample=2,
        include_silver_augmented=True,
        augmented_best_per_prompt=True,
    )
    log.info(f"[self_improve] Dataset rebuilt: train={n_train}, eval={n_eval}")
    return n_train, n_eval


# ─────────────────────────────────────────────────────────────────────────────
# Retrain + deploy
# ─────────────────────────────────────────────────────────────────────────────

def retrain_and_deploy() -> bool:
    """
    Run full retrain → merge → GGUF → Ollama deploy pipeline.

    Dynamically scales epochs and timeout based on training set size.
    Returns True if successful.
    """
    log.info("[self_improve] Starting retraining...")

    # Count training examples to scale epochs and timeout
    n_train = 0
    if _TRAIN_JSONL.exists():
        with open(_TRAIN_JSONL) as f:
            n_train = sum(1 for line in f if line.strip())

    # Fewer epochs for larger datasets: 10 for <80, 5 for ~200, 3 for 400+
    num_epochs = max(3, min(10, 600 // max(n_train, 1)))
    # Estimate timeout: steps≈n_train*epochs/8, ~40s/step, 1.5x safety
    est_steps = (n_train * num_epochs) // 8
    timeout_s = max(3600, int(est_steps * 40 * 1.5))
    log.info(f"[self_improve] Training {n_train} examples × {num_epochs} epochs "
             f"(~{est_steps} steps, timeout={timeout_s}s)")

    # 1. Train
    result = subprocess.run(
        [sys.executable, "-m", "src.training.train", "--epochs", str(num_epochs)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        log.error(f"[self_improve] Training failed:\n{result.stderr[-500:]}")
        return False
    log.info("[self_improve] Training complete.")

    # 2. Merge LoRA
    result = subprocess.run(
        [sys.executable, "-m", "src.training.merge_lora"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        log.error(f"[self_improve] Merge failed:\n{result.stderr[-500:]}")
        return False
    log.info("[self_improve] LoRA merged.")

    # 3. Convert to GGUF + register Ollama
    result = subprocess.run(
        [sys.executable, "-m", "src.inference.convert_to_gguf", "--quant", "Q8_0"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        log.error(f"[self_improve] GGUF conversion failed:\n{result.stderr[-500:]}")
        return False
    log.info("[self_improve] GGUF deployed to Ollama.")

    if os.environ.get("HF_TOKEN"):
        log.info("[self_improve] Publishing to Hugging Face Hub...")
        pub = subprocess.run(
            [sys.executable, "-m", "src.training.publish_hf", "--quant", "Q8_0"],
            cwd=str(_REPO_ROOT),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if pub.returncode != 0:
            log.warning(f"[self_improve] HF publish failed (non-fatal): {(pub.stderr or pub.stdout)[-400:]}")
        else:
            log.info("[self_improve] HF publish complete.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main autonomous loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ChatTLA self-improvement loop: generate → validate → fix → retrain"
    )
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of generate/fix iterations to run (default: 5)")
    parser.add_argument("--prompts", type=int, default=None,
                        help="Limit number of prompts per iteration (default: all)")
    parser.add_argument("--retrain-threshold", type=int, default=15,
                        help="Retrain after accumulating this many new examples (default: 15)")
    parser.add_argument("--model", default="deepseek-r1:8b",
                        help="Ollama model tag to use for generation")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature for generation")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Skip retraining (generate/fix/augment only)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Count existing augmented examples
    existing_aug = 0
    if _AUGMENTED_JSONL.exists():
        with open(_AUGMENTED_JSONL) as f:
            existing_aug = sum(1 for line in f if line.strip())

    total_new = 0
    retrain_count = 0

    print("="*70)
    print("  ChatTLA Self-Improvement Loop")
    print(f"  Iterations: {args.iterations} | Prompts/iter: {args.prompts or 'all'}")
    print(f"  Retrain threshold: {args.retrain_threshold} | Model: {args.model}")
    print(f"  Existing augmented examples: {existing_aug}")
    print("="*70)

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'─'*70}")
        print(f"  Iteration {iteration}/{args.iterations}")
        print(f"{'─'*70}")

        t0 = time.time()
        stats = run_iteration(
            model=args.model,
            num_prompts=args.prompts,
            temperature=args.temperature,
        )
        elapsed = time.time() - t0

        total_new += stats.total_new_examples

        print(f"\n  Iteration {iteration} results ({elapsed:.0f}s):")
        print(f"    Prompts tried:       {stats.prompts_tried}")
        print(f"    Specs generated:     {stats.specs_generated}")
        print(f"    SANY pass (raw):     {stats.sany_pass_raw}")
        print(f"    SANY pass (fixed):   {stats.sany_pass_fixed}")
        print(f"    Bug-fix examples:    {stats.bug_fix_examples}")
        print(f"    Spec-gen examples:   {stats.spec_gen_examples}")
        print(f"    New examples total:  {stats.total_new_examples}")
        print(f"    Cumulative new:      {total_new}")

        # Check if we should retrain
        if not args.no_retrain and total_new >= args.retrain_threshold:
            print(f"\n  ⟳ Retrain threshold reached ({total_new} ≥ {args.retrain_threshold})")
            print(f"    Rebuilding dataset...")
            n_train, n_eval = rebuild_dataset()
            print(f"    Dataset: {n_train} train, {n_eval} eval")

            print(f"    Starting retrain + deploy pipeline...")
            t1 = time.time()
            success = retrain_and_deploy()
            retrain_elapsed = time.time() - t1

            if success:
                retrain_count += 1
                total_new = 0  # Reset counter
                print(f"    ✓ Retrain #{retrain_count} complete ({retrain_elapsed/60:.1f} min)")
            else:
                print(f"    ✗ Retrain failed after {retrain_elapsed/60:.1f} min. Continuing...")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  Self-Improvement Loop Complete")
    print(f"  Retrains completed: {retrain_count}")

    # Count total augmented examples
    aug_count = 0
    if _AUGMENTED_JSONL.exists():
        with open(_AUGMENTED_JSONL) as f:
            aug_count = sum(1 for line in f if line.strip())
    print(f"  Total augmented examples: {aug_count}")
    print(f"  Remaining un-trained examples: {total_new}")
    print(f"{'='*70}")

    # If there are untrained examples, offer to rebuild
    if total_new > 0 and not args.no_retrain:
        print(f"\n  {total_new} examples pending. Rebuilding dataset for next manual retrain...")
        rebuild_dataset()


if __name__ == "__main__":
    main()
