"""
augment.py — Synthetic training data augmentation for ChatTLA.

Tackles the core problem: with only 57 training examples (32 base records ×
~1.8 tasks each), the model doesn't see enough valid TLA+ to reliably
produce SANY-clean output.  This module generates additional training
examples through three strategies:

Strategy 1: Spec Variants
    Take a known-good spec → vary variable names, constant values, and
    number of processes to create semantically similar but syntactically
    distinct training examples.

Strategy 2: Error-Correction Pairs (bug_fix task)
    Inject realistic mutations into valid specs (the same errors the model
    makes) → pair with the original as the correction target.  This teaches
    the self-correction skill directly.

Strategy 3: Module Decomposition
    Split a large spec into sub-problems: just the Init/Next, just the
    invariants, just the type definition.  Each becomes a focused training
    example.

Usage
-----
    python -m src.training.augment                     # augment combined.jsonl in-place
    python -m src.training.augment --dry-run           # preview how many examples would be generated
    python -m src.training.augment --sany-validate     # only keep augmented specs that pass SANY
"""

from __future__ import annotations

import json
import logging
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional

from src.shared.schemas.dataset_schema import DatasetRecord, Annotation
from src.training.dataset_builder import (
    _DEVELOPER_PROMPT,
    build_messages_spec_generation,
)

log = logging.getLogger(__name__)

_REPO_ROOT       = Path(__file__).resolve().parents[2]
_COMBINED_JSONL  = _REPO_ROOT / "data" / "validated" / "combined.jsonl"
_AUGMENTED_OUT   = _REPO_ROOT / "data" / "processed" / "augmented.jsonl"


# ---------------------------------------------------------------------------
# Strategy 1: Spec Variants (rename identifiers, adjust constants)
# ---------------------------------------------------------------------------

_VAR_RENAMES = [
    {"pc": "program_counter", "state": "proc_state"},
    {"msg": "message", "msgs": "messages", "chan": "channel"},
    {"flag": "request", "turn": "priority"},
    {"q": "queue", "head": "front", "tail": "rear"},
    {"lock": "mutex", "owner": "holder"},
]

_CONST_VARIANTS = [
    {"N": ["3", "4", "5", "7"]},
    {"K": ["2", "4", "8", "16"]},
    {"Participants": ["Participants"]},  # no change for sets
]


def augment_variant(record: DatasetRecord, rng: random.Random) -> Optional[DatasetRecord]:
    """
    Create a variant of a spec by renaming variables and adjusting constants.
    Returns None if the spec is too short to meaningfully vary.
    """
    tla = record.tla_content
    if len(tla.splitlines()) < 15:
        return None

    # Pick a random rename mapping that has at least one matching variable
    rename_map = rng.choice(_VAR_RENAMES)
    applied = False
    new_tla = tla
    for old_name, new_name in rename_map.items():
        if re.search(rf"\b{re.escape(old_name)}\b", new_tla):
            # Rename in variable declarations, definitions, and uses
            new_tla = re.sub(rf"\b{re.escape(old_name)}\b", new_name, new_tla)
            applied = True

    # Also vary a constant value if possible
    for const_map in _CONST_VARIANTS:
        for const_name, values in const_map.items():
            pattern = rf"({re.escape(const_name)}\s*==\s*)(\d+)"
            m = re.search(pattern, new_tla)
            if m:
                new_val = rng.choice(values)
                new_tla = re.sub(pattern, rf"\g<1>{new_val}", new_tla)
                applied = True

    if not applied:
        return None

    variant = deepcopy(record)
    variant.tla_content = new_tla
    variant.id = DatasetRecord.make_id(new_tla)
    variant.source = f"synthetic:variant:{record.source}"
    return variant


# ---------------------------------------------------------------------------
# Strategy 2: Error-Correction Pairs (bug_fix training data)
# ---------------------------------------------------------------------------

_MUTATIONS = [
    # Missing closing delimiter
    ("remove_closing", lambda tla: tla.replace("====", "")),
    # Double-prime a variable
    ("double_prime", lambda tla: re.sub(r"(\w)'(\s)", r"\1''\2", tla, count=1)),
    # Remove EXTENDS line
    ("remove_extends", lambda tla: re.sub(r"^EXTENDS\s+.*$", "", tla, count=1, flags=re.MULTILINE)),
    # Change VARIABLES to VAR (invalid keyword)
    ("bad_keyword", lambda tla: tla.replace("VARIABLES", "VAR", 1) if "VARIABLES" in tla else tla),
    # Add PlusCal-style 'begin' keyword
    ("pluscal_leak", lambda tla: tla.replace("Init ==", "begin\nInit ==", 1)),
    # Break an UNCHANGED tuple
    ("bad_unchanged", lambda tla: re.sub(r"UNCHANGED\s*<<([^>]+)>>", r"UNCHANGED \1", tla, count=1)),
    # Add equals in CONSTANT declaration
    ("const_equals", lambda tla: re.sub(r"^(CONSTANTS?\s+\w+)\s*$", r"\1 = 5", tla, count=1, flags=re.MULTILINE)),
]


def augment_bug_fix(
    record: DatasetRecord,
    rng: random.Random,
    n_mutations: int = 2,
) -> list[dict]:
    """
    Generate bug_fix training examples by injecting mutations into a valid spec.

    Returns a list of message dicts (ready for JSONL) — one per mutation.
    """
    examples = []
    tla = record.tla_content.strip()
    if len(tla.splitlines()) < 10:
        return examples

    # Pick n_mutations random mutations
    chosen = rng.sample(_MUTATIONS, min(n_mutations, len(_MUTATIONS)))

    for name, mutator in chosen:
        buggy = mutator(tla)
        if buggy.strip() == tla.strip():
            continue  # mutation had no effect

        # Build the error description (simulated)
        error_msg = f"SANY parse error after applying '{name}' mutation."

        messages = [
            {"role": "developer", "content": _DEVELOPER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"This TLA+ spec has a syntax error:\n\n"
                    f"Error: {error_msg}\n\n"
                    f"Buggy spec:\n{buggy}\n\n"
                    f"Fix the spec and output only the corrected TLA+ module."
                ),
            },
            {"role": "assistant", "channel": "analysis", "content": "Correcting TLA+ syntax errors and module structure."},
            {"role": "assistant", "channel": "final", "content": tla},
        ]
        examples.append({"messages": messages})

    return examples


# ---------------------------------------------------------------------------
# Strategy 3: Module Decomposition (focused sub-tasks)
# ---------------------------------------------------------------------------

def augment_decomposition(record: DatasetRecord) -> list[dict]:
    """
    Split a spec into focused sub-tasks: Init/Next only, invariants only, etc.
    Returns a list of message dicts.
    """
    examples: list[dict] = []
    tla = record.tla_content.strip()

    if not record.annotation or not record.annotation.natural_language_description:
        return examples

    nl = record.annotation.natural_language_description.strip()

    # Sub-task: Write just the TypeOK invariant
    typeok_match = re.search(r"^(TypeOK\s*==.*?)(?=^\w+\s*==|\Z)", tla, re.MULTILINE | re.DOTALL)
    if typeok_match:
        examples.append({"messages": [
            {"role": "developer", "content": _DEVELOPER_PROMPT},
            {"role": "user", "content": f"Write a TypeOK invariant for this system:\n\n{nl}"},
            {"role": "assistant", "channel": "analysis", "content": "I'll define the type-correctness invariant covering all state variables."},
            {"role": "assistant", "channel": "final", "content": typeok_match.group(1).strip()},
        ]})

    # Sub-task: Write Init and Next given VARIABLES
    vars_match = re.search(r"^VARIABLES\s+(.+?)$", tla, re.MULTILINE)
    init_match = re.search(r"^(Init\s*==.*?)(?=^Next\s*==|\Z)", tla, re.MULTILINE | re.DOTALL)
    next_match = re.search(r"^(Next\s*==.*?)(?=^Spec\s*==|^TypeOK\s*==|^\w+\s*==|\Z)", tla, re.MULTILINE | re.DOTALL)
    if vars_match and init_match and next_match:
        init_next = init_match.group(1).strip() + "\n\n" + next_match.group(1).strip()
        examples.append({"messages": [
            {"role": "developer", "content": _DEVELOPER_PROMPT},
            {"role": "user", "content": f"Given VARIABLES {vars_match.group(1).strip()}, write Init and Next for:\n\n{nl}"},
            {"role": "assistant", "channel": "analysis", "content": "I'll define the initial state and transition relation."},
            {"role": "assistant", "channel": "final", "content": init_next},
        ]})

    return examples


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def augment(
    combined_path: Path = _COMBINED_JSONL,
    output_path: Path = _AUGMENTED_OUT,
    sany_validate: bool = False,
    dry_run: bool = False,
    seed: int = 42,
) -> int:
    """
    Generate augmented training data from combined.jsonl.

    Parameters
    ----------
    sany_validate : bool   If True, run SANY on generated variants and discard failures.
    dry_run       : bool   If True, count but don't write.

    Returns the total number of augmented examples generated.
    """
    rng = random.Random(seed)

    records: list[DatasetRecord] = []
    for line in combined_path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            records.append(DatasetRecord.from_dict(json.loads(line)))

    print(f"[augment] Loaded {len(records)} base records")

    all_examples: list[dict] = []

    for r in records:
        # Strategy 1: Variant
        variant = augment_variant(r, rng)
        if variant:
            msgs = build_messages_spec_generation(variant)
            if msgs:
                all_examples.append({"messages": msgs})

        # Strategy 2: Bug-fix pairs
        bug_fixes = augment_bug_fix(r, rng, n_mutations=2)
        all_examples.extend(bug_fixes)

        # Strategy 3: Decomposition
        decomps = augment_decomposition(r)
        all_examples.extend(decomps)

    print(f"[augment] Generated {len(all_examples)} augmented examples")

    if sany_validate and not dry_run:
        from src.validators.sany_validator import validate_string as sany_check

        valid_examples: list[dict] = []
        for ex in all_examples:
            # Only validate examples that contain full specs (variant / bug_fix targets)
            final_content = ""
            for msg in ex["messages"]:
                if msg.get("channel") == "final":
                    final_content = msg["content"]
                    break
            if "---- MODULE" in final_content and "====" in final_content:
                m = re.search(r"----\s*MODULE\s+(\w+)", final_content)
                module_name = m.group(1) if m else "Spec"
                try:
                    result = sany_check(final_content, module_name=module_name)
                    if result.valid:
                        valid_examples.append(ex)
                except Exception:
                    pass
            else:
                # Sub-task examples (decomposition) — keep as-is
                valid_examples.append(ex)

        print(f"[augment] After SANY validation: {len(valid_examples)} / {len(all_examples)} kept")
        all_examples = valid_examples

    if dry_run:
        print(f"[augment] DRY RUN — would write {len(all_examples)} examples to {output_path}")
        return len(all_examples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[augment] Wrote {len(all_examples)} examples → {output_path}")
    return len(all_examples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment ChatTLA training data")
    parser.add_argument("--dry-run", action="store_true", help="Count examples without writing")
    parser.add_argument("--sany-validate", action="store_true",
                        help="Discard augmented specs that fail SANY validation")
    parser.add_argument("--combined", default=str(_COMBINED_JSONL))
    parser.add_argument("--output", default=str(_AUGMENTED_OUT))
    args = parser.parse_args()

    augment(
        combined_path=Path(args.combined),
        output_path=Path(args.output),
        sany_validate=args.sany_validate,
        dry_run=args.dry_run,
    )
