#!/usr/bin/env python3
"""
build_multitask_sft.py — Expand diamond_sft.jsonl into multi-task training data.

Inspired by fm-alpaca (Cao et al. 2024), which decomposes formal verification
into 6 sub-tasks and shows that 7-8B models trained on this multi-task mix
match 70B+ base models on full-proof generation.

For each Diamond-tier spec, we mint 5 training examples across these tasks:

  1. full_gen      — NL description → full TLA+ spec (the original task)
  2. completion    — NL + spec prefix → continuation to ====
  3. infill_next   — NL + spec with Next masked → just the Next body
  4. segment_init  — NL + VARIABLES + TypeOK → just the Init definition
  5. req_analysis  — NL → markdown checklist of required invariants/properties

The intuition: instead of asking the model to do one hard 8% task, ask it
to do 5 easier tasks that compose to the same end result. Each task gets its
own SFT signal so the model learns to handle each piece independently.

Output: data/processed/multitask_sft.jsonl  (5x the input size)

Usage:
    python scripts/build_multitask_sft.py
    python scripts/build_multitask_sft.py --tasks full_gen,completion,segment_init
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.training.dataset_builder import _DEVELOPER_PROMPT  # noqa: E402

_DIAMOND_JSONL = _REPO / "data" / "processed" / "diamond_sft.jsonl"
_OUTPUT_JSONL = _REPO / "data" / "processed" / "multitask_sft.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Spec parsing — pull out individual pieces from a full TLA+ module
# ─────────────────────────────────────────────────────────────────────────────

def _extract_module_name(spec: str) -> str:
    m = re.search(r"----\s*MODULE\s+(\w+)", spec)
    return m.group(1) if m else "Spec"


def _extract_section(spec: str, name: str) -> Optional[str]:
    """Extract a top-level definition like 'name == ...' including its body.

    Strips trailing comment-only lines so following operators' comments
    don't leak into this section.
    """
    lines = spec.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if re.match(rf"^\s*{re.escape(name)}\s*==", line):
            start_idx = i
            break
    if start_idx is None:
        return None

    out_lines = [lines[start_idx]]
    for line in lines[start_idx + 1:]:
        if re.match(r"^[A-Za-z_]\w*\s*==", line):
            break
        if re.match(r"^={4,}", line):
            break
        out_lines.append(line)

    # Strip trailing blank lines AND trailing comment-only lines
    # (those usually belong to the NEXT operator, not this one)
    def _is_comment_or_blank(s: str) -> bool:
        s = s.strip()
        return not s or s.startswith("\\*") or s.startswith("(*")

    while out_lines and _is_comment_or_blank(out_lines[-1]):
        out_lines.pop()
    return "\n".join(out_lines) if out_lines else None


def _extract_variables_decl(spec: str) -> Optional[str]:
    """Extract the VARIABLES declaration line."""
    m = re.search(r"^\s*VARIABLES?\s+.+$", spec, re.MULTILINE)
    return m.group(0).strip() if m else None


def _extract_constants_decl(spec: str) -> Optional[str]:
    """Extract CONSTANT(S) declaration(s)."""
    matches = re.findall(r"^\s*CONSTANTS?\s+.+$", spec, re.MULTILINE)
    return "\n".join(m.strip() for m in matches) if matches else None


def _extract_extends(spec: str) -> Optional[str]:
    m = re.search(r"^\s*EXTENDS\s+.+$", spec, re.MULTILINE)
    return m.group(0).strip() if m else None


def _extract_invariants(spec: str) -> list[str]:
    """Find candidate invariant operator names defined in the spec."""
    invariants = []
    for m in re.finditer(r"^\s*(\w+)\s*==", spec, re.MULTILINE):
        name = m.group(1)
        if name in ("Init", "Next", "Spec", "vars"):
            continue
        if name == "TypeOK" or re.match(
            r"(Safety|.*Inv(ariant)?$|Mutex|MutualExclusion|Conservation|"
            r"Bounded|NoOverflow|NoUnderflow|AtMost|NoWrite|Valid|"
            r".*Conserved|.*Bounded|.*Stable|.*Threshold)",
            name,
        ):
            invariants.append(name)
    return invariants


def _split_user_prompt(user_content: str) -> str:
    """Strip the 'Write a TLA+ specification for the following:' preamble.

    Some Diamond specs have it doubled (we wrapped twice in earlier pipelines)
    so we strip both layers if present.
    """
    text = user_content.strip()
    prefix = "Write a TLA+ specification for the following:"
    while text.startswith(prefix):
        text = text[len(prefix):].lstrip("\n").strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Task minters — each function returns a multitask SFT example or None
# ─────────────────────────────────────────────────────────────────────────────

def mint_full_gen(record: dict, nl: str, spec: str) -> Optional[dict]:
    """Task 1: NL description → full TLA+ spec.

    This is the original task — keep the diamond spec as the gold answer.
    """
    return {
        "_task": "full_gen",
        "_prompt_id": record.get("_prompt_id", ""),
        "_tier": "diamond",
        "messages": [
            {"role": "developer", "content": _DEVELOPER_PROMPT},
            {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{nl}"},
            {"role": "assistant", "channel": "analysis", "content":
                "I'll write a verified TLA+ specification with meaningful invariants that constrain behavior."},
            {"role": "assistant", "channel": "final", "content": spec.strip()},
        ],
    }


def mint_completion(record: dict, nl: str, spec: str) -> Optional[dict]:
    """Task 2: NL + spec prefix (header through Init) → continuation (Next + invariants + Spec).

    Tests whether the model can complete a partial spec given the structure
    is already established.
    """
    init_section = _extract_section(spec, "Init")
    if not init_section:
        return None

    # Find where Init ends in the original spec; everything before is the prefix
    init_pos = spec.find(init_section)
    if init_pos < 0:
        return None
    prefix_end = init_pos + len(init_section)
    prefix = spec[:prefix_end]
    continuation = spec[prefix_end:].strip()
    if not continuation or "Next" not in continuation:
        return None

    return {
        "_task": "completion",
        "_prompt_id": record.get("_prompt_id", ""),
        "_tier": "diamond",
        "messages": [
            {"role": "developer", "content": _DEVELOPER_PROMPT},
            {"role": "user", "content":
                f"Complete this TLA+ specification by adding Next, any invariants, "
                f"and Spec. The system being modeled:\n\n{nl}\n\n"
                f"Partial spec:\n```\n{prefix}\n```"},
            {"role": "assistant", "channel": "analysis", "content":
                "I'll complete the spec by adding the Next transition relation, invariants, and Spec."},
            {"role": "assistant", "channel": "final", "content": continuation},
        ],
    }


def mint_infill_next(record: dict, nl: str, spec: str) -> Optional[dict]:
    """Task 3: NL + spec with Next masked → fill in Next body.

    Tests targeted ability to write the Next transition relation given context.
    """
    next_section = _extract_section(spec, "Next")
    if not next_section:
        return None

    masked_spec = spec.replace(next_section, "Next == \\* TODO: fill in transition relation", 1)

    return {
        "_task": "infill_next",
        "_prompt_id": record.get("_prompt_id", ""),
        "_tier": "diamond",
        "messages": [
            {"role": "developer", "content": _DEVELOPER_PROMPT},
            {"role": "user", "content":
                f"Fill in the Next transition relation for this TLA+ spec. The system:\n\n{nl}\n\n"
                f"Spec with Next missing:\n```\n{masked_spec}\n```\n\n"
                f"Output ONLY the Next definition (and any helper action definitions it needs)."},
            {"role": "assistant", "channel": "analysis", "content":
                "I'll write the Next transition relation that captures all valid state transitions."},
            {"role": "assistant", "channel": "final", "content": next_section},
        ],
    }


def mint_segment_init(record: dict, nl: str, spec: str) -> Optional[dict]:
    """Task 4: NL + VARIABLES + TypeOK → just the Init definition.

    Smallest possible task: given the type signature, produce the initial state.
    """
    variables_decl = _extract_variables_decl(spec)
    typeok = _extract_section(spec, "TypeOK")
    init_section = _extract_section(spec, "Init")
    if not (variables_decl and typeok and init_section):
        return None

    constants = _extract_constants_decl(spec)
    constants_block = f"{constants}\n" if constants else ""

    return {
        "_task": "segment_init",
        "_prompt_id": record.get("_prompt_id", ""),
        "_tier": "diamond",
        "messages": [
            {"role": "developer", "content": _DEVELOPER_PROMPT},
            {"role": "user", "content":
                f"Write the Init predicate for this TLA+ spec. The system:\n\n{nl}\n\n"
                f"Declarations:\n```\n{constants_block}{variables_decl}\n\n{typeok}\n```\n\n"
                f"Output ONLY the Init definition. It must satisfy TypeOK."},
            {"role": "assistant", "channel": "analysis", "content":
                "I'll define Init by assigning concrete starting values to every variable."},
            {"role": "assistant", "channel": "final", "content": init_section},
        ],
    }


def mint_req_analysis(record: dict, nl: str, spec: str) -> Optional[dict]:
    """Task 5: NL → markdown checklist of required invariants and properties.

    No formal output. Pure CoT — teaches the model to reason about
    requirements before writing TLA+. Uses the actual invariants from the
    spec as the gold answer.
    """
    invariants = _extract_invariants(spec)
    if not invariants:
        return None

    # Build a markdown checklist from the invariant operator names
    # Convert PascalCase / snake_case to readable text. Common acronym
    # words (OK, ID, FIFO) stay together.
    _ACRONYMS = {"OK", "ID", "FIFO", "LIFO", "ACK", "NACK", "RPC"}

    def humanize(name: str) -> str:
        # Split on capital letter boundaries, but keep adjacent caps together
        parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]*|[a-z]+", name)
        # Glue acronyms back if they got split across two parts
        out = []
        for p in parts:
            if out and (out[-1] + p) in _ACRONYMS:
                out[-1] = out[-1] + p
            else:
                out.append(p)
        return " ".join(out) if out else name

    checklist_lines = ["**Required properties for this system:**", ""]
    for inv in invariants:
        checklist_lines.append(f"- `{inv}` — {humanize(inv)}")
    checklist = "\n".join(checklist_lines)

    return {
        "_task": "req_analysis",
        "_prompt_id": record.get("_prompt_id", ""),
        "_tier": "diamond",
        "messages": [
            {"role": "developer", "content":
                "You are a TLA+ requirements analyst. Given a natural-language description of a system, "
                "list the safety properties and invariants that should hold. Output a markdown checklist."},
            {"role": "user", "content":
                f"Analyze this system and list the safety properties / invariants it should satisfy:\n\n{nl}"},
            {"role": "assistant", "channel": "analysis", "content":
                "I'll identify the key safety properties this system must maintain."},
            {"role": "assistant", "channel": "final", "content": checklist},
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

_TASK_FUNCS = {
    "full_gen": mint_full_gen,
    "completion": mint_completion,
    "infill_next": mint_infill_next,
    "segment_init": mint_segment_init,
    "req_analysis": mint_req_analysis,
}


def expand_to_multitask(input_path: Path, output_path: Path, tasks: list[str]) -> dict:
    """Read diamond_sft.jsonl, mint multi-task examples, write to output."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    stats = {task: 0 for task in tasks}
    stats["total_input"] = 0
    stats["total_output"] = 0
    stats["skipped"] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_keys = set()

    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["total_input"] += 1

            messages = record.get("messages", [])
            user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
            final_msg = next(
                (m["content"] for m in messages
                 if m.get("role") == "assistant" and m.get("channel") == "final"),
                "",
            )
            if not user_msg or not final_msg:
                stats["skipped"] += 1
                continue

            nl = _split_user_prompt(user_msg)
            spec = final_msg.strip()

            for task in tasks:
                func = _TASK_FUNCS[task]
                example = func(record, nl, spec)
                if example is None:
                    continue
                # Dedupe by (task, prompt_id, final_content_hash) to avoid
                # writing identical examples for the same input record.
                key = (task, example.get("_prompt_id", ""),
                       hash(example["messages"][-1]["content"]))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                fout.write(json.dumps(example, ensure_ascii=False) + "\n")
                stats[task] += 1
                stats["total_output"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(_DIAMOND_JSONL),
                        help="Input diamond_sft.jsonl path")
    parser.add_argument("--output", default=str(_OUTPUT_JSONL),
                        help="Output multitask_sft.jsonl path")
    parser.add_argument("--tasks", default="all",
                        help="Comma-separated task names (or 'all'). "
                             f"Available: {','.join(_TASK_FUNCS.keys())}")
    args = parser.parse_args()

    if args.tasks == "all":
        tasks = list(_TASK_FUNCS.keys())
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]
        for t in tasks:
            if t not in _TASK_FUNCS:
                parser.error(f"Unknown task: {t}. Available: {','.join(_TASK_FUNCS.keys())}")

    stats = expand_to_multitask(Path(args.input), Path(args.output), tasks)

    print(f"\n[multitask_sft] Input:  {args.input}")
    print(f"[multitask_sft] Output: {args.output}")
    print(f"[multitask_sft] Tasks: {', '.join(tasks)}")
    print()
    print(f"  Diamond records read: {stats['total_input']}")
    print(f"  Skipped (malformed):  {stats['skipped']}")
    print(f"  Total examples minted: {stats['total_output']}")
    print()
    print("  Per-task counts:")
    for task in tasks:
        print(f"    {task:15s} {stats[task]:5d}")
    print()
    print(f"  Expansion ratio: {stats['total_output'] / max(stats['total_input'], 1):.2f}x")


if __name__ == "__main__":
    main()
