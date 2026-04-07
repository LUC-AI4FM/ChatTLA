#!/usr/bin/env python3
"""Emit staged (progressive) SFT examples from a curated diamond dataset.

The FormaLLM evaluation (docs/formallm.md) found that **progressive prompting
is the only strategy that produced TLC passes (8.6%)**. Single-pass generation
fails because the model has to satisfy module syntax, type consistency,
behavioral semantics, and temporal logic simultaneously. Decomposing the task
into sequential turns lets each turn focus on one concern.

This script takes a flat curated dataset (one full spec per row) and explodes
each row into N staged training examples — one per generation stage. Each
stage is a short, focused supervised target conditioned on the cumulative
prefix of the spec. After training on these, the inference path can either
emit the full spec (single-pass) or be queried stage-by-stage with validation
between stages.

Stages (default):
  1. header   → ---- MODULE ... ----  +  EXTENDS line
  2. params   → CONSTANTS + VARIABLES
  3. init     → Init operator
  4. next     → Next operator (and any helper actions)
  5. close    → Spec, TypeOK, invariants, ==== terminator

Usage:
  python -m scripts.emit_staged_sft \\
      --in  data/processed/diamond_curated.jsonl \\
      --out data/processed/diamond_curated_staged.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

# Make `python scripts/emit_staged_sft.py` work as well as `python -m`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.postprocess import normalize_spec  # noqa: E402


_DEVELOPER_PROMPT_STAGED = """\
You are ChatTLA, an expert at writing verified TLA+ formal specifications.
You generate specs **incrementally**, one section at a time, so each section
can be validated by SANY/TLC before continuing. Output only the section the
user asks for. Do not repeat earlier sections. Do not emit markdown fences,
prose, or <think> blocks.
Reasoning: medium\
"""


_HEADER_RE  = re.compile(r"(-{2,}\s*MODULE\s+\w+\s*-{2,})", re.MULTILINE)
_EXTENDS_RE = re.compile(r"^EXTENDS\b[^\n]*", re.MULTILINE)
_CONST_RE   = re.compile(r"^CONSTANTS?\b[^\n]*(?:\n[^\n]*)*?(?=\n\s*$|\n[A-Z])", re.MULTILINE)
_VARS_RE    = re.compile(r"^VARIABLES?\b[^\n]*", re.MULTILINE)
_TERM_RE    = re.compile(r"={3,}\s*$", re.MULTILINE)


def _operator_block(spec: str, name: str) -> str | None:
    """Find an operator definition `Name == ...` and return the block including
    indented continuation lines."""
    pat = re.compile(rf"^{name}\s*==.*?(?=^\S|\Z)", re.MULTILINE | re.DOTALL)
    m = pat.search(spec)
    if not m:
        return None
    return m.group(0).rstrip()


def _split_stages(spec: str) -> dict[str, str] | None:
    """Carve a spec into 5 stages. Returns None if the spec is too small or
    too irregular to stage cleanly."""
    spec = spec.strip()

    header_m = _HEADER_RE.search(spec)
    if not header_m:
        return None
    header_end = header_m.end()

    extends_m = _EXTENDS_RE.search(spec, header_end)
    header_block = spec[: extends_m.end() if extends_m else header_end].strip()

    const_m = _CONST_RE.search(spec)
    vars_m = _VARS_RE.search(spec)
    if not vars_m:
        return None
    params_pieces = []
    if const_m:
        params_pieces.append(const_m.group(0).strip())
    params_pieces.append(vars_m.group(0).strip())
    params_block = "\n".join(params_pieces)

    init_block = _operator_block(spec, "Init")
    next_block = _operator_block(spec, "Next")
    if not init_block or not next_block:
        return None

    spec_block = _operator_block(spec, "Spec") or ""
    type_block = _operator_block(spec, "TypeOK") or ""
    inv_blocks = []
    for inv_name in ("Invariant", "Safety", "MutualExclusion", "NoOverflow"):
        b = _operator_block(spec, inv_name)
        if b:
            inv_blocks.append(b)

    terminator = "=" * 78
    close_pieces = [b for b in (type_block, spec_block, *inv_blocks) if b]
    close_pieces.append(terminator)
    close_block = "\n\n".join(close_pieces)

    return {
        "header": header_block,
        "params": params_block,
        "init":   init_block,
        "next":   next_block,
        "close":  close_block,
    }


_USER_INSTR = {
    "header": "Begin a new TLA+ module. Emit ONLY the `---- MODULE Name ----` header line and the `EXTENDS` clause. Nothing else.",
    "params": "Continuing the module shown below, emit ONLY the `CONSTANTS` (if any) and `VARIABLES` declarations.",
    "init":   "Continuing the module shown below, emit ONLY the `Init` operator definition.",
    "next":   "Continuing the module shown below, emit ONLY the `Next` operator (and any helper sub-actions it directly references).",
    "close":  "Continuing the module shown below, emit ONLY the closing section: `TypeOK`, `Spec == Init /\\ [][Next]_vars`, any safety invariants, and the `====` terminator.",
}


_STAGE_ORDER = ("header", "params", "init", "next", "close")


def _build_user_prompt(stage: str, nl: str, prefix: str) -> str:
    instr = _USER_INSTR[stage]
    if stage == "header":
        return f"Specification description:\n{nl}\n\n{instr}"
    return (
        f"Specification description:\n{nl}\n\n"
        f"Module so far:\n{prefix}\n\n{instr}"
    )


def _user_text_from_record(record: dict) -> str | None:
    """Pull the original NL description from the curated record."""
    for m in record.get("messages", []):
        if m.get("role") == "user":
            content = m.get("content", "")
            # Strip the boilerplate prefix the dataset_builder added.
            content = re.sub(r"^Write a TLA\+ specification for the following:\s*", "", content)
            content = re.sub(r"^Write a TLA\+ specification for the following:\s*", "", content)
            return content.strip()
    return None


def _final_spec_from_record(record: dict) -> str | None:
    for m in record.get("messages", []):
        if m.get("role") == "assistant" and m.get("channel") == "final":
            return m.get("content", "")
    return None


def stage_record(record: dict) -> Iterator[dict]:
    """Yield 0..5 staged messages records from one curated record."""
    nl = _user_text_from_record(record)
    final = _final_spec_from_record(record)
    if not nl or not final:
        return
    cleaned, _ = normalize_spec(final)
    stages = _split_stages(cleaned)
    if not stages:
        return

    prefix_parts: list[str] = []
    for stage in _STAGE_ORDER:
        target = stages[stage].strip()
        if not target:
            prefix_parts.append(target)
            continue
        prefix_so_far = "\n\n".join(p for p in prefix_parts if p).strip()
        user_prompt = _build_user_prompt(stage, nl, prefix_so_far)
        yield {
            "_tier": "diamond_staged",
            "_prompt_id": f"{record.get('_prompt_id','?')}::{stage}",
            "_source": "emit_staged_sft",
            "_stage": stage,
            "_parent_prompt_id": record.get("_prompt_id"),
            "messages": [
                {"role": "developer", "content": _DEVELOPER_PROMPT_STAGED},
                {"role": "user",      "content": user_prompt},
                {"role": "assistant", "channel": "analysis", "content": f"I will emit only the {stage} section."},
                {"role": "assistant", "channel": "final",    "content": target},
            ],
        }
        prefix_parts.append(target)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in",  dest="inp", default="data/processed/diamond_curated.jsonl")
    p.add_argument("--out", dest="out", default="data/processed/diamond_curated_staged.jsonl")
    args = p.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_staged_records = 0
    with inp.open() as fh, out.open("w") as fo:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yielded = 0
            for staged in stage_record(rec):
                fo.write(json.dumps(staged) + "\n")
                yielded += 1
            if yielded:
                n_staged_records += 1
                n_out += yielded

    print(f"[emit_staged_sft] {n_in} input rows -> {n_staged_records} staged "
          f"({n_out} stage examples) -> {out}")


if __name__ == "__main__":
    main()
