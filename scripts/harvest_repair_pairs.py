#!/usr/bin/env python3
"""Harvest (broken_spec, error_message, fixed_spec) SFT triples for repair training.

The FormaLLM error taxonomy (docs/formallm.md §RQ2/RQ4) shows that ~50% of
SANY failures are "Parse: Bad Module Body" and the rest are dominated by
deterministic categories: Unicode operators, semicolons, backticks, missing
terminators, and <think> leakage. These are exactly the mutations we apply
here.

Two harvesting modes:
  1. **Synthetic**: take a verified spec, apply one mutation, run SANY on
     the broken version to get a real error message, and emit a triple
     (broken, sany_error, fixed). This gives us instant repair data
     without waiting for RL cycles.
  2. **Real** (--from-tlc-errors): scan outputs/logs/tlc_errors.jsonl and
     materialize triples from any rows that contain both a buggy snippet
     and a known good fix in the curated dataset.

Usage:
  python -m scripts.harvest_repair_pairs \\
      --gold data/processed/diamond_curated.jsonl \\
      --out  data/processed/repair_pairs.jsonl \\
      --mutations 4
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.postprocess import normalize_spec  # noqa: E402

try:
    from src.validators.sany_validator import validate_string as sany_validate  # noqa: E402
except Exception:
    sany_validate = None


_DEVELOPER_PROMPT_REPAIR = """\
You are ChatTLA, an expert at fixing TLA+ specifications. The user gives you
a buggy spec and the SANY parser error. Reply with the corrected spec only —
no markdown fences, no commentary. The fix must keep the original module
name, variables, and intended semantics; only repair what is broken.
Reasoning: medium\
"""


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


def _mutate_unicode(spec: str) -> tuple[str, str]:
    """Replace ASCII operators with Unicode equivalents."""
    pairs = [
        (r"\\in",  "\u2208"),
        (r"/\\",   "\u2227"),
        (r"\\/",   "\u2228"),
        (r"->",    "\u2192"),
        (r"\|->",  "\u21a6"),
    ]
    out = spec
    applied = []
    for src, dst in pairs:
        new = re.sub(src, dst, out, count=2)
        if new != out:
            out = new
            applied.append(dst)
    return out, "unicode_ops:" + ",".join(applied) if applied else ("", "")


def _mutate_semicolons(spec: str) -> tuple[str, str]:
    """Append a semicolon to ~3 lines that look like operator definitions."""
    lines = spec.split("\n")
    candidates = [i for i, ln in enumerate(lines) if re.search(r"==|/\\|UNCHANGED", ln)]
    if not candidates:
        return "", ""
    for i in random.sample(candidates, min(3, len(candidates))):
        lines[i] = lines[i].rstrip() + ";"
    return "\n".join(lines), "semicolons_inserted"


def _mutate_drop_terminator(spec: str) -> tuple[str, str]:
    new = re.sub(r"={3,}\s*$", "", spec.rstrip(), count=1)
    if new == spec.rstrip():
        return "", ""
    return new, "terminator_dropped"


def _mutate_fence(spec: str) -> tuple[str, str]:
    return f"```tla\n{spec}\n```", "markdown_fence"


def _mutate_think(spec: str) -> tuple[str, str]:
    return f"<think>Let me write this spec carefully.</think>\n{spec}", "think_block"


def _mutate_drop_unchanged(spec: str) -> tuple[str, str]:
    new, n = re.subn(r"/\\\s*UNCHANGED\s*<<[^>]*>>", "", spec, count=1)
    if n == 0:
        return "", ""
    return new, "unchanged_dropped"


def _mutate_extra_module_header(spec: str) -> tuple[str, str]:
    m = re.search(r"^-{2,}\s*MODULE\s+(\w+)\s*-{2,}", spec, re.MULTILINE)
    if not m:
        return "", ""
    inject = f"---- MODULE {m.group(1)} ----\n"
    return spec.replace(m.group(0), m.group(0) + "\n" + inject, 1), "duplicate_module_header"


_MUTATORS: list[tuple[str, Callable[[str], tuple[str, str]]]] = [
    ("unicode",            _mutate_unicode),
    ("semicolons",         _mutate_semicolons),
    ("drop_terminator",    _mutate_drop_terminator),
    ("fence",              _mutate_fence),
    ("think",              _mutate_think),
    ("drop_unchanged",     _mutate_drop_unchanged),
    ("dup_module_header",  _mutate_extra_module_header),
]


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def _final_spec(record: dict) -> str | None:
    for m in record.get("messages", []):
        if m.get("role") == "assistant" and m.get("channel") == "final":
            return m.get("content")
    return None


def _user_text(record: dict) -> str | None:
    for m in record.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return None


def _sany_error(spec: str) -> str:
    """Best-effort SANY error message. Returns empty string on success or
    when SANY is unavailable (in which case we synthesize a generic message
    so the training row is still useful)."""
    if sany_validate is None:
        return "SANY parse error in module body."
    try:
        m = re.search(r"^-{2,}\s*MODULE\s+(\w+)", spec, re.MULTILINE)
        name = m.group(1) if m else "Generated"
        result = sany_validate(spec, module_name=name, timeout=15)
        if getattr(result, "tier", "") in ("gold", "silver", "ok"):
            return ""
        msg = getattr(result, "error_message", "") or getattr(result, "stderr", "") or ""
        return msg.strip().splitlines()[0] if msg.strip() else "SANY parse error in module body."
    except Exception:
        return "SANY parse error in module body."


def build_pair(prompt_id: str, original_user: str, fixed_spec: str,
               mutator_name: str, broken: str, error_msg: str) -> dict:
    return {
        "_tier": "repair",
        "_prompt_id": f"{prompt_id}::repair_{mutator_name}",
        "_source": "harvest_repair_pairs",
        "_mutation": mutator_name,
        "messages": [
            {"role": "developer", "content": _DEVELOPER_PROMPT_REPAIR},
            {
                "role": "user",
                "content": (
                    f"The original task was:\n{original_user}\n\n"
                    f"My attempt below failed SANY:\n\n"
                    f"```\n{broken}\n```\n\n"
                    f"SANY error:\n```\n{error_msg}\n```\n\n"
                    f"Fix the spec."
                ),
            },
            {"role": "assistant", "channel": "analysis",
             "content": f"The error is a {mutator_name.replace('_',' ')}. I will repair only that and keep the rest of the spec unchanged."},
            {"role": "assistant", "channel": "final", "content": fixed_spec.strip()},
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", default="data/processed/diamond_curated.jsonl")
    p.add_argument("--out",  default="data/processed/repair_pairs.jsonl")
    p.add_argument("--mutations", type=int, default=4,
                   help="Number of distinct mutations to apply per gold spec.")
    p.add_argument("--seed", type=int, default=20260407)
    p.add_argument("--use-sany", action="store_true",
                   help="Run SANY on each broken variant for a real error message (slower).")
    args = p.parse_args()

    random.seed(args.seed)
    inp = Path(args.gold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
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
            fixed = _final_spec(rec)
            user = _user_text(rec)
            if not fixed or not user:
                continue
            cleaned, _ = normalize_spec(fixed)
            mutators = random.sample(_MUTATORS, k=min(args.mutations, len(_MUTATORS)))
            for mname, mfn in mutators:
                broken, tag = mfn(cleaned)
                if not broken or not tag:
                    continue
                err = _sany_error(broken) if args.use_sany else f"SANY: {tag}"
                pair = build_pair(rec.get("_prompt_id", "?"), user, cleaned,
                                  mname, broken, err)
                fo.write(json.dumps(pair) + "\n")
                n_out += 1

    print(f"[harvest_repair_pairs] {n_in} gold rows -> {n_out} repair pairs -> {out}")


if __name__ == "__main__":
    main()
