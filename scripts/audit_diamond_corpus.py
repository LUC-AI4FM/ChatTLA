#!/usr/bin/env python3
"""Audit any SFT jsonl corpus against the Diamond semantic gate.

Background
----------
`project_semantic_analysis_20260405.md` flagged that the legacy gold corpus
has 0% mutation coverage — it was syntactically valid but semantically empty.
The Diamond gate (src/validators/tlc_validator.SemanticInfo.is_diamond)
enforces:

  distinct_states > 1
  not trivial_invariant
  invariants_checked > 0
  mutation_tested AND mutation_caught

Diamond generation already enforces this, but legacy data still flows into
training via train.jsonl. This script audits a corpus and either:

  --report  : just print pass/fail counts
  --filter  : write a copy with non-Diamond rows dropped
  --rescore : re-run TLC + mutation test on every spec to materialize
              missing _semantic fields, then filter

By default it does --report.

Usage:
    python -m scripts.audit_diamond_corpus --in data/processed/train.jsonl
    python -m scripts.audit_diamond_corpus --in data/processed/train.jsonl \\
        --filter --out data/processed/train_diamond_only.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _is_diamond_from_semantic(sem: dict) -> bool:
    """Mirrors SemanticInfo.is_diamond using the dict serialization.

    Note: older serialized records omit `mutation_tested` but include
    `mutation_caught`. If `mutation_caught` is True, `mutation_tested` must
    also have been True at validation time, so we infer it."""
    if not isinstance(sem, dict):
        return False
    mutation_caught = bool(sem.get("mutation_caught", False))
    mutation_tested = bool(sem.get("mutation_tested", mutation_caught))
    return (
        int(sem.get("distinct_states", 0)) > 1
        and not bool(sem.get("trivial_invariant", False))
        and int(sem.get("invariants_checked", 0)) > 0
        and mutation_tested
        and mutation_caught
    )


def _final_spec(record: dict) -> str | None:
    for m in record.get("messages", []):
        if m.get("role") == "assistant" and m.get("channel") == "final":
            return m.get("content")
    return None


def _rescore(record: dict, timeout: int = 30) -> dict | None:
    """Run TLC + semantic info on this record's spec. Returns a fresh
    `_semantic` dict or None if validation failed."""
    from src.validators.tlc_validator import validate_string

    spec = _final_spec(record)
    if not spec:
        return None
    m = re.search(r"^-{2,}\s*MODULE\s+(\w+)", spec, re.MULTILINE)
    name = m.group(1) if m else "Audit"
    try:
        res = validate_string(spec, module_name=name, timeout=timeout)
    except Exception:
        return None
    sem = res.semantic
    return {
        "distinct_states": sem.distinct_states,
        "action_coverage": sem.action_coverage,
        "invariants_checked": sem.invariants_checked,
        "trivial_invariant": sem.trivial_invariant,
        "mutation_tested": sem.mutation_tested,
        "mutation_caught": sem.mutation_caught,
        "_audit_tier": res.tier,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in",  dest="inp", required=True)
    p.add_argument("--out", dest="out", default=None)
    p.add_argument("--filter",  action="store_true",
                   help="Write a Diamond-only copy of the corpus to --out.")
    p.add_argument("--rescore", action="store_true",
                   help="Run TLC + mutation test on rows missing _semantic.")
    p.add_argument("--rescore-all", action="store_true",
                   help="Re-run TLC even on rows that already have _semantic.")
    p.add_argument("--timeout", type=int, default=30)
    args = p.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        sys.exit(f"input file not found: {inp}")
    out_fh = None
    if args.filter:
        if not args.out:
            sys.exit("--filter requires --out")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fh = out_path.open("w")

    n_total = 0
    n_diamond = 0
    n_no_semantic = 0
    n_rescored = 0
    n_kept = 0
    by_tier: dict[str, int] = {}

    for line in inp.open():
        line = line.strip()
        if not line:
            continue
        n_total += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        sem = rec.get("_semantic")
        if (sem is None and args.rescore) or args.rescore_all:
            new_sem = _rescore(rec, timeout=args.timeout)
            if new_sem is not None:
                sem = new_sem
                rec["_semantic"] = new_sem
                n_rescored += 1

        if sem is None:
            n_no_semantic += 1
            tier = "unknown"
        else:
            tier = sem.get("_audit_tier") or rec.get("_tier") or "unknown"

        is_diamond = _is_diamond_from_semantic(sem or {})
        if is_diamond:
            n_diamond += 1
        by_tier[tier] = by_tier.get(tier, 0) + 1

        if out_fh and is_diamond:
            out_fh.write(json.dumps(rec) + "\n")
            n_kept += 1

    if out_fh:
        out_fh.close()

    print(f"[audit_diamond_corpus] {inp}")
    print(f"  total rows:        {n_total}")
    print(f"  diamond:           {n_diamond} ({100*n_diamond/max(n_total,1):.1f}%)")
    print(f"  missing _semantic: {n_no_semantic}")
    if args.rescore or args.rescore_all:
        print(f"  rescored:          {n_rescored}")
    if by_tier:
        print(f"  tier breakdown:")
        for t, c in sorted(by_tier.items(), key=lambda kv: -kv[1]):
            print(f"    {t:24s} {c}")
    if out_fh:
        print(f"  diamond-only kept: {n_kept} -> {args.out}")


if __name__ == "__main__":
    main()
