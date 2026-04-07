#!/usr/bin/env python3
"""validate_diamond_cli.py — thin CLI wrapper around scripts.diamond_sft_gen.validate_diamond.

Reads a .tla file (or stdin) and prints a JSON verdict:
  {"is_diamond": bool, "tier": "gold|silver|bronze", "distinct_states": int,
   "mutation_tested": bool, "mutation_caught": bool, "invariants_checked": int,
   "trivial_invariant": bool, "error": str}

Used by the parallel diamond generation subagents to self-grade their work.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.diamond_sft_gen import validate_diamond  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", help=".tla file (omit to read stdin)")
    args = p.parse_args()

    spec = Path(args.path).read_text() if args.path else sys.stdin.read()
    r = validate_diamond(spec)
    out = {
        "is_diamond": bool(r.is_diamond),
        "tier": r.tlc_tier,
        "sany_pass": r.sany_pass,
        "distinct_states": r.distinct_states,
        "action_coverage": r.action_coverage,
        "invariants_checked": r.invariants_checked,
        "mutation_tested": r.mutation_tested,
        "mutation_caught": r.mutation_caught,
        "trivial_invariant": r.trivial_invariant,
        "error": r.error,
    }
    print(json.dumps(out, indent=2))
    return 0 if r.is_diamond else 1


if __name__ == "__main__":
    raise SystemExit(main())
