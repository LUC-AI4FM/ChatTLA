#!/usr/bin/env python3
"""validate_one_diamond.py — single-spec Diamond validator wrapper.

Reads a TLA+ spec from a file (or stdin) and prints a JSON validation
report. Used by the Diamond-extension subagents so they don't have to
write Python import boilerplate around scripts.diamond_sft_gen.

Usage:
    python3 scripts/validate_one_diamond.py path/to/Spec.tla
    cat Spec.tla | python3 scripts/validate_one_diamond.py -

Output (JSON, one object on stdout):
    {
      "module": "...",
      "sany_pass": bool,
      "tlc_tier": "bronze|silver|gold",
      "is_diamond": bool,
      "distinct_states": int,
      "invariants_checked": int,
      "trivial_invariant": bool,
      "mutation_tested": bool,
      "mutation_caught": bool,
      "error": "..."
    }
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.diamond_sft_gen import validate_diamond, _get_module_name


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_one_diamond.py <spec.tla|->", file=sys.stderr)
        return 2
    arg = sys.argv[1]
    spec = sys.stdin.read() if arg == "-" else Path(arg).read_text(encoding="utf-8")
    if "MODULE" not in spec:
        print(json.dumps({"error": "no MODULE found"}))
        return 1
    r = validate_diamond(spec, prompt_id="oneshot")
    out = {
        "module": _get_module_name(spec),
        "sany_pass": r.sany_pass,
        "tlc_tier": r.tlc_tier,
        "is_diamond": r.is_diamond,
        "distinct_states": r.distinct_states,
        "invariants_checked": r.invariants_checked,
        "trivial_invariant": r.trivial_invariant,
        "mutation_tested": r.mutation_tested,
        "mutation_caught": r.mutation_caught,
        "error": r.error,
    }
    print(json.dumps(out, indent=2))
    return 0 if r.is_diamond else 1


if __name__ == "__main__":
    raise SystemExit(main())
