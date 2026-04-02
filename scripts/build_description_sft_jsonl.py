#!/usr/bin/env python3
"""
Build SFT JSONL from data/derived/tla_descriptions.json + on-disk .tla files.

- Train split: all rows whose module_name is NOT in benchmark holdout list.
- Holdout split: rows whose module_name is in holdout (for contamination checks / ablations).

Output format matches augmented.jsonl (developer + user + assistant analysis/final).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "tla_description_sources"))
sys.path.insert(0, str(_REPO))

from description_prompt import condense_description_row  # noqa: E402
from src.training.module_family import is_model_check_shim  # noqa: E402

_DEFAULT_DESC = _REPO / "data" / "derived" / "tla_descriptions.json"
_DEFAULT_BM = _REPO / "data" / "benchmarks" / "benchmark_to_module.json"
_OUT_TRAIN = _REPO / "data" / "processed" / "description_sft.jsonl"
_OUT_HOLDOUT = _REPO / "data" / "processed" / "description_sft_holdout.jsonl"

_DEVELOPER = """\
You are ChatTLA, an expert at writing verified TLA+ formal specifications.
Respond only with the TLA+ module, no commentary or explanation.
1. Start the module with ---- MODULE <ModuleName> ----
2. End with ====
3. Include EXTENDS, VARIABLES, Init, Next, and Spec operators
4. After the TLA+ module, append a TLC configuration block:
   SPECIFICATION Spec
   INVARIANT TypeOK   (if TypeOK is defined)

Critical TLA+ syntax rules:
- EXTENDS Integers for Int, +, -, *, \\div; EXTENDS Sequences for Seq, Append, Len, Head, Tail; EXTENDS FiniteSets for Cardinality, IsFiniteSet
- Declare ALL state variables in a VARIABLES line (every primed variable x' must appear in VARIABLES)
- Use = (not ==) inside Init and Next action conjuncts: /\\ x = value
- Function construction: [x \\in S |-> expr] (NOT [x \\in S : expr])
- Use \\in SUBSET S for set quantification (NOT \\E x \\subseteq S)
- Do NOT use PlusCal syntax (:=, --algorithm, labels, while, goto)
- TypeOK must be defined if referenced as INVARIANT
- Spec == Init /\\ [][Next]_vars where vars == <<v1, v2, ...>>"""


def load_holdout_modules(benchmark_path: Path) -> set[str]:
    if not benchmark_path.exists():
        return set()
    meta = json.loads(benchmark_path.read_text(encoding="utf-8"))
    return set(meta.get("holdout_module_names") or [])


def build_example(row: dict, tla_text: str) -> dict:
    user = (
        "Write a TLA+ specification matching the following structured description "
        f"(module name should match the reference: `{row.get('module_name', 'Spec')}`).\n\n"
        + condense_description_row(row)
    )
    return {
        "_tier": "description_sft",
        "_source": "tla_descriptions.json",
        "_module_name": row.get("module_name"),
        "messages": [
            {"role": "system", "content": _DEVELOPER},
            {"role": "user", "content": user},
            {"role": "assistant", "content": tla_text.strip()},
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build description_sft.jsonl from tla_descriptions + .tla files")
    ap.add_argument("--descriptions", type=Path, default=_DEFAULT_DESC)
    ap.add_argument("--benchmark-map", type=Path, default=_DEFAULT_BM)
    ap.add_argument("--out-train", type=Path, default=_OUT_TRAIN)
    ap.add_argument("--out-holdout", type=Path, default=_OUT_HOLDOUT)
    ap.add_argument("--max-tla-chars", type=int, default=120_000, help="Cap reference spec size")
    ap.add_argument(
        "--no-skip-mc-shims",
        action="store_true",
        help="Include TLC MC* wrapper modules (default: skip — they pair long NL with stub .tla)",
    )
    args = ap.parse_args()

    rows = json.loads(args.descriptions.read_text(encoding="utf-8"))
    holdout = load_holdout_modules(args.benchmark_map)

    train_ex: list[dict] = []
    hold_ex: list[dict] = []
    skipped = 0

    for row in rows:
        mn = row.get("module_name")
        paths = row.get("paths") or {}
        local = paths.get("local_tla")
        if not local:
            skipped += 1
            continue
        tla_path = _REPO / local
        if not tla_path.is_file():
            skipped += 1
            continue
        try:
            tla_text = tla_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            skipped += 1
            continue
        if len(tla_text) > args.max_tla_chars:
            tla_text = tla_text[: args.max_tla_chars] + "\n\\* [truncated for training cap]\n"
        if not args.no_skip_mc_shims and is_model_check_shim(mn, tla_text):
            skipped += 1
            continue
        ex = build_example(row, tla_text)
        if mn in holdout:
            hold_ex.append(ex)
        else:
            train_ex.append(ex)

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    with args.out_train.open("w", encoding="utf-8") as f:
        for ex in train_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with args.out_holdout.open("w", encoding="utf-8") as f:
        for ex in hold_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[build_description_sft] train={len(train_ex)} -> {args.out_train}")
    print(f"[build_description_sft] holdout={len(hold_ex)} -> {args.out_holdout}")
    print(f"[build_description_sft] skipped_rows={skipped} (includes MC* TLC shims unless --no-skip-mc-shims) holdout_modules={len(holdout)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
