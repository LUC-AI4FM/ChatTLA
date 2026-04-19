#!/usr/bin/env python3
"""build_fork_a_corpora.py — Fork A combined corpora.

Concatenates the existing incremental-training corpus with the new
validator-segregated tlaplus/examples corpora, with configurable oversampling
of the new material to bias the gradient toward validator-verified specs.

Outputs:
  data/processed/fork_a_tlc_sft.jsonl   — base + tlc_target (upweighted)
  data/processed/fork_a_tlaps_sft.jsonl — base + tlaps_target (upweighted)

The base corpus is the same one v14 was trained on (diamond_sft_v4_upweight2x_plus_multitask.jsonl),
so Phase 2 SFT is strictly incremental over v14's data distribution.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE = _REPO_ROOT / "data" / "processed" / "diamond_sft_v4_upweight2x_plus_multitask.jsonl"
_TLC_NEW = _REPO_ROOT / "data" / "processed" / "tlc_target_sft.jsonl"
_TLAPS_NEW = _REPO_ROOT / "data" / "processed" / "tlaps_target_sft.jsonl"
_OUT_TLC = _REPO_ROOT / "data" / "processed" / "fork_a_tlc_sft.jsonl"
_OUT_TLAPS = _REPO_ROOT / "data" / "processed" / "fork_a_tlaps_sft.jsonl"
_SUMMARY = _REPO_ROOT / "data" / "processed" / "fork_a_corpora_summary.json"

OVERSAMPLE_NEW = 2
SEED = 20260419


def _load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    for p in (_BASE, _TLC_NEW, _TLAPS_NEW):
        if not p.exists():
            print(f"[error] missing input: {p}", file=sys.stderr)
            return 1

    base = _load(_BASE)
    tlc_new = _load(_TLC_NEW)
    tlaps_new = _load(_TLAPS_NEW)

    rng = random.Random(SEED)

    tlc_combined = list(base) + tlc_new * OVERSAMPLE_NEW
    tlaps_combined = list(base) + tlaps_new * OVERSAMPLE_NEW

    rng.shuffle(tlc_combined)
    rng.shuffle(tlaps_combined)

    _write(tlc_combined, _OUT_TLC)
    _write(tlaps_combined, _OUT_TLAPS)

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "oversample_new": OVERSAMPLE_NEW,
        "seed": SEED,
        "base_rows": len(base),
        "tlc_new_rows": len(tlc_new),
        "tlaps_new_rows": len(tlaps_new),
        "tlc_combined_rows": len(tlc_combined),
        "tlaps_combined_rows": len(tlaps_combined),
        "outputs": {
            "tlc": str(_OUT_TLC.relative_to(_REPO_ROOT)),
            "tlaps": str(_OUT_TLAPS.relative_to(_REPO_ROOT)),
        },
    }
    with _SUMMARY.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] base rows:              {len(base)}")
    print(f"[ok] tlc_new (×{OVERSAMPLE_NEW}):         {len(tlc_new)} -> {len(tlc_new) * OVERSAMPLE_NEW}")
    print(f"[ok] tlaps_new (×{OVERSAMPLE_NEW}):       {len(tlaps_new)} -> {len(tlaps_new) * OVERSAMPLE_NEW}")
    print(f"[ok] fork_a_tlc_sft.jsonl:   {len(tlc_combined)} rows -> {_OUT_TLC}")
    print(f"[ok] fork_a_tlaps_sft.jsonl: {len(tlaps_combined)} rows -> {_OUT_TLAPS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
