#!/usr/bin/env python3
"""carve_diamond_holdout.py — split diamond_generated.jsonl into train + holdout.

Holdout policy:
  - 3 specs per batch, picked deterministically by sorted module name
    (so the split is reproducible and not dependent on JSONL row order).
  - 10 batches x 3 = 30 holdout specs.
  - Remaining 170 + (the 73 existing diamond_curated specs from prior runs,
    NOT touched here) become the SFT training pool.

Outputs:
  data/processed/diamond_eval_holdout.jsonl   -- never train on this
  data/processed/diamond_generated_train.jsonl -- the 170 train specs
                                                  (chat-format conversion is a
                                                  separate step)

The training script must filter against diamond_eval_holdout.jsonl by module
name to be safe even if files are accidentally concatenated.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GEN = _REPO_ROOT / "outputs" / "diamond_gen" / "diamond_generated.jsonl"
_EXISTING_TRAIN = _REPO_ROOT / "data" / "processed" / "train.jsonl"
_HOLDOUT = _REPO_ROOT / "data" / "processed" / "diamond_eval_holdout.jsonl"
_TRAIN = _REPO_ROOT / "data" / "processed" / "diamond_generated_train.jsonl"

HOLDOUT_PER_BATCH = 3


def _existing_train_modules() -> set[str]:
    """Modules already present in the existing train.jsonl. Holdout MUST exclude
    these — otherwise the baseline model has already been trained on them and
    the eval is biased."""
    import re
    if not _EXISTING_TRAIN.exists():
        return set()
    mods: set[str] = set()
    with _EXISTING_TRAIN.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for m in r.get("messages", []):
                if m.get("role") == "assistant" and "MODULE" in m.get("content", ""):
                    mm = re.search(r"MODULE\s+(\w+)", m["content"])
                    if mm:
                        mods.add(mm.group(1))
                    break
    return mods


def main() -> int:
    if not _GEN.exists():
        sys.exit(f"missing {_GEN}; run aggregate_diamond_gen.py first")

    contaminated = _existing_train_modules()
    print(f"  existing train.jsonl has {len(contaminated)} modules; "
          f"any overlap with our generated set will be excluded from holdout")

    by_batch: dict[str, list[dict]] = defaultdict(list)
    with _GEN.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("is_diamond"):
                continue  # only diamond rows are eligible for either split
            by_batch[r["batch"]].append(r)

    holdout: list[dict] = []
    train: list[dict] = []
    for batch, recs in sorted(by_batch.items()):
        recs_sorted = sorted(recs, key=lambda r: r["module"])
        # Pick the first HOLDOUT_PER_BATCH modules NOT already in train.jsonl.
        eligible = [r for r in recs_sorted if r["module"] not in contaminated]
        if len(eligible) < HOLDOUT_PER_BATCH:
            print(f"  WARN batch {batch}: only {len(eligible)} clean modules available")
        ho = eligible[:HOLDOUT_PER_BATCH]
        ho_mods = {r["module"] for r in ho}
        tr = [r for r in recs_sorted if r["module"] not in ho_mods]
        holdout.extend(ho)
        train.extend(tr)
        print(f"  {batch:30s} train={len(tr):3d}  holdout={len(ho)}  "
              f"holdout_modules={[r['module'] for r in ho]}")

    _HOLDOUT.parent.mkdir(parents=True, exist_ok=True)
    _HOLDOUT.write_text("".join(json.dumps(r) + "\n" for r in holdout))
    _TRAIN.write_text("".join(json.dumps(r) + "\n" for r in train))

    print()
    print(f"holdout: {len(holdout)} -> {_HOLDOUT}")
    print(f"train  : {len(train)} -> {_TRAIN}")
    print()
    print("Holdout module names (these MUST be excluded from any training set):")
    for r in holdout:
        print(f"  {r['batch']:30s} {r['module']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
