#!/usr/bin/env python3
"""Finalize puzzles_classical batch: validate all 20 modules and write JSONL."""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.diamond_sft_gen import validate_diamond  # noqa: E402

WORK = REPO / "outputs/diamond_gen/puzzles_classical_work"
OUT = REPO / "outputs/diamond_gen/puzzles_classical.jsonl"
TOPICS = json.loads((REPO / "data/diamond_gen_topics.json").read_text())

# Build module -> desc map for puzzles_classical batch.
batch = next(b for b in TOPICS["batches"] if b["id"] == "puzzles_classical")
desc_by_module = {t["module"]: t["desc"] for t in batch["topics"]}
order = [t["module"] for t in batch["topics"]]

records = []
diamond_count = 0
for mod in order:
    p = WORK / f"{mod}.tla"
    spec = p.read_text()
    r = validate_diamond(spec)
    rec = {
        "module": mod,
        "topic_desc": desc_by_module[mod],
        "spec": spec,
        "is_diamond": bool(r.is_diamond),
        "tier": r.tlc_tier,
        "distinct_states": int(r.distinct_states),
        "invariants_checked": int(r.invariants_checked),
        "mutation_caught": bool(r.mutation_caught),
        "trivial_invariant": bool(r.trivial_invariant),
        "attempts": 1,
        "fail_reason": "" if r.is_diamond else (r.error or "not_diamond"),
    }
    records.append(rec)
    if r.is_diamond:
        diamond_count += 1
    print(f"{mod}: diamond={r.is_diamond} tier={r.tlc_tier} states={r.distinct_states}")

with open(OUT, "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

print(f"\nWrote {len(records)} records to {OUT}")
print(f"Diamond: {diamond_count}/{len(records)}")
failures = [r["module"] for r in records if not r["is_diamond"]]
print(f"Failures: {failures}")
