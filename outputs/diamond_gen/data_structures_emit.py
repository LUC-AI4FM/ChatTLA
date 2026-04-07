#!/usr/bin/env python3
"""Generate data_structures.jsonl from validated specs."""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.diamond_sft_gen import validate_diamond  # noqa

WORK = REPO / "outputs/diamond_gen/data_structures_work"
OUT = REPO / "outputs/diamond_gen/data_structures.jsonl"
TOPICS = REPO / "data/diamond_gen_topics.json"

batch = next(b for b in json.loads(TOPICS.read_text())["batches"]
             if b["id"] == "data_structures")

# Map module name -> topic_desc; allow renames by best-effort lookup.
desc_map = {t["module"]: t["desc"] for t in batch["topics"]}
order = [t["module"] for t in batch["topics"]]

results = []
for mod in order:
    p = WORK / f"{mod}.tla"
    if not p.exists():
        results.append({"module": mod, "topic_desc": desc_map[mod], "spec": "",
                        "is_diamond": False, "tier": "missing", "distinct_states": 0,
                        "invariants_checked": 0, "mutation_caught": False,
                        "trivial_invariant": False, "attempts": 0,
                        "fail_reason": "no_spec_file"})
        continue
    spec = p.read_text()
    r = validate_diamond(spec)
    results.append({
        "module": mod,
        "topic_desc": desc_map[mod],
        "spec": spec,
        "is_diamond": bool(r.is_diamond),
        "tier": r.tlc_tier,
        "distinct_states": r.distinct_states,
        "invariants_checked": r.invariants_checked,
        "mutation_caught": r.mutation_caught,
        "trivial_invariant": r.trivial_invariant,
        "attempts": 1,
        "fail_reason": r.error or "",
    })

with OUT.open("w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

n = sum(1 for r in results if r["is_diamond"])
print(f"BATCH data_structures DONE: {n}/20 diamond")
fails = [r["module"] for r in results if not r["is_diamond"]]
if fails:
    print("Failures:", fails)
