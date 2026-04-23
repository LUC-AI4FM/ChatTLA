#!/usr/bin/env python3
"""Emit transactions_databases.jsonl from validated specs."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "outputs/diamond_gen/transactions_databases_work"
OUT  = ROOT / "outputs/diamond_gen/transactions_databases.jsonl"

sys.path.insert(0, str(ROOT))
from scripts.diamond_sft_gen import validate_diamond  # noqa

# Module -> (topic_desc, attempts) order from data/diamond_gen_topics.json
TOPICS = json.load(open(ROOT / "data/diamond_gen_topics.json"))
batch = next(b for b in TOPICS["batches"] if b["id"] == "transactions_databases")
order = [(t["module"], t["desc"]) for t in batch["topics"]]

# Map suggested topic module -> actual file basename in this batch.
NAME_MAP = {m: m for m, _ in order}

# Track attempts roughly: from session, OCC and WAL took 2, SplitBrain 2.
ATTEMPTS = {"OptimisticConcurrency": 2, "WriteAheadLog": 2, "SplitBrain": 2,
            "TransferAtomic": 2}

records = []
for module, desc in order:
    fname = NAME_MAP[module]
    path = WORK / f"{fname}.tla"
    if not path.exists():
        records.append({
            "module": module, "topic_desc": desc, "spec": "",
            "is_diamond": False, "tier": "missing", "distinct_states": 0,
            "invariants_checked": 0, "mutation_caught": False,
            "trivial_invariant": False, "attempts": 0,
            "fail_reason": "no spec file",
        })
        continue
    spec = path.read_text()
    r = validate_diamond(spec)
    records.append({
        "module": module,
        "topic_desc": desc,
        "spec": spec,
        "is_diamond": bool(r.is_diamond),
        "tier": r.tlc_tier,
        "distinct_states": r.distinct_states,
        "invariants_checked": r.invariants_checked,
        "mutation_caught": r.mutation_caught,
        "trivial_invariant": r.trivial_invariant,
        "attempts": ATTEMPTS.get(module, 1),
        "fail_reason": r.error or "",
    })
    print(f"{module}: diamond={r.is_diamond} tier={r.tlc_tier} "
          f"states={r.distinct_states} mut={r.mutation_caught}")

with open(OUT, "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

n = sum(1 for r in records if r["is_diamond"])
print(f"\nWROTE {OUT}")
print(f"BATCH transactions_databases DONE: {n}/{len(records)} diamond")
fails = [r["module"] for r in records if not r["is_diamond"]]
if fails:
    print("FAILURES:", ", ".join(fails))
