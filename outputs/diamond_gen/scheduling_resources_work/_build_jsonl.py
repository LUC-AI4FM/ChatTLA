#!/usr/bin/env python3
"""Validate every spec in scheduling_resources_work and emit the JSONL."""
import json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WORK = REPO / "outputs/diamond_gen/scheduling_resources_work"
OUT  = REPO / "outputs/diamond_gen/scheduling_resources.jsonl"
TOPICS = REPO / "data/diamond_gen_topics.json"

with open(TOPICS) as f:
    batches = json.load(f)["batches"]
topics = next(b for b in batches if b["id"] == "scheduling_resources")["topics"]
desc_by_module = {t["module"]: t["desc"] for t in topics}

# Module-name remappings if any
ALIASES = {}

results = []
for topic in topics:
    mod = ALIASES.get(topic["module"], topic["module"])
    tla_path = WORK / f"{mod}.tla"
    if not tla_path.exists():
        results.append({
            "module": mod, "topic_desc": topic["desc"], "spec": "",
            "is_diamond": False, "tier": "missing", "distinct_states": 0,
            "invariants_checked": 0, "mutation_caught": False,
            "trivial_invariant": False, "attempts": 0,
            "fail_reason": "file_missing",
        })
        continue
    spec = tla_path.read_text()
    proc = subprocess.run(
        ["python3", "scripts/validate_diamond_cli.py", str(tla_path)],
        cwd=REPO, capture_output=True, text=True,
    )
    try:
        v = json.loads(proc.stdout)
    except Exception:
        v = {"is_diamond": False, "tier": "error", "distinct_states": 0,
             "invariants_checked": 0, "mutation_caught": False,
             "trivial_invariant": False, "error": proc.stdout[:200]}
    results.append({
        "module": mod,
        "topic_desc": topic["desc"],
        "spec": spec,
        "is_diamond": bool(v.get("is_diamond")),
        "tier": v.get("tier", ""),
        "distinct_states": int(v.get("distinct_states", 0)),
        "invariants_checked": int(v.get("invariants_checked", 0)),
        "mutation_caught": bool(v.get("mutation_caught", False)),
        "trivial_invariant": bool(v.get("trivial_invariant", False)),
        "attempts": 1,
        "fail_reason": v.get("error", "") if not v.get("is_diamond") else "",
    })

with open(OUT, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

n_diamond = sum(1 for r in results if r["is_diamond"])
print(f"Wrote {len(results)} rows to {OUT}")
print(f"BATCH scheduling_resources DONE: {n_diamond}/20 diamond")
fails = [r["module"] for r in results if not r["is_diamond"]]
if fails:
    print("FAILURES:", ", ".join(fails))
