#!/usr/bin/env python3
"""build_diamond_sft_v3.py — assemble the SFT corpus for the incremental
gpt-oss:20b fine-tune from checkpoint-131.

Recipe (per user choice 2026-04-07):
  base = data/processed/train.jsonl                  (existing 713 records,
                                                      what checkpoint-131 was
                                                      trained on)
  new  = 170 diamond_gen_v2 specs in chat-message
         format, CoT from cot_train_part{1..4}.json
  mix  = base + (new x2)                              (oversample new 2x to bias
                                                      the gradient toward the
                                                      new algorithm families)

Outputs:
  data/processed/diamond_sft_v3.jsonl                 the final corpus the
                                                      trainer reads
  data/processed/diamond_sft_v3_summary.json

Holdout safety:
  Loads data/processed/diamond_eval_holdout.jsonl, builds the set of holdout
  module names, and asserts that NO record in the output contains any holdout
  module's MODULE keyword. Hard exit if any leak is found.

Inputs required:
  data/processed/train.jsonl                          existing chat-format records
  data/processed/diamond_generated_train.jsonl       170 new specs (no chat fmt)
  outputs/diamond_gen/cot_train_part1.json           CoT slice 1 (43 modules)
  outputs/diamond_gen/cot_train_part2.json           CoT slice 2 (43 modules)
  outputs/diamond_gen/cot_train_part3.json           CoT slice 3 (42 modules)
  outputs/diamond_gen/cot_train_part4.json           CoT slice 4 (42 modules)
  data/processed/diamond_eval_holdout.jsonl           30 holdout records
"""
from __future__ import annotations
import json
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE = _REPO_ROOT / "data" / "processed" / "train.jsonl"
_NEW_SPECS = _REPO_ROOT / "data" / "processed" / "diamond_generated_train.jsonl"
_COT_PARTS = [
    _REPO_ROOT / "outputs" / "diamond_gen" / f"cot_train_part{i}.json"
    for i in (1, 2, 3, 4)
]
_HOLDOUT = _REPO_ROOT / "data" / "processed" / "diamond_eval_holdout.jsonl"
_OUT = _REPO_ROOT / "data" / "processed" / "diamond_sft_v3.jsonl"
_SUMMARY = _REPO_ROOT / "data" / "processed" / "diamond_sft_v3_summary.json"

OVERSAMPLE_NEW = 2

DEVELOPER_PROMPT = """You are ChatTLA, an expert at writing verified TLA+ formal specifications.
When asked to write a TLA+ spec, follow these rules exactly:
1. Start the module with ---- MODULE <ModuleName> ----
2. End with ====
3. Include EXTENDS, VARIABLES, Init, Next, and Spec operators
4. After the TLA+ module, append a TLC configuration block:
   SPECIFICATION Spec
   INVARIANT TypeOK   (if TypeOK is defined)
5. Output only valid TLA+ code. No markdown fences, no explanation outside the spec.
Reasoning: medium"""

_MOD_RE = re.compile(r"MODULE\s+(\w+)")


def _user_turn(topic_desc: str) -> str:
    return f"Write a TLA+ specification for the following:\n\n{topic_desc}\n"


def _build_record(rec: dict, cot: str) -> dict:
    return {
        "_tier": "diamond_curated",
        "_prompt_id": f"diamond_gen_v2/{rec['module']}",
        "_source": "diamond_gen_v2",
        "_timestamp": datetime.utcnow().isoformat(),
        "_semantic": {
            "distinct_states": rec.get("distinct_states", 0),
            "invariants_checked": rec.get("invariants_checked", 0),
            "mutation_caught": rec.get("mutation_caught", True),
            "trivial_invariant": rec.get("trivial_invariant", False),
        },
        "messages": [
            {"role": "developer", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": _user_turn(rec["topic_desc"])},
            {"role": "assistant", "content": cot},
            {"role": "assistant", "content": rec["spec"]},
        ],
    }


def _record_modules(rec: dict) -> set[str]:
    """All TLA+ modules referenced in any assistant turn of a chat record."""
    mods: set[str] = set()
    for m in rec.get("messages", []):
        if m.get("role") == "assistant":
            for mm in _MOD_RE.findall(m.get("content", "")):
                mods.add(mm)
    return mods


def main() -> int:
    for p in (_BASE, _NEW_SPECS, _HOLDOUT, *_COT_PARTS):
        if not p.exists():
            sys.exit(f"missing {p}")

    holdout_modules: set[str] = set()
    with _HOLDOUT.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            holdout_modules.add(json.loads(line)["module"])
    print(f"[v3] holdout: {len(holdout_modules)} modules to exclude")

    cot_map: dict[str, str] = {}
    for p in _COT_PARTS:
        d = json.loads(p.read_text())
        cot_map.update(d)
        print(f"[v3] loaded {len(d)} CoT entries from {p.name}")
    print(f"[v3] total CoT entries: {len(cot_map)}")

    new_records: list[dict] = []
    missing_cot: list[str] = []
    leaked_in_new: list[str] = []
    with _NEW_SPECS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mod = rec["module"]
            if mod in holdout_modules:
                leaked_in_new.append(mod)
                continue
            cot = cot_map.get(mod)
            if not cot or len(cot) < 200:
                missing_cot.append(mod)
                continue
            new_records.append(_build_record(rec, cot))
    print(f"[v3] new diamond_gen_v2 records: {len(new_records)}")
    if missing_cot:
        print(f"[v3] WARNING {len(missing_cot)} new specs lacked usable CoT — skipped:")
        for m in missing_cot:
            print(f"     {m}")
    if leaked_in_new:
        sys.exit(f"[v3] FATAL: {len(leaked_in_new)} holdout modules in new specs file")

    base_records: list[dict] = []
    base_leaks: list[str] = []
    with _BASE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mods = _record_modules(rec)
            if mods & holdout_modules:
                base_leaks.append(",".join(sorted(mods & holdout_modules)))
                continue  # quietly drop any base record that contains a holdout module
            base_records.append(rec)
    print(f"[v3] base train.jsonl rows kept: {len(base_records)}")
    if base_leaks:
        print(f"[v3] dropped {len(base_leaks)} base records that mentioned holdout modules: {base_leaks[:5]}")

    out: list[dict] = []
    out.extend(base_records)
    for _ in range(OVERSAMPLE_NEW):
        out.extend(new_records)

    final_leaks: list[str] = []
    for r in out:
        mods = _record_modules(r)
        if mods & holdout_modules:
            final_leaks.append(",".join(sorted(mods & holdout_modules)))
    if final_leaks:
        sys.exit(f"[v3] FATAL: {len(final_leaks)} records leaked holdout: {final_leaks[:5]}")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("".join(json.dumps(r) + "\n" for r in out))

    summary = {
        "out": str(_OUT),
        "base_kept": len(base_records),
        "base_dropped_for_holdout": len(base_leaks),
        "new_unique": len(new_records),
        "oversample": OVERSAMPLE_NEW,
        "total_records": len(out),
        "holdout_modules_excluded": sorted(holdout_modules),
        "missing_cot": missing_cot,
    }
    _SUMMARY.write_text(json.dumps(summary, indent=2))
    print()
    print(f"[v3] wrote {len(out)} records -> {_OUT}")
    print(f"[v3] summary -> {_SUMMARY}")
    print(f"[v3] composition: base={len(base_records)} + new={len(new_records)}*{OVERSAMPLE_NEW} = {len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
