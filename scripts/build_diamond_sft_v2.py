#!/usr/bin/env python3
"""build_diamond_sft_v2.py — assemble the final SFT corpus for the
incremental gpt-oss:20b fine-tune.

Inputs:
  data/processed/diamond_generated_train.jsonl   170 new train specs (no chat fmt)
  outputs/diamond_gen/cot_train.json              {module: cot_string} from the
                                                  background CoT generator
  data/processed/diamond_curated.jsonl            73 existing chat-format records

Output:
  data/processed/diamond_sft_v2.jsonl             243 chat-format records ready
                                                  for the trainer
  data/processed/diamond_sft_v2_summary.json

Chat format mirrors diamond_curated.jsonl exactly: developer / user / assistant(CoT) / assistant(spec).

Holdout safety:
  We re-load data/processed/diamond_eval_holdout.jsonl and assert that NO holdout
  module name appears in the output. If any does, exit non-zero — this is the
  belt-and-suspenders against accidentally training on the eval set.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_IN = _REPO_ROOT / "data" / "processed" / "diamond_generated_train.jsonl"
_COT_IN = _REPO_ROOT / "outputs" / "diamond_gen" / "cot_train.json"
_CURATED_IN = _REPO_ROOT / "data" / "processed" / "diamond_curated.jsonl"
_HOLDOUT_IN = _REPO_ROOT / "data" / "processed" / "diamond_eval_holdout.jsonl"
_OUT = _REPO_ROOT / "data" / "processed" / "diamond_sft_v2.jsonl"
_SUMMARY = _REPO_ROOT / "data" / "processed" / "diamond_sft_v2_summary.json"

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


def _user_turn(topic_desc: str) -> str:
    return f"Write a TLA+ specification for the following:\n\n{topic_desc}\n"


def _build_record(rec: dict, cot: str, source_tag: str) -> dict:
    return {
        "_tier": "diamond_curated",
        "_prompt_id": f"{source_tag}/{rec['module']}",
        "_source": source_tag,
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


def main() -> int:
    if not _TRAIN_IN.exists():
        sys.exit(f"missing {_TRAIN_IN}; run carve_diamond_holdout.py first")
    if not _COT_IN.exists():
        sys.exit(f"missing {_COT_IN}; the CoT background agent has not finished")
    if not _CURATED_IN.exists():
        sys.exit(f"missing {_CURATED_IN}")
    if not _HOLDOUT_IN.exists():
        sys.exit(f"missing {_HOLDOUT_IN}; run carve_diamond_holdout.py first")

    holdout_modules: set[str] = set()
    with _HOLDOUT_IN.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            holdout_modules.add(json.loads(line)["module"])
    print(f"[build] holdout has {len(holdout_modules)} modules to exclude")

    cot_map: dict[str, str] = json.loads(_COT_IN.read_text())
    print(f"[build] loaded {len(cot_map)} CoT entries from {_COT_IN}")

    out_records: list[dict] = []
    missing_cot: list[str] = []
    leaked_holdout: list[str] = []

    with _TRAIN_IN.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mod = rec["module"]
            if mod in holdout_modules:
                leaked_holdout.append(mod)
                continue  # safety: never include a holdout module
            cot = cot_map.get(mod)
            if not cot or len(cot) < 200:
                missing_cot.append(mod)
                continue
            out_records.append(_build_record(rec, cot, "diamond_gen_v2"))

    n_new = len(out_records)
    print(f"[build] new train records (with CoT): {n_new}")
    if missing_cot:
        print(f"[build] WARNING: {len(missing_cot)} train specs lacked usable CoT — skipped:")
        for m in missing_cot:
            print(f"    {m}")
    if leaked_holdout:
        print(f"[build] (filtered out {len(leaked_holdout)} holdout modules from train input — expected to be 0)")

    n_curated_kept = 0
    with _CURATED_IN.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # The legacy curated records use _prompt_id like
            # "gold_all_benchmarks/gold_all_benchmarks_18". They do not collide
            # with diamond_gen_v2 module names by construction. Pass through.
            out_records.append(rec)
            n_curated_kept += 1
    print(f"[build] legacy diamond_curated rows passed through: {n_curated_kept}")

    out_records_safe: list[dict] = []
    final_leaks: list[str] = []
    for r in out_records:
        # Best-effort module name extraction from the assistant spec turn for
        # legacy records (which lack a top-level module field).
        spec_turn = ""
        for m in r.get("messages", []):
            if m.get("role") == "assistant" and "MODULE" in m.get("content", ""):
                spec_turn = m["content"]
                break
        mod = ""
        if spec_turn:
            import re
            mm = re.search(r"MODULE\s+(\w+)", spec_turn)
            if mm:
                mod = mm.group(1)
        if mod and mod in holdout_modules:
            final_leaks.append(mod)
            continue
        out_records_safe.append(r)

    if final_leaks:
        print(f"[build] FATAL: holdout leaked into output via legacy record(s): {final_leaks}")
        return 2

    _OUT.write_text("".join(json.dumps(r) + "\n" for r in out_records_safe))
    summary = {
        "total": len(out_records_safe),
        "new_diamond_gen_v2": n_new,
        "legacy_curated": n_curated_kept,
        "holdout_excluded": len(holdout_modules),
        "missing_cot": missing_cot,
        "out": str(_OUT),
    }
    _SUMMARY.write_text(json.dumps(summary, indent=2))
    print()
    print(f"[build] wrote {len(out_records_safe)} chat-format records -> {_OUT}")
    print(f"[build] summary -> {_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
