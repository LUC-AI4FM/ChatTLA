#!/usr/bin/env python3
"""build_tlaplus_examples_sft.py — Fork A corpus builder.

Reads the labeled tlaplus/examples dump produced by the scraper
(data/processed/tlaplus_examples_labeled.jsonl) and emits two
validator-segregated SFT corpora:

  data/processed/tlc_target_sft.jsonl     — specs that pass TLC
  data/processed/tlaps_target_sft.jsonl   — specs that have verified TLAPS proofs

Excludes any spec whose module name collides with the 30 diamond_eval_holdout
modules so downstream evals remain unseen.

Also writes a per-corpus summary JSON with counts and source-spec manifests.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LABELED = _REPO_ROOT / "data" / "processed" / "tlaplus_examples_labeled.jsonl"
_HOLDOUT = _REPO_ROOT / "data" / "processed" / "diamond_eval_holdout.jsonl"
_OUT_TLC = _REPO_ROOT / "data" / "processed" / "tlc_target_sft.jsonl"
_OUT_TLAPS = _REPO_ROOT / "data" / "processed" / "tlaps_target_sft.jsonl"
_SUMMARY = _REPO_ROOT / "data" / "processed" / "tlaplus_examples_sft_summary.json"

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

_MOD_RE = re.compile(r"----\s*MODULE\s+(\w+)")


def _load_holdout_modules() -> set[str]:
    mods = set()
    if not _HOLDOUT.exists():
        return mods
    with _HOLDOUT.open() as f:
        for line in f:
            rec = json.loads(line)
            if "module" in rec:
                mods.add(rec["module"])
    return mods


def _user_turn(description: str) -> str:
    return f"Write a TLA+ specification for the following:\n\n{description}\n"


def _analysis_for(description: str, tla_source: str) -> str:
    mods = _MOD_RE.findall(tla_source)
    mod = mods[0] if mods else "Spec"
    return (
        f"The problem asks for {description.strip().rstrip('.')}. "
        f"I'll model it as TLA+ module {mod} with explicit VARIABLES, "
        "an Init predicate, a Next-state action, and a safety invariant "
        "expressed in terms of those variables."
    )


def _record(rec: dict, tier: str) -> dict:
    description = rec.get("description") or rec.get("spec_name", "")
    tla_source = rec.get("tla_source", "")
    return {
        "_tier": tier,
        "_prompt_id": f"tlaplus_examples/{rec['spec_name']}",
        "_source": "tlaplus_examples_v1",
        "_timestamp": datetime.utcnow().isoformat(),
        "_features": rec.get("features", {}),
        "_spec_path": rec.get("spec_path", ""),
        "messages": [
            {"role": "developer", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": _user_turn(description)},
            {
                "role": "assistant",
                "channel": "analysis",
                "content": _analysis_for(description, tla_source),
            },
            {"role": "assistant", "channel": "final", "content": tla_source},
        ],
    }


def main() -> int:
    if not _LABELED.exists():
        print(f"[error] labeled corpus missing: {_LABELED}", file=sys.stderr)
        print(
            "        run the scraper first to emit tlaplus_examples_labeled.jsonl",
            file=sys.stderr,
        )
        return 1

    holdout = _load_holdout_modules()
    print(f"[info] loaded {len(holdout)} holdout module names to exclude")

    n_total = 0
    n_holdout_skipped = 0
    n_no_source = 0
    tlc_rows: list[dict] = []
    tlaps_rows: list[dict] = []
    tlc_specs: list[str] = []
    tlaps_specs: list[str] = []

    with _LABELED.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_total += 1

            spec_name = rec.get("spec_name", "")
            tla_source = rec.get("tla_source", "")
            features = rec.get("features", {}) or {}

            source_mods = set(_MOD_RE.findall(tla_source))
            source_mods.add(spec_name)
            if source_mods & holdout:
                n_holdout_skipped += 1
                continue

            if not tla_source.strip():
                n_no_source += 1
                continue

            if features.get("tlc_pass"):
                tlc_rows.append(_record(rec, "tlc_target"))
                tlc_specs.append(spec_name)
            if features.get("tlaps_pass"):
                tlaps_rows.append(_record(rec, "tlaps_target"))
                tlaps_specs.append(spec_name)

    with _OUT_TLC.open("w") as f:
        for r in tlc_rows:
            f.write(json.dumps(r) + "\n")
    with _OUT_TLAPS.open("w") as f:
        for r in tlaps_rows:
            f.write(json.dumps(r) + "\n")

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "source_rows": n_total,
        "holdout_skipped": n_holdout_skipped,
        "empty_source_skipped": n_no_source,
        "tlc_target": {"count": len(tlc_rows), "specs": sorted(tlc_specs)},
        "tlaps_target": {"count": len(tlaps_rows), "specs": sorted(tlaps_specs)},
    }
    with _SUMMARY.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] source rows:            {n_total}")
    print(f"[ok] holdout collisions:     {n_holdout_skipped}")
    print(f"[ok] empty sources skipped:  {n_no_source}")
    print(f"[ok] tlc_target_sft.jsonl:   {len(tlc_rows)} rows -> {_OUT_TLC}")
    print(f"[ok] tlaps_target_sft.jsonl: {len(tlaps_rows)} rows -> {_OUT_TLAPS}")
    print(f"[ok] summary:                {_SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
