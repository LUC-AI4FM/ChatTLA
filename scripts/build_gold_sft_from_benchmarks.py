#!/usr/bin/env python3
"""
Extract ALL unique gold (SANY+TLC verified) specs from benchmark CSVs
and build a comprehensive SFT training dataset.

Scans both outputs/benchmark_results/*.csv and the RL-loop/ subdirectory.
Deduplicates by full spec content hash. Pairs each gold spec with the
corresponding benchmark problem description from benchmark_suite.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_BENCHMARK_DIRS = [
    _REPO / "outputs" / "benchmark_results",
    _REPO / "outputs" / "benchmark_results" / "RL-loop",
]
_BENCHMARK_SUITE = _REPO / "data" / "benchmarks" / "benchmark_suite.json"
_OUT = _REPO / "data" / "processed" / "gold_all_benchmarks_sft.jsonl"

_DEVELOPER_PROMPT = """\
You are ChatTLA, an expert at writing verified TLA+ formal specifications.
When asked to write a TLA+ spec, follow these rules exactly:
1. Start the module with ---- MODULE <ModuleName> ----
2. End with ====
3. Include EXTENDS, VARIABLES, Init, Next, and Spec operators
4. After the TLA+ module, append a TLC configuration block:
   SPECIFICATION Spec
   INVARIANT TypeOK   (if TypeOK is defined)
5. Output only valid TLA+ code. No markdown fences, no explanation outside the spec.

Critical TLA+ syntax rules:
- EXTENDS Integers for Int, +, -, *, \\div; EXTENDS Sequences for Seq, Append, Len, Head, Tail; EXTENDS FiniteSets for Cardinality, IsFiniteSet
- Declare ALL state variables in a VARIABLES line (every primed variable x' must appear in VARIABLES)
- Use = (not ==) inside Init and Next action conjuncts: /\\ x = value
- Function construction: [x \\in S |-> expr] (NOT [x \\in S : expr])
- Use \\in SUBSET S for set quantification (NOT \\E x \\subseteq S)
- Do NOT use PlusCal syntax (:=, --algorithm, labels, while, goto)
- TypeOK must be defined if referenced as INVARIANT
- Spec == Init /\\ [][Next]_vars where vars == <<v1, v2, ...>>\
"""


def _spec_hash(spec: str) -> str:
    """Full content hash for dedup."""
    # Normalize whitespace for hash to avoid false duplicates from trailing spaces
    normalized = re.sub(r"[ \t]+\n", "\n", spec.strip())
    return hashlib.sha256(normalized.encode()).hexdigest()[:20]


def _has_core_structure(spec: str) -> bool:
    """Check spec has Init, Next, VARIABLE(S)."""
    return bool(
        re.search(r"\bInit\b", spec)
        and re.search(r"\bNext\b", spec)
        and re.search(r"\bVARIABLE", spec)
    )


def main() -> int:
    if not _BENCHMARK_SUITE.exists():
        print(f"ERROR: {_BENCHMARK_SUITE} not found")
        return 1

    benchmarks = json.loads(_BENCHMARK_SUITE.read_text())
    bm_by_id = {bm["id"]: bm for bm in benchmarks}

    # Collect all gold specs from all CSV files
    raw_golds: list[dict] = []
    files_scanned = 0
    for bdir in _BENCHMARK_DIRS:
        if not bdir.exists():
            continue
        for csv_file in sorted(bdir.glob("*.csv")):
            files_scanned += 1
            try:
                with open(csv_file) as f:
                    for row in csv.DictReader(f):
                        if row.get("sany_pass") == "1" and row.get("tlc_pass") == "1":
                            spec = row.get("generated_spec", "").strip()
                            bid = row.get("benchmark_id", "")
                            if spec and len(spec) > 50 and bid in bm_by_id:
                                raw_golds.append({
                                    "benchmark_id": bid,
                                    "spec": spec,
                                    "structural_score": float(row.get("structural_score", 0)),
                                    "source": csv_file.name,
                                })
            except (OSError, csv.Error):
                continue

    print(f"Scanned {files_scanned} CSV files, found {len(raw_golds)} raw gold specs")

    # Deduplicate by full spec content hash, keeping best per hash
    by_hash: dict[str, dict] = {}
    for g in raw_golds:
        h = _spec_hash(g["spec"])
        existing = by_hash.get(h)
        # Keep the one with highest structural score, then longest spec
        if existing is None or (
            g["structural_score"] > existing["structural_score"]
            or (g["structural_score"] == existing["structural_score"]
                and len(g["spec"]) > len(existing["spec"]))
        ):
            by_hash[h] = g

    unique_golds = list(by_hash.values())
    print(f"After dedup: {len(unique_golds)} unique gold specs")

    # Filter for core structure
    valid_golds = [g for g in unique_golds if _has_core_structure(g["spec"])]
    print(f"After structure check: {len(valid_golds)} valid specs")

    # Group by benchmark_id for stats
    by_problem: dict[str, list[dict]] = {}
    for g in valid_golds:
        by_problem.setdefault(g["benchmark_id"], []).append(g)

    # Sort within each problem: prefer higher structural_score, then longer spec
    for specs in by_problem.values():
        specs.sort(key=lambda x: (x["structural_score"], len(x["spec"])), reverse=True)

    # Build SFT examples
    examples = []
    for bid in sorted(by_problem):
        bm = bm_by_id[bid]
        user_content = f"Write a TLA+ specification for the following:\n\n{bm['description']}"
        if bm.get("hints"):
            user_content += f"\n\nHints: {bm['hints']}"

        for i, g in enumerate(by_problem[bid]):
            examples.append({
                "_tier": "gold_benchmark",
                "_source": "benchmark_csv_harvest",
                "_benchmark_id": bid,
                "_variant": i,
                "messages": [
                    {"role": "developer", "content": _DEVELOPER_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": "I'll write a verified TLA+ specification."},
                    {"role": "assistant", "content": g["spec"]},
                ],
            })

    # Write output
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Print summary
    problems_covered = sorted(by_problem.keys())
    missing = sorted(set(bm_by_id.keys()) - set(problems_covered))
    print(f"\n{'='*60}")
    print(f"Output: {_OUT}")
    print(f"Total examples: {len(examples)}")
    print(f"Problems covered: {len(problems_covered)}/20")
    print(f"Missing problems: {', '.join(missing) if missing else 'none'}")
    print(f"\nPer-problem breakdown:")
    for bid in problems_covered:
        name = bm_by_id[bid]["name"]
        cnt = len(by_problem[bid])
        lens = [len(g["spec"]) for g in by_problem[bid]]
        print(f"  {bid} ({name}): {cnt} specs, {min(lens)}-{max(lens)} chars")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
