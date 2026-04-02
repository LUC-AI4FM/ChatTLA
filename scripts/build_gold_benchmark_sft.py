#!/usr/bin/env python3
"""
Build SFT training data from TLC-verified gold benchmark specs.

Reads all benchmark CSV results, extracts specs where sany_pass=1 AND tlc_pass=1,
deduplicates by spec content, and pairs them with the benchmark problem descriptions
to create high-quality training examples.

These are the model's own best outputs — verified by both SANY and TLC — so they
represent the strongest training signal we have for specification correctness.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_BENCHMARK_DIR = _REPO / "outputs" / "benchmark_results"
_BENCHMARK_SUITE = _REPO / "data" / "benchmarks" / "benchmark_suite.json"
_OUT = _REPO / "data" / "processed" / "gold_benchmark_sft.jsonl"

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


def main() -> int:
    # Load benchmark problem descriptions
    if not _BENCHMARK_SUITE.exists():
        print(f"[gold_benchmark_sft] ERROR: {_BENCHMARK_SUITE} not found")
        return 1

    benchmarks = json.loads(_BENCHMARK_SUITE.read_text())
    bm_by_id = {bm["id"]: bm for bm in benchmarks}

    # Extract all gold specs from CSV results
    gold_specs: list[dict] = []
    for csv_file in sorted(_BENCHMARK_DIR.glob("*.csv")):
        try:
            with open(csv_file) as f:
                for row in csv.DictReader(f):
                    if row.get("sany_pass") == "1" and row.get("tlc_pass") == "1":
                        spec = row.get("generated_spec", "").strip()
                        bid = row.get("benchmark_id", "")
                        if spec and len(spec) > 50 and bid in bm_by_id:
                            gold_specs.append({"benchmark_id": bid, "spec": spec})
        except (OSError, csv.Error):
            continue

    # Deduplicate: keep best (longest) spec per benchmark problem per content hash
    # Group by benchmark_id, then deduplicate within each group
    by_problem: dict[str, list[dict]] = {}
    for g in gold_specs:
        by_problem.setdefault(g["benchmark_id"], []).append(g)

    unique_examples: list[dict] = []
    for bid, specs in by_problem.items():
        seen_hashes = set()
        # Sort by length descending — prefer more complete specs
        specs.sort(key=lambda x: len(x["spec"]), reverse=True)
        for s in specs:
            h = hashlib.sha256(s["spec"][:500].encode()).hexdigest()[:16]
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_examples.append(s)
                # Keep up to 3 diverse gold solutions per problem
                if len(seen_hashes) >= 3:
                    break

    # Build training examples in harmony format
    examples = []
    for ex in unique_examples:
        bm = bm_by_id[ex["benchmark_id"]]
        user_content = f"Write a TLA+ specification for the following:\n\n{bm['description']}"
        if bm.get("hints"):
            user_content += f"\n\nHints: {bm['hints']}"

        # Verify the spec has core structure
        spec = ex["spec"]
        if not (re.search(r'\bInit\b', spec) and re.search(r'\bNext\b', spec)
                and re.search(r'\bVARIABLE', spec)):
            continue

        examples.append({
            "_tier": "gold_benchmark",
            "_source": "benchmark_csv",
            "_benchmark_id": ex["benchmark_id"],
            "messages": [
                {"role": "developer", "content": _DEVELOPER_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": "I'll write a verified TLA+ specification."},
                {"role": "assistant", "content": spec},
            ],
        })

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    problems_covered = len(set(ex["_benchmark_id"] for ex in examples))
    print(f"[gold_benchmark_sft] {len(examples)} examples from {problems_covered} problems -> {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
