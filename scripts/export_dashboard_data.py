#!/usr/bin/env python3
"""Export ChatTLA data to JSON files for the web dashboard."""

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR = Path(__file__).parent.parent.parent / "webpage" / "src" / "_extra" / "TLA-extraction" / "data"


def extract_module_name(tla_content: str) -> str:
    """Extract module name from TLA+ spec content."""
    match = re.search(r"MODULE\s+(\w+)", tla_content or "")
    return match.group(1) if match else "Unknown"


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def export_stats():
    """Compute overview statistics."""
    train = load_jsonl(DATA_DIR / "processed" / "train.jsonl")
    augmented = load_jsonl(DATA_DIR / "processed" / "augmented.jsonl")
    evalu = load_jsonl(DATA_DIR / "processed" / "eval.jsonl")
    validated = load_jsonl(DATA_DIR / "validated" / "combined.jsonl")
    rejected = load_jsonl(DATA_DIR / "rejected" / "rejected.jsonl")
    dpo = load_jsonl(DATA_DIR / "processed" / "rl" / "dpo_pairs.jsonl")

    # Tier distributions
    def tier_dist(items):
        dist = defaultdict(int)
        for item in items:
            dist[item.get("_tier", "untagged")] += 1
        return dict(dist)

    # Domain distribution from benchmarks
    with open(DATA_DIR / "benchmarks" / "benchmark_suite.json") as f:
        benchmarks = json.load(f)

    domain_dist = defaultdict(int)
    for b in benchmarks:
        domain_dist[b["domain"]] += 1

    return {
        "counts": {
            "training_examples": len(train),
            "augmented_specs": len(augmented),
            "eval_examples": len(evalu),
            "validated_specs": len(validated),
            "rejected_specs": len(rejected),
            "dpo_pairs": len(dpo),
            "benchmarks": len(benchmarks),
        },
        "train_tiers": tier_dist(train),
        "augmented_tiers": tier_dist(augmented),
        "benchmark_domains": dict(domain_dist),
    }


def export_benchmarks():
    """Export benchmark suite definitions."""
    with open(DATA_DIR / "benchmarks" / "benchmark_suite.json") as f:
        return json.load(f)


def export_benchmark_timeline():
    """Extract performance metrics over RL training cycles."""
    results_dir = OUTPUTS_DIR / "benchmark_results"
    timeline = []

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".csv"):
            continue

        cycle_match = re.search(r"_c(\d+)_", fname)
        date_match = re.search(r"(\d{8})_(\d{6})", fname)
        if not cycle_match:
            continue

        cycle = int(cycle_match.group(1))
        date = date_match.group(1) if date_match else None

        filepath = results_dir / fname
        with open(filepath) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        sany_pass = sum(1 for r in rows if r.get("sany_pass") == "1")
        tlc_pass = sum(1 for r in rows if r.get("tlc_pass") == "1")
        total = len(rows)
        avg_structural = sum(
            float(r.get("structural_score", 0)) for r in rows
        ) / total

        # Tier breakdown
        tier_counts = defaultdict(int)
        for r in rows:
            tier_counts[r.get("tlc_tier", "unknown")] += 1

        timeline.append(
            {
                "cycle": cycle,
                "date": date,
                "total": total,
                "sany_pass": sany_pass,
                "tlc_pass": tlc_pass,
                "sany_rate": round(sany_pass / total, 3),
                "tlc_rate": round(tlc_pass / total, 3),
                "avg_structural": round(avg_structural, 3),
                "tiers": dict(tier_counts),
            }
        )

    timeline.sort(key=lambda x: x["cycle"])
    return timeline


def export_benchmark_latest():
    """Export the latest/best benchmark results per benchmark."""
    results_dir = OUTPUTS_DIR / "benchmark_results"

    # Key named result sets
    named_results = {}
    for label, fname in [
        ("baseline", "benchmark_results_baseline.csv"),
        ("latest_sft", "benchmark_20b_sft_20260402_075610.csv"),
    ]:
        filepath = results_dir / fname
        if filepath.exists():
            with open(filepath) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            named_results[label] = [
                {
                    "benchmark_id": r["benchmark_id"],
                    "name": r["name"],
                    "domain": r["domain"],
                    "difficulty": int(r["difficulty"]),
                    "sany_pass": int(r["sany_pass"]),
                    "tlc_pass": int(r["tlc_pass"]),
                    "structural_score": round(float(r.get("structural_score", 0)), 3),
                    "tlc_tier": r.get("tlc_tier", ""),
                    "model": r.get("model", ""),
                }
                for r in rows
            ]

    return named_results


def export_training_data():
    """Export training data metadata (descriptions + tier, not full specs)."""
    train = load_jsonl(DATA_DIR / "processed" / "train.jsonl")

    items = []
    for i, entry in enumerate(train):
        msgs = entry.get("messages", [])
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]

        description = user_msgs[0]["content"] if user_msgs else ""
        # Extract just the description text (strip "Write a TLA+ specification for the following:" prefix)
        description = re.sub(
            r"^Write a TLA\+? specification for the following:\s*",
            "",
            description,
            flags=re.IGNORECASE,
        ).strip()

        # Get spec from last assistant message
        spec = assistant_msgs[-1]["content"] if assistant_msgs else ""
        module_name = extract_module_name(spec)

        items.append(
            {
                "index": i,
                "tier": entry.get("_tier", "untagged"),
                "module_name": module_name,
                "description": description[:500],
                "spec_length": len(spec),
                "spec_preview": spec[:300],
            }
        )

    return items


def export_validated_specs():
    """Export validated specifications with full TLA+ content."""
    validated = load_jsonl(DATA_DIR / "validated" / "combined.jsonl")

    items = []
    for entry in validated:
        tla = entry.get("tla_content", "")
        annotation = entry.get("annotation", {}) or {}

        items.append(
            {
                "id": entry.get("id", "")[:12],
                "source": entry.get("source", ""),
                "license": entry.get("license", ""),
                "module_name": extract_module_name(tla),
                "description": (
                    annotation.get("natural_language_description", "")[:500]
                ),
                "tla_content": tla,
                "cfg_content": entry.get("cfg_content", ""),
                "metadata": entry.get("metadata", {}),
            }
        )

    return items


def export_modules():
    """Export module descriptions from tla_descriptions.json."""
    with open(DATA_DIR / "derived" / "tla_descriptions.json") as f:
        descriptions = json.load(f)

    items = []
    for entry in descriptions:
        desc = entry.get("description", {})
        technical = desc.get("technical", {})

        items.append(
            {
                "id": entry.get("id", ""),
                "module_name": entry.get("module_name", ""),
                "narrative": desc.get("narrative", "")[:500],
                "algorithm": technical.get("algorithm", ""),
                "variables": [
                    v.get("name", "") for v in technical.get("variables", [])
                ],
                "actions": [
                    a.get("name", "") for a in technical.get("actions", [])
                ],
            }
        )

    return items


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Exporting stats...")
    stats = export_stats()

    print("Exporting benchmarks...")
    benchmarks = export_benchmarks()

    print("Exporting benchmark timeline...")
    timeline = export_benchmark_timeline()

    print("Exporting benchmark latest results...")
    benchmark_latest = export_benchmark_latest()

    print("Exporting training data metadata...")
    training = export_training_data()

    print("Exporting validated specifications...")
    specs = export_validated_specs()

    print("Exporting module descriptions...")
    modules = export_modules()

    # Bundle everything into one JSON file for simplicity
    dashboard_data = {
        "stats": stats,
        "benchmarks": benchmarks,
        "benchmark_timeline": timeline,
        "benchmark_latest": benchmark_latest,
        "training_data": training,
        "validated_specs": specs,
        "modules": modules,
        "generated_at": "2026-04-02",
    }

    outpath = OUT_DIR / "dashboard_data.json"
    with open(outpath, "w") as f:
        json.dump(dashboard_data, f, separators=(",", ":"))

    size_kb = outpath.stat().st_size / 1024
    print(f"Wrote {outpath} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
