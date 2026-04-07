#!/usr/bin/env python3
"""
audit_fm_alpaca.py — Audit fm-alpaca's TLA+ subset against the Diamond gate.

Background:
    fm-alpaca (Cao et al. 2024, https://huggingface.co/datasets/fm-universe/FM-alpaca)
    is a multi-task SFT dataset with 14,372 formal-methods examples across
    Coq, Lean, TLA+, ACSL, and Dafny. The TLA+ subset has 533 examples
    decomposed into 5 task types: ProofGen, ProofComplete, ProofInfill,
    SegGen, ReqAna.

Hypothesis:
    Like the project_diamond_pipeline_20260405 finding (0/484 of an existing
    TLA+ corpus passes the Diamond semantic gate), fm-alpaca's TLA+ outputs
    may also be syntactically plausible but not actually verifiable.

Method:
    For each TLA+ ProofGen example, check:
      1. Self-contained:  has both `MODULE X` header and `====` footer
      2. SANY parses:     standalone module passes the SANY parser
      3. TLC verifies:    TLC runs to completion with no violations
      4. Diamond passes:  full semantic gate (mutation test, etc.)

Usage:
    python scripts/audit_fm_alpaca.py
    python scripts/audit_fm_alpaca.py --download  # download the dataset first
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_FM_ALPACA_URL = "https://huggingface.co/datasets/fm-universe/FM-alpaca/resolve/main/fm-alpaca-train.jsonl"
_DEFAULT_PATH = _REPO / "data" / "external" / "fm-alpaca-train.jsonl"


def download_fm_alpaca(dest: Path) -> None:
    """Download fm-alpaca-train.jsonl from HuggingFace."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[fm_alpaca] Already downloaded: {dest}")
        return
    print(f"[fm_alpaca] Downloading from {_FM_ALPACA_URL} ...")
    subprocess.run(
        ["curl", "-sL", "-o", str(dest), _FM_ALPACA_URL],
        check=True, timeout=300,
    )
    print(f"[fm_alpaca] Saved to {dest}")


def load_tla_examples(path: Path) -> list[dict]:
    """Load all TLA+ examples from fm-alpaca, grouped by task."""
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("lang") == "TLA":
                out.append(d)
    return out


def audit_proofgen(examples: list[dict]) -> dict:
    """Run audit on ProofGen examples (the 'full spec generation' task)."""
    from src.validators.sany_validator import validate_string as sany_validate
    from src.validators.tlc_validator import validate_string as tlc_validate

    proofgen = [e for e in examples if e.get("task") == "ProofGen"]
    print(f"\n[audit] TLA+ ProofGen examples: {len(proofgen)}")

    results = {
        "total": len(proofgen),
        "self_contained": 0,
        "sany_pass": 0,
        "tlc_gold": 0,
        "diamond": 0,
        "details": [],
    }

    for i, ex in enumerate(proofgen):
        spec_text = ex.get("output", "")
        # Strip code fences if present
        spec_text = re.sub(r"```\w*\n?", "", spec_text)
        spec_text = re.sub(r"```", "", spec_text).strip()

        is_self_contained = bool(
            re.search(r"----\s*MODULE\s+\w+", spec_text) and "====" in spec_text
        )
        sany_ok = False
        tlc_tier = "n/a"
        diamond_ok = False

        if is_self_contained:
            results["self_contained"] += 1
            mod_match = re.search(r"MODULE\s+(\w+)", spec_text)
            mod_name = mod_match.group(1) if mod_match else "FmAlpacaTest"
            try:
                sany_result = sany_validate(spec_text, module_name=mod_name)
                if sany_result.valid:
                    sany_ok = True
                    results["sany_pass"] += 1
                    try:
                        tlc_result = tlc_validate(spec_text, module_name=mod_name, timeout=30)
                        tlc_tier = tlc_result.tier
                        if tlc_tier == "gold":
                            results["tlc_gold"] += 1
                            if tlc_result.is_diamond:
                                results["diamond"] += 1
                                diamond_ok = True
                    except Exception:
                        pass
            except Exception:
                pass

        results["details"].append({
            "uid": ex.get("uid", "?")[:8],
            "self_contained": is_self_contained,
            "sany_ok": sany_ok,
            "tlc_tier": tlc_tier,
            "diamond": diamond_ok,
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(proofgen)}...")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(_DEFAULT_PATH),
                        help="Path to fm-alpaca-train.jsonl")
    parser.add_argument("--download", action="store_true",
                        help="Download fm-alpaca first if not present")
    parser.add_argument("--output", default=str(_REPO / "outputs" / "audit_fm_alpaca.json"))
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.download or not input_path.exists():
        # Try /tmp first if it's already there
        tmp_path = Path("/tmp/fm-alpaca.jsonl")
        if tmp_path.exists() and not input_path.exists():
            input_path = tmp_path
        else:
            download_fm_alpaca(input_path)

    print(f"[audit] Loading {input_path} ...")
    examples = load_tla_examples(input_path)
    print(f"[audit] Total TLA+ examples: {len(examples)}")

    # Task breakdown
    from collections import Counter
    by_task = Counter(e.get("task", "?") for e in examples)
    print("\n[audit] TLA+ task breakdown:")
    for task, count in by_task.most_common():
        print(f"  {task:20s} {count}")

    # Run the actual audit on ProofGen
    results = audit_proofgen(examples)

    print("\n" + "=" * 60)
    print("[audit] FM-ALPACA TLA+ ProofGen RESULTS")
    print("=" * 60)
    print(f"  Total examples:           {results['total']}")
    print(f"  Self-contained (MODULE+====): {results['self_contained']:3d} / {results['total']} ({results['self_contained'] / max(results['total'], 1):.0%})")
    print(f"  SANY parses:                  {results['sany_pass']:3d} / {results['total']} ({results['sany_pass'] / max(results['total'], 1):.0%})")
    print(f"  TLC gold tier:                {results['tlc_gold']:3d} / {results['total']} ({results['tlc_gold'] / max(results['total'], 1):.0%})")
    print(f"  Diamond gate (semantic):      {results['diamond']:3d} / {results['total']} ({results['diamond'] / max(results['total'], 1):.0%})")
    print()

    # Save full results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "fm-universe/FM-alpaca",
        "subset": "TLA",
        "task": "ProofGen",
        "task_breakdown": dict(by_task),
        "results": {k: v for k, v in results.items() if k != "details"},
        "details": results["details"],
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[audit] Full results: {out_path}")


if __name__ == "__main__":
    main()
