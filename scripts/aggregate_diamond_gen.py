#!/usr/bin/env python3
"""aggregate_diamond_gen.py — re-validate every generated diamond spec in parallel,
then write the final outputs/diamond_gen/diamond_generated.jsonl plus a summary.

Why re-validate
---------------
The 10 generation subagents each ran the validator themselves and reported their
own is_diamond flags. We don't take their word for it — we re-run
scripts.diamond_sft_gen.validate_diamond on every spec from a clean process and
compute the authoritative pass/fail tally.
"""
from __future__ import annotations
import json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

GEN_DIR = _REPO_ROOT / "outputs" / "diamond_gen"
OUT_PATH = GEN_DIR / "diamond_generated.jsonl"
SUMMARY_PATH = GEN_DIR / "diamond_generated_summary.json"


def revalidate(record: dict) -> dict:
    """Re-run the Diamond validator on a record's spec text from a clean process."""
    from scripts.diamond_sft_gen import validate_diamond
    spec = record["spec"]
    r = validate_diamond(spec, prompt_id=record.get("module", ""))
    return {
        "module": record.get("module"),
        "batch": record.get("_batch"),
        "topic_desc": record.get("topic_desc"),
        "spec": spec,
        "agent_is_diamond": bool(record.get("is_diamond")),
        "is_diamond": bool(r.is_diamond),
        "tier": r.tlc_tier,
        "sany_pass": r.sany_pass,
        "distinct_states": r.distinct_states,
        "invariants_checked": r.invariants_checked,
        "mutation_caught": r.mutation_caught,
        "trivial_invariant": r.trivial_invariant,
        "error": r.error,
    }


def main() -> int:
    batch_files = sorted(GEN_DIR.glob("*.jsonl"))
    batch_files = [p for p in batch_files if p.name != OUT_PATH.name]

    records: list[dict] = []
    for bf in batch_files:
        batch_id = bf.stem
        with bf.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_batch"] = batch_id
                records.append(rec)

    print(f"[aggregate] {len(records)} records across {len(batch_files)} batches")
    print(f"[aggregate] re-validating in parallel ...")

    t0 = time.monotonic()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(revalidate, r): r for r in records}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                res = fut.result()
            except Exception as e:
                src = futs[fut]
                res = {
                    "module": src.get("module"),
                    "batch": src.get("_batch"),
                    "spec": src.get("spec", ""),
                    "is_diamond": False,
                    "tier": "error",
                    "error": f"validator_crash: {e}",
                }
            results.append(res)
            mark = "D" if res.get("is_diamond") else "."
            sys.stdout.write(mark)
            if i % 50 == 0:
                sys.stdout.write(f" {i}\n")
            sys.stdout.flush()
    sys.stdout.write("\n")
    elapsed = time.monotonic() - t0

    n_diamond = sum(1 for r in results if r["is_diamond"])
    by_batch: dict[str, dict[str, int]] = {}
    by_tier: dict[str, int] = {}
    for r in results:
        b = r.get("batch", "?")
        d = by_batch.setdefault(b, {"total": 0, "diamond": 0})
        d["total"] += 1
        d["diamond"] += int(r["is_diamond"])
        by_tier[r.get("tier", "?")] = by_tier.get(r.get("tier", "?"), 0) + 1

    OUT_PATH.write_text("".join(json.dumps(r) + "\n" for r in results))

    summary = {
        "total": len(results),
        "diamond": n_diamond,
        "by_batch": by_batch,
        "by_tier": by_tier,
        "elapsed_seconds": round(elapsed, 1),
        "out_path": str(OUT_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(f"\n[aggregate] re-validation complete in {elapsed:.1f}s")
    print(f"[aggregate] DIAMOND: {n_diamond}/{len(results)}")
    print(f"[aggregate] by tier: {by_tier}")
    print(f"[aggregate] by batch:")
    for b, d in sorted(by_batch.items()):
        print(f"  {b:30s} {d['diamond']:3d}/{d['total']:3d}")
    print(f"[aggregate] wrote {OUT_PATH}")
    print(f"[aggregate] summary {SUMMARY_PATH}")
    return 0 if n_diamond == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
