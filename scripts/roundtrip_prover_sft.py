"""
roundtrip_prover_sft.py — Verify per-theorem chunks by reconstructing a
synthetic .tla module and running tlapm on it.

Strategy
--------
For each chunk in data/processed/prover_chunks.jsonl:
  1. Build synthetic = preamble + statement + proof + "===="
  2. Rename the MODULE to a unique name (so it doesn't collide with the
     original file in the same directory).
  3. Write the synthetic .tla into the *same directory* as the original
     (so EXTENDS of sibling modules resolves correctly), then run tlapm.
  4. Keep chunks that hit tier in {proved, partial} with proved > 0.
  5. Drop the synthetic file afterward.

Output
------
data/processed/prover_chunks_verified.jsonl  — chunks that round-tripped
data/processed/prover_chunks_failed.jsonl    — chunks that didn't (with reason)

Usage
-----
    python scripts/roundtrip_prover_sft.py            # all chunks
    python scripts/roundtrip_prover_sft.py --limit 5  # smoke test
    python scripts/roundtrip_prover_sft.py --filter AddTwo
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.validators.tlaps_validator import validate_file  # noqa: E402

CHUNKS_IN = REPO / "data" / "processed" / "prover_chunks.jsonl"
OUT_VERIFIED = REPO / "data" / "processed" / "prover_chunks_verified.jsonl"
OUT_FAILED = REPO / "data" / "processed" / "prover_chunks_failed.jsonl"

MODULE_RE = re.compile(r"MODULE\s+(\w+)")


def synthesize(chunk: dict) -> tuple[str, str]:
    """Return (synthetic_tla_text, new_module_name)."""
    body = "\n".join([chunk["preamble"], chunk["statement"], chunk["proof"]])
    m = MODULE_RE.search(body)
    if not m:
        raise ValueError("no MODULE in preamble")
    orig_name = m.group(1)
    new_name = f"RT_{orig_name}_{chunk['theorem_line']}"
    body = MODULE_RE.sub(f"MODULE {new_name}", body, count=1)
    if not body.rstrip().endswith("===="):
        body = body.rstrip() + "\n" + ("=" * 78) + "\n"
    return body, new_name


def run_one(chunk: dict, timeout: int = 60) -> dict:
    src = REPO / chunk["source_file"]
    parent = src.parent
    try:
        synth_text, new_name = synthesize(chunk)
    except ValueError as e:
        return {"ok": False, "reason": f"synthesize: {e}"}

    tmp = parent / f"{new_name}.tla"
    tmp.write_text(synth_text, encoding="utf-8")
    try:
        result = validate_file(tmp, timeout=timeout)
    except Exception as e:
        return {"ok": False, "reason": f"validator-exception: {e}"}
    finally:
        tmp.unlink(missing_ok=True)

    return {
        "ok": result.tier in ("proved", "partial") and result.obligations_proved > 0,
        "tier": result.tier,
        "proved": result.obligations_proved,
        "total": result.obligations_total,
        "failed": result.obligations_failed,
        "seconds": round(result.runtime_seconds, 2),
        "errors": result.errors[:3],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Stop after N chunks")
    ap.add_argument("--filter", default=None, help="Only chunks whose source_file contains this substring")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    chunks = [json.loads(l) for l in CHUNKS_IN.open()]
    if args.filter:
        chunks = [c for c in chunks if args.filter in c["source_file"]]
    if args.limit:
        chunks = chunks[: args.limit]

    print(f"[roundtrip] {len(chunks)} chunks to verify")
    verified: list[dict] = []
    failed: list[dict] = []
    t0 = time.monotonic()

    for i, ch in enumerate(chunks, 1):
        tag = f"{Path(ch['source_file']).name}:L{ch['theorem_line']}"
        res = run_one(ch, timeout=args.timeout)
        if res["ok"]:
            print(f"  [{i:3d}] OK    {res['proved']:3d}/{res.get('total','?'):3d}  {res['seconds']:5.1f}s  {tag}")
            verified.append({**ch, "_roundtrip": res})
        else:
            reason = res.get("reason") or f"tier={res.get('tier')} proved={res.get('proved')}/{res.get('total')}"
            print(f"  [{i:3d}] FAIL  {reason}  {tag}")
            failed.append({**ch, "_roundtrip": res})

    elapsed = time.monotonic() - t0
    print(f"\n[roundtrip] {len(verified)}/{len(chunks)} verified in {elapsed:.1f}s")
    if verified:
        proved_sum = sum(v["_roundtrip"]["proved"] for v in verified)
        total_sum = sum(v["_roundtrip"].get("total", 0) for v in verified)
        print(f"           {proved_sum}/{total_sum} obligations across verified chunks")

    OUT_VERIFIED.write_text("\n".join(json.dumps(v) for v in verified) + ("\n" if verified else ""))
    OUT_FAILED.write_text("\n".join(json.dumps(v) for v in failed) + ("\n" if failed else ""))
    print(f"[roundtrip] wrote {OUT_VERIFIED} ({len(verified)})")
    print(f"[roundtrip] wrote {OUT_FAILED} ({len(failed)})")


if __name__ == "__main__":
    main()
