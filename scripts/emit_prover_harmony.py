"""
emit_prover_harmony.py — Convert verified prover chunks into harmony-format
JSONL train/eval splits ready for SFTTrainer.

Input:  data/processed/prover_chunks_verified.jsonl
Output: data/processed/prover_train.jsonl
        data/processed/prover_eval.jsonl

Format matches the existing spec-gen dataset (developer/user/assistant with
channel field), so train.py can consume it via the same code path with only
a path change.

Split: 90/10 train/eval, deterministic by seed=42, stratified by source file
so the eval set isn't dominated by CRDT_proof.tla (15 chunks).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IN = REPO / "data" / "processed" / "prover_chunks_verified.jsonl"
OUT_TRAIN = REPO / "data" / "processed" / "prover_train.jsonl"
OUT_EVAL = REPO / "data" / "processed" / "prover_eval.jsonl"

DEVELOPER_PROMPT = """You are ChatTLA-Prover, an expert at writing TLAPS proofs for TLA+ theorems.
You will be given a TLA+ module containing definitions, an Init/Next/Spec, and a THEOREM (or LEMMA) statement at the end.
Your job is to write the PROOF body for that final theorem. Output only the proof — proof steps and the final QED.
Use TLAPS syntax: hierarchical step bullets like <1>a, <1>b, <1> QED, the keywords BY, BY DEF, OBVIOUS, USE, DEFINE, SUFFICES, ASSUME, PROVE, CASE, PICK, WITNESS, and the backends PTL, Zenon, SMT.
Reference earlier theorems and lemmas by name when needed.
Do not repeat the THEOREM/LEMMA line. Do not output the closing ====.
Reasoning: medium"""


def to_messages(chunk: dict) -> list[dict]:
    user_content = (
        f"Write the TLAPS proof for the final theorem in the following module.\n\n"
        f"```tla\n{chunk['preamble']}\n{chunk['statement']}\n```"
    )
    return [
        {"role": "developer", "content": DEVELOPER_PROMPT},
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "channel": "analysis",
            "content": "I'll write a TLAPS proof using induction over Spec, breaking the inductive case into Init, stuttering, and Next.",
        },
        {
            "role": "assistant",
            "channel": "final",
            "content": chunk["proof"],
        },
    ]


def stratified_split(rows: list[dict], eval_frac: float, seed: int) -> tuple[list, list]:
    """Group by source file, take ~eval_frac of each group's chunks for eval."""
    rng = random.Random(seed)
    by_src: dict[str, list] = defaultdict(list)
    for r in rows:
        by_src[r["source_file"]].append(r)

    train: list[dict] = []
    evalset: list[dict] = []
    for src, items in sorted(by_src.items()):
        rng.shuffle(items)
        n_eval = max(1, int(round(len(items) * eval_frac))) if len(items) >= 4 else 0
        evalset.extend(items[:n_eval])
        train.extend(items[n_eval:])
    rng.shuffle(train)
    rng.shuffle(evalset)
    return train, evalset


def main() -> None:
    rows = [json.loads(l) for l in IN.open()]
    print(f"[emit] {len(rows)} verified chunks")

    train_chunks, eval_chunks = stratified_split(rows, eval_frac=0.10, seed=42)
    print(f"[emit] split: {len(train_chunks)} train / {len(eval_chunks)} eval")

    def write(path: Path, chunks: list[dict]) -> None:
        with path.open("w") as f:
            for ch in chunks:
                row = {
                    "_source": "tlaps_formallm",
                    "_source_file": ch["source_file"],
                    "_theorem_line": ch["theorem_line"],
                    "_obligations_proved": ch["_roundtrip"]["proved"],
                    "_obligations_total": ch["_roundtrip"].get("total", 0),
                    "messages": to_messages(ch),
                }
                f.write(json.dumps(row) + "\n")
        print(f"[emit] wrote {path} ({len(chunks)} rows)")

    write(OUT_TRAIN, train_chunks)
    write(OUT_EVAL, eval_chunks)

    # Length stats so we can pick a sane max_length
    import statistics
    user_lens = [len(json.dumps(to_messages(c)[1]["content"])) for c in rows]
    asst_lens = [len(c["proof"]) for c in rows]
    print(f"[emit] user content chars  p50={statistics.median(user_lens):.0f} p90={sorted(user_lens)[int(0.9*len(user_lens))]} max={max(user_lens)}")
    print(f"[emit] proof  content chars p50={statistics.median(asst_lens):.0f} p90={sorted(asst_lens)[int(0.9*len(asst_lens))]} max={max(asst_lens)}")


if __name__ == "__main__":
    main()
