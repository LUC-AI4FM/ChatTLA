"""
build_prover_sft.py — Extract per-theorem (preamble, statement, proof) chunks
from the deduped FormaLLM TLAPS corpus.

Output: data/processed/prover_chunks.jsonl  (raw chunks, pre-roundtrip)
        Each row: {preamble, statement, proof, source_file, theorem_line}

The next stage (roundtrip_prover_sft.py) takes this file, runs each chunk
through tlapm in a synthetic module, drops the ones that don't verify, and
emits prover_train.jsonl / prover_eval.jsonl in harmony format.

Chunking strategy
-----------------
- Top-level THEOREM/LEMMA/COROLLARY at column 0 marks a chunk start.
- The chunk runs until the next top-level THEOREM/LEMMA/COROLLARY or the
  closing ==== of the module.
- Within a chunk, statement lines run from THEOREM up to the first line
  matching ^\\s*(PROOF|<\\d+>|BY|OBVIOUS); the rest is the proof body.
- Preamble = the entire file *up to* the THEOREM line. We deliberately keep
  prior theorem proof bodies in the preamble — they cost a few tokens but
  give the model the "house style" and lemma references it needs.
- One-line proofs (THEOREM ... BY ...) are skipped in this pass — they need
  a different splitter and add little signal.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN_FILES = [
    REPO / "outputs" / "tlaps_formallm_scan.json",
    REPO / "outputs" / "tlaps_recovery_scan.json",
]
OUT = REPO / "data" / "processed" / "prover_chunks.jsonl"

THEOREM_RE = re.compile(r"^(THEOREM|LEMMA|COROLLARY)\b")
PROOF_START_RE = re.compile(r"^\s*(PROOF\b|<\d+>|BY\b|OBVIOUS\b)")
END_MODULE_RE = re.compile(r"^={4,}")

# A proof ends when we hit an *interstitial definition* at column 0:
#   - `Name ==`           (operator def)
#   - `Name(args) ==`     (operator def with params)
#   - `vars == ...`       (state-tuple def)
# We do NOT try to whitelist "proof-like lines" because real TLAPS proofs can
# contain arbitrary continuation lines like `[Next]_vars` (multi-line ASSUME
# bodies), `OBVIOUS`, `WITNESS`, `PICK`, `SUFFICES`, etc. Permissive-stop is
# safer: assume everything is part of the proof until we see a clear def.
DEF_LINE_RE = re.compile(r"^[A-Za-z_]\w*\s*(\([^)]*\))?\s*==")
END_MODULE_LINE_RE = re.compile(r"^={4,}")
COMMENT_LINE_RE = re.compile(r"^(\(\*|\\\*)")
SECTION_SEP_RE = re.compile(r"^-{4,}")

RANK = {"proved": 4, "all_proved": 3, "partial": 2, "unproved": 1, "no_theorems": 0, "parse_error": -1}


def select_files() -> list[Path]:
    """Return the deduped set of useful TLAPS source files."""
    best: dict[str, dict] = {}
    for sf in SCAN_FILES:
        for x in json.loads(sf.read_text()):
            key = x["file"].replace("_clean.tla", ".tla")
            if key not in best or RANK.get(x["tier"], -9) > RANK.get(best[key]["tier"], -9):
                best[key] = x
    kept = [x for x in best.values() if x["tier"] in ("proved", "partial") and x["total"] > 0]
    return [REPO / x["file"] for x in kept]


def find_theorem_starts(lines: list[str]) -> list[int]:
    return [i for i, ln in enumerate(lines) if THEOREM_RE.match(ln)]


def find_block_end(lines: list[str], start: int) -> int:
    """End of theorem block: index of next THEOREM/LEMMA/COROLLARY or closing ===="""
    for i in range(start + 1, len(lines)):
        if THEOREM_RE.match(lines[i]) or END_MODULE_RE.match(lines[i]):
            return i
    return len(lines)


def split_statement_proof(block: list[str]) -> tuple[list[str], list[str]]:
    """Within a theorem block, split into (statement_lines, proof_lines).

    The first line is always the THEOREM line itself. Walk forward until we
    see a line starting with PROOF, <n>, BY, or OBVIOUS — that's where the
    proof starts. Then truncate the proof at the first column-0 line that is
    not itself part of a proof (interstitial defs / blank-then-def).
    """
    proof_start = None
    for i in range(1, len(block)):
        if PROOF_START_RE.match(block[i]):
            proof_start = i
            break
    if proof_start is None:
        return block, []  # one-liner; caller will skip

    stmt_lines = block[:proof_start]
    # Walk proof lines. Stop at:
    #   - interstitial def at column 0  (Name == ...)
    #   - end-of-module ====
    #   - blank line(s) followed by a comment block ((* or \*) or section
    #     separator (-----) — these mark the end of one-line BY proofs
    #     and section boundaries between theorems
    # Blank lines INSIDE multi-step proofs (followed by another <n> bullet
    # or proof keyword) are kept.
    proof_end = len(block)
    j = proof_start
    while j < len(block):
        ln = block[j]
        if DEF_LINE_RE.match(ln) or END_MODULE_LINE_RE.match(ln):
            proof_end = j
            break
        if not ln.strip():
            # Look ahead past blank lines
            k = j + 1
            while k < len(block) and not block[k].strip():
                k += 1
            if k >= len(block):
                proof_end = j
                break
            nxt = block[k]
            if (
                COMMENT_LINE_RE.match(nxt)
                or SECTION_SEP_RE.match(nxt)
                or DEF_LINE_RE.match(nxt)
                or END_MODULE_LINE_RE.match(nxt)
            ):
                proof_end = j
                break
        j += 1
    # Trim trailing blank lines
    while proof_end > proof_start and not block[proof_end - 1].strip():
        proof_end -= 1
    return stmt_lines, block[proof_start:proof_end]


def chunk_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    starts = find_theorem_starts(lines)
    chunks: list[dict] = []
    for s in starts:
        e = find_block_end(lines, s)
        block = lines[s:e]
        stmt_lines, proof_lines = split_statement_proof(block)
        if not proof_lines:
            continue  # skip one-liners for now
        # Strip trailing blank lines from proof for cleanliness
        while proof_lines and not proof_lines[-1].strip():
            proof_lines.pop()
        if not proof_lines:
            continue
        chunks.append({
            "preamble": "\n".join(lines[:s]),
            "statement": "\n".join(stmt_lines),
            "proof": "\n".join(proof_lines),
            "source_file": str(path.relative_to(REPO)),
            "theorem_line": s + 1,  # 1-indexed for human readability
        })
    return chunks


def main() -> None:
    files = select_files()
    print(f"[build_prover_sft] {len(files)} source files from dedupe")

    all_chunks: list[dict] = []
    per_file: Counter = Counter()
    for p in files:
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        cs = chunk_file(p)
        per_file[p.name] = len(cs)
        all_chunks.extend(cs)

    print(f"[build_prover_sft] {len(all_chunks)} multi-line theorem chunks extracted")
    for name, n in per_file.most_common():
        print(f"  {n:4d}  {name}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
    print(f"[build_prover_sft] wrote {OUT}")


if __name__ == "__main__":
    main()
