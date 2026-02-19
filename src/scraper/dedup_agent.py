"""
dedup_agent.py — Content deduplication for TLA+ training records.

Two-pass deduplication strategy:
  1. Exact dedup: SHA-256 hash of normalised spec content (catches identical files)
  2. Near-dedup: MinHash LSH at Jaccard threshold 0.8 (catches variant copies,
     whitespace changes, minor renames)

The FormaLLM seed corpus is loaded first to anchor the hash sets.  All
subsequent scrapes check against the seed before being accepted.

MinHash LSH details
-------------------
- Token set: whitespace-split tokens from the TLA+ source, lowercased
- Number of hash functions (num_perm): 128 — good precision/recall balance
- Threshold: 0.8 Jaccard coefficient — empirically good for TLA+ (catches
  near-copies like MultiPaxos variants while allowing distinct specs at the
  same threshold)

Research note
-------------
datasketch's MinHashLSH operates in sub-linear time, making it viable even
when the corpus grows to tens of thousands of specs.  For our target of
~60k specs this adds <5 min overhead versus the hours TLC validation takes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

from datasketch import MinHash, MinHashLSH

from src.shared.schemas.dataset_schema import DatasetRecord

_NUM_PERM = 128
_JACCARD_THRESHOLD = 0.8


class DedupAgent:
    """
    Stateful deduplication agent.  Call `add_seed(records)` first to load
    the anchor corpus, then `filter(records)` to drop duplicates.
    """

    def __init__(
        self,
        num_perm: int = _NUM_PERM,
        threshold: float = _JACCARD_THRESHOLD,
    ):
        self._exact_hashes: set[str] = set()
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._num_perm = num_perm
        self._n_added = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def add_seed(self, records: Iterable[DatasetRecord]) -> int:
        """
        Load seed records (e.g. FormaLLM) into the dedup index.
        These are always kept — they are the anchor.

        Returns the number of records indexed.
        """
        count = 0
        for record in records:
            self._add(record.id, record.tla_content)
            count += 1
        print(f"[dedup_agent] Seeded with {count} records.")
        return count

    def filter(self, records: Iterable[DatasetRecord]) -> Iterator[DatasetRecord]:
        """
        Yield only records that are not near-duplicates of anything already indexed.
        Accepted records are added to the index so later records deduplicate
        against them too (online dedup).
        """
        n_seen = 0
        n_kept = 0
        for record in records:
            n_seen += 1
            content_hash = self._content_hash(record.tla_content)

            # Fast path: exact duplicate
            if content_hash in self._exact_hashes:
                continue

            # Slower path: near-duplicate via MinHash
            mh = self._make_minhash(record.tla_content)
            neighbours = self._lsh.query(mh)
            if neighbours:
                continue  # near-duplicate found

            # Accept — add to index
            self._add(record.id, record.tla_content)
            n_kept += 1
            yield record

        print(f"[dedup_agent] filter: kept {n_kept}/{n_seen} records ({n_seen - n_kept} duplicates dropped).")

    def size(self) -> int:
        """Number of records currently indexed."""
        return self._n_added

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _add(self, record_id: str, tla_content: str) -> None:
        content_hash = self._content_hash(tla_content)
        self._exact_hashes.add(content_hash)
        mh = self._make_minhash(tla_content)
        key = f"{record_id}:{self._n_added}"
        try:
            self._lsh.insert(key, mh)
        except ValueError:
            pass  # already inserted (can happen with duplicate IDs)
        self._n_added += 1

    def _make_minhash(self, tla_content: str) -> MinHash:
        mh = MinHash(num_perm=self._num_perm)
        tokens = self._tokenise(tla_content)
        for token in tokens:
            mh.update(token.encode("utf-8"))
        return mh

    @staticmethod
    def _tokenise(tla_content: str) -> list[str]:
        """
        Normalise and tokenise TLA+ source for MinHash.
        Lowercasing and whitespace normalisation make the hash robust to
        minor formatting differences.
        """
        return tla_content.lower().split()

    @staticmethod
    def _content_hash(tla_content: str) -> str:
        return hashlib.sha256(tla_content.strip().encode("utf-8")).hexdigest()


def dedup_jsonl_files(
    seed_path: Path,
    input_paths: list[Path],
    output_path: Path,
) -> int:
    """
    Convenience function: load seed JSONL, run dedup on input JSONL files,
    write accepted records to output JSONL.

    Returns the number of records written.
    """
    agent = DedupAgent()

    # Load seed
    if seed_path.exists():
        def _iter_seed() -> Iterator[DatasetRecord]:
            for line in seed_path.open(encoding="utf-8"):
                line = line.strip()
                if line:
                    yield DatasetRecord.from_dict(json.loads(line))
        agent.add_seed(_iter_seed())

    # Stream through input files, filter, write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for path in input_paths:
            if not path.exists():
                print(f"[dedup_agent] Input not found: {path}")
                continue

            def _iter_input(p: Path) -> Iterator[DatasetRecord]:
                for line in p.open(encoding="utf-8"):
                    line = line.strip()
                    if line:
                        yield DatasetRecord.from_dict(json.loads(line))

            for record in agent.filter(_iter_input(path)):
                fout.write(record.to_json(indent=None) + "\n")
                count += 1

    print(f"[dedup_agent] Wrote {count} deduplicated records → {output_path}")
    return count
