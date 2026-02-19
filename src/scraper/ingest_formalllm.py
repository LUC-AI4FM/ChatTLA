"""
ingest_formalllm.py — Seed corpus ingestion from the FormaLLM submodule.

FormaLLM (https://github.com/LUC-FMitF/FormaLLM) is our Tier-1 seed dataset.
It contains 205 MIT-licensed, curated TLA+ specs in the directory structure:

    data/FormaLLM/data/<SpecFamily>/
        tla/  <module>.tla, <module>_clean.tla
        txt/  <module>_comments.txt, <module>_comments_clean.txt
        cfg/  <module>.cfg  (may be absent)

The `all_models.json` index maps each spec's files.

This module reads that index, constructs a DatasetRecord for each spec, and
writes normalised JSONL records to `data/raw/formalllm.jsonl`.

Design decision: We use the `_clean` variants as the canonical TLA+ content
(FormaLLM already stripped irrelevant comments to make specs easier to work
with), while keeping originals available on disk via the submodule.

Usage
-----
    python -m src.scraper.ingest_formalllm
    # or from pipeline.py: ingest_formalllm.run()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from src.shared.schemas.dataset_schema import DatasetRecord

# Paths relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORMALLLM_ROOT = _REPO_ROOT / "data" / "FormaLLM"
_ALL_MODELS_JSON = _FORMALLLM_ROOT / "data" / "all_models.json"
_OUTPUT_JSONL = _REPO_ROOT / "data" / "raw" / "formalllm.jsonl"


def iter_records() -> Iterator[DatasetRecord]:
    """
    Yield DatasetRecord objects for each FormaLLM spec entry.

    Skips entries where the clean .tla file is missing on disk.
    """
    with _ALL_MODELS_JSON.open(encoding="utf-8") as f:
        index = json.load(f)

    for entry in index["data"]:
        spec_id = entry["id"]
        module = entry["model"]
        family_dir = _FORMALLLM_ROOT / "data" / module

        # Prefer clean variant; fall back to original
        tla_clean_name = entry.get("tla_clean") or entry.get("tla_original")
        if not tla_clean_name:
            continue

        tla_path = family_dir / "tla" / tla_clean_name
        if not tla_path.exists():
            # Try original as fallback
            orig_name = entry.get("tla_original")
            if orig_name:
                tla_path = family_dir / "tla" / orig_name
            if not tla_path.exists():
                print(f"[ingest_formalllm] WARNING: missing {tla_path}, skipping")
                continue

        tla_content = tla_path.read_text(encoding="utf-8", errors="replace")

        # --- .cfg -----------------------------------------------------------
        cfg_content: str | None = None
        cfg_name = entry.get("cfg")
        if cfg_name:
            cfg_path = family_dir / "cfg" / cfg_name
            if cfg_path.exists():
                cfg_content = cfg_path.read_text(encoding="utf-8", errors="replace")

        # --- Annotation text (pre-existing comments) ------------------------
        comments_clean = entry.get("comments_clean") or entry.get("comments")
        nl_description = ""
        if comments_clean:
            comments_path = family_dir / "txt" / comments_clean
            if comments_path.exists():
                nl_description = comments_path.read_text(encoding="utf-8", errors="replace").strip()

        # --- Build record ---------------------------------------------------
        record = DatasetRecord(
            id=DatasetRecord.make_id(tla_content),
            source=f"formalllm:{spec_id}:{module}",
            license="MIT",
            tla_content=tla_content,
            cfg_content=cfg_content,
            metadata={
                "formalllm_id": spec_id,
                "module_name": module,
                "tla_file": str(tla_path.relative_to(_REPO_ROOT)),
            },
        )

        # Attach pre-existing natural language description if available
        if nl_description:
            from src.shared.schemas.dataset_schema import Annotation
            record.annotation = Annotation(natural_language_description=nl_description)

        yield record


def run(output_path: Path = _OUTPUT_JSONL, overwrite: bool = False) -> int:
    """
    Ingest all FormaLLM specs and write to JSONL.

    Returns
    -------
    int   Number of records written.
    """
    if output_path.exists() and not overwrite:
        print(f"[ingest_formalllm] {output_path} already exists; use overwrite=True to regenerate.")
        # Count existing
        return sum(1 for _ in output_path.open())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for record in iter_records():
            fout.write(record.to_json(indent=None) + "\n")
            count += 1

    print(f"[ingest_formalllm] Wrote {count} records → {output_path}")
    return count


if __name__ == "__main__":
    run()
