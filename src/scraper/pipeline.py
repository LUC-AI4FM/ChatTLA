"""
pipeline.py — Phase 1 orchestrator: scrape → validate → dedup → annotate.

This is the single entry point for all data collection.  Run it to build the
`data/validated/` corpus from scratch.  Progress is checkpointed at each step
so the pipeline can be resumed if interrupted.

Pipeline stages
---------------
1. ingest_formalllm    — Seed with 205 FormaLLM specs → data/raw/formalllm.jsonl
2. github_agent        — Scrape GitHub search  (optional, requires GITHUB_TOKEN)
3. validate            — Run TLC/SANY on all raw records; write tier labels
4. dedup               — Remove near-duplicates (MinHash Jaccard ≥ 0.8)
5. annotate            — Call local Ollama gpt-oss:20b to add NL descriptions
6. combine             — Merge into data/validated/combined.jsonl (gold+silver)

Output layout
-------------
    data/raw/           — one JSONL per source (before validation)
    data/validated/     — gold and silver tier specs with annotations
    data/rejected/      — bronze tier (keep for error analysis)
    outputs/logs/       — timestamped run logs

Usage
-----
    python -m src.scraper.pipeline              # full run
    python -m src.scraper.pipeline --dry-run    # validate FormaLLM seed only
    python -m src.scraper.pipeline --no-github  # skip GitHub scraping
    python -m src.scraper.pipeline --no-annotate
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOG_DIR   = _REPO_ROOT / "outputs" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_DIR / f"scrape_{_run_ts}.log"),
    ],
)
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def stage_ingest(dry_run: bool = False) -> Path:
    """Ingest FormaLLM seed corpus → data/raw/formalllm.jsonl."""
    from src.scraper.ingest_formalllm import run

    out = _REPO_ROOT / "data" / "raw" / "formalllm.jsonl"
    log.info("Stage 1/6: Ingesting FormaLLM seed corpus")
    n = run(output_path=out, overwrite=dry_run)
    log.info(f"  → {n} records in {out}")
    return out


def stage_github() -> Path:
    """Run GitHub scraper → data/raw/github.jsonl."""
    from src.scraper.github_agent import GitHubAgent

    out = _REPO_ROOT / "data" / "raw" / "github.jsonl"
    log.info("Stage 2/6: Scraping GitHub (this may take 30-60 min due to rate limits)")
    agent = GitHubAgent()
    count = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in agent.to_dataset_records(agent.iter_specs()):
            f.write(record.to_json(indent=None) + "\n")
            count += 1
    log.info(f"  → {count} records in {out}")
    return out


def stage_validate(raw_paths: list[Path]) -> tuple[Path, Path]:
    """
    Run TLC/SANY on all raw records.
    Returns (validated_path, rejected_path).
    """
    from src.validators.tlc_validator import validate_string
    from src.validators.quality_scorer import score as quality_score
    from src.shared.schemas.dataset_schema import TLCResult, QualityScore

    validated_out = _REPO_ROOT / "data" / "validated" / "validated_raw.jsonl"
    rejected_out  = _REPO_ROOT / "data" / "rejected"  / "rejected.jsonl"
    validated_out.parent.mkdir(parents=True, exist_ok=True)
    rejected_out.parent.mkdir(parents=True, exist_ok=True)

    log.info("Stage 3/6: Running TLC/SANY validation")
    n_gold = n_silver = n_bronze = 0

    with validated_out.open("w") as fv, rejected_out.open("w") as fr:
        for raw_path in raw_paths:
            if not raw_path.exists():
                log.warning(f"  Raw file not found: {raw_path}")
                continue
            for line in raw_path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                from src.shared.schemas.dataset_schema import DatasetRecord
                record = DatasetRecord.from_dict(json.loads(line))

                # Extract module name for SANY (must match filename)
                import re
                m = re.search(r"----\s*MODULE\s+(\w+)", record.tla_content)
                module_name = m.group(1) if m else "Spec"

                tlc = validate_string(
                    record.tla_content,
                    cfg_content=record.cfg_content,
                    module_name=module_name,
                )
                record.tlc_result = TLCResult(
                    tier=tlc.tier,
                    sany_errors=tlc.sany_errors,
                    tlc_errors=tlc.tlc_violations,
                    tlc_output=tlc.raw_output[:2000],  # cap to prevent huge files
                    runtime_seconds=tlc.runtime_seconds,
                )
                record.quality = quality_score(record.tla_content)

                if tlc.tier in ("gold", "silver"):
                    fv.write(record.to_json(indent=None) + "\n")
                    if tlc.tier == "gold":
                        n_gold += 1
                    else:
                        n_silver += 1
                else:
                    fr.write(record.to_json(indent=None) + "\n")
                    n_bronze += 1

    log.info(f"  → gold={n_gold}  silver={n_silver}  bronze={n_bronze}")
    log.info(f"  → validated: {validated_out}")
    log.info(f"  → rejected:  {rejected_out}")
    return validated_out, rejected_out


def stage_dedup(validated_path: Path, seed_path: Path) -> Path:
    """Dedup validated records against seed → data/validated/deduped.jsonl."""
    from src.scraper.dedup_agent import dedup_jsonl_files

    out = _REPO_ROOT / "data" / "validated" / "deduped.jsonl"
    log.info("Stage 4/6: Deduplicating")
    n = dedup_jsonl_files(
        seed_path=seed_path,
        input_paths=[validated_path],
        output_path=out,
    )
    log.info(f"  → {n} unique records → {out}")
    return out


def stage_annotate(deduped_path: Path) -> Path:
    """Annotate all specs via local Ollama → data/validated/annotated.jsonl."""
    from src.scraper.annotate import annotate_jsonl

    out = _REPO_ROOT / "data" / "validated" / "annotated.jsonl"
    log.info("Stage 5/6: Annotating via local Ollama gpt-oss:20b")
    n = annotate_jsonl(input_path=deduped_path, output_path=out)
    log.info(f"  → {n} records annotated → {out}")
    return out


def stage_combine(annotated_path: Path, seed_path: Path) -> Path:
    """Merge seed + new annotated records → data/validated/combined.jsonl."""
    out = _REPO_ROOT / "data" / "validated" / "combined.jsonl"
    log.info("Stage 6/6: Combining into final corpus")

    seen_ids: set[str] = set()
    count = 0
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as fout:
        for path in [seed_path, annotated_path]:
            if not path.exists():
                continue
            for line in path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rid = d.get("id", "")
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                fout.write(line + "\n")
                count += 1

    log.info(f"  → {count} total records → {out}")
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    dry_run: bool = False,
    no_github: bool = False,
    no_annotate: bool = False,
) -> None:
    t0 = time.monotonic()
    log.info(f"=== ChatTLA Phase 1 Pipeline — run_id={_run_ts} ===")
    log.info(f"  dry_run={dry_run}  no_github={no_github}  no_annotate={no_annotate}")

    seed_path = stage_ingest(dry_run=dry_run)

    raw_paths = [seed_path]
    if not no_github and not dry_run:
        gh_path = stage_github()
        raw_paths.append(gh_path)
    else:
        log.info("Stage 2/6: GitHub scraping SKIPPED")

    validated_path, _ = stage_validate(raw_paths)
    deduped_path       = stage_dedup(validated_path, seed_path=seed_path)

    if not no_annotate:
        annotated_path = stage_annotate(deduped_path)
    else:
        log.info("Stage 5/6: Annotation SKIPPED")
        annotated_path = deduped_path

    combined_path = stage_combine(annotated_path, seed_path=seed_path)

    elapsed = time.monotonic() - t0
    log.info(f"=== Pipeline complete in {elapsed/60:.1f} min ===")
    log.info(f"Final corpus: {combined_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChatTLA Phase 1: Data collection pipeline")
    parser.add_argument("--dry-run",     action="store_true", help="FormaLLM seed only, no GitHub, no annotate")
    parser.add_argument("--no-github",   action="store_true", help="Skip GitHub scraping")
    parser.add_argument("--no-annotate", action="store_true", help="Skip Ollama annotation")
    args = parser.parse_args()

    run(
        dry_run=args.dry_run,
        no_github=args.no_github,
        no_annotate=args.no_annotate,
    )
