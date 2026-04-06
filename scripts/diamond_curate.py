#!/usr/bin/env python3
"""
diamond_curate.py — Diamond spec curation pipeline with LLM judging.

Pipeline stages:
  1. LOAD     — Read Diamond specs from diamond_sft.jsonl, deduplicate
  2. JUDGE    — LLM (Claude Opus 4.6 via subagents) scores each spec on 5 dimensions
  3. FILTER   — Keep specs scoring >= threshold (default: 3.5/5 average)
  4. REASON   — LLM generates chain-of-thought reasoning for surviving specs
  5. ASSEMBLE — Produce final curated SFT dataset with reasoning

The LLM judging step writes judgments to outputs/logs/diamond_judgments.jsonl
so it can be resumed without re-judging already-scored specs.

Usage:
    # Full pipeline: judge + filter + reason + assemble
    python -m scripts.diamond_curate

    # Judge only (writes judgments file for review)
    python -m scripts.diamond_curate --judge-only

    # Skip judging, use existing judgments to filter + assemble
    python -m scripts.diamond_curate --skip-judge

    # Custom threshold
    python -m scripts.diamond_curate --threshold 4.0

    # Export judgment prompts for manual/subagent execution
    python -m scripts.diamond_curate --export-prompts
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("diamond_curate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIAMOND_IN = _REPO_ROOT / "data" / "processed" / "diamond_sft.jsonl"
_JUDGMENTS_FILE = _REPO_ROOT / "outputs" / "logs" / "diamond_judgments.jsonl"
_CURATED_OUT = _REPO_ROOT / "data" / "processed" / "diamond_curated.jsonl"
_PROMPTS_DIR = _REPO_ROOT / "outputs" / "judge_prompts"
_TRAIN_OUT = _REPO_ROOT / "data" / "processed" / "train.jsonl"

DEFAULT_THRESHOLD = 3.5  # minimum average score (out of 5) to keep


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpecRecord:
    """A Diamond spec with its metadata."""
    prompt_id: str
    prompt_text: str
    spec: str
    source: str
    tier: str
    semantic: dict = field(default_factory=dict)
    # Populated by judge
    judgment: Optional[dict] = None
    reasoning: Optional[str] = None

    @property
    def avg_score(self) -> float:
        if not self.judgment:
            return 0.0
        scores = self.judgment.get("scores", {})
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)


@dataclass
class Judgment:
    """LLM judgment of a spec's quality."""
    prompt_id: str
    scores: dict  # dimension -> score (1-5)
    avg_score: float
    rationale: str
    verdict: str  # "keep" | "reject" | "borderline"
    chain_of_thought: str = ""  # generated reasoning for training
    judged_at: str = ""


# ---------------------------------------------------------------------------
# Stage 1: LOAD
# ---------------------------------------------------------------------------

def load_diamond_specs() -> list[SpecRecord]:
    """Load and deduplicate Diamond specs."""
    if not _DIAMOND_IN.exists():
        log.error(f"Diamond specs not found: {_DIAMOND_IN}")
        return []

    seen_ids: set[str] = set()
    records: list[SpecRecord] = []

    with open(_DIAMOND_IN) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            pid = obj.get("_prompt_id", "")
            if pid in seen_ids:
                continue
            seen_ids.add(pid)

            # Extract spec and prompt from messages
            msgs = obj.get("messages", [])
            spec = ""
            prompt = ""
            for m in msgs:
                if m.get("role") == "user":
                    prompt = m.get("content", "")
                if m.get("channel") == "final":
                    spec = m.get("content", "")

            if not spec:
                continue

            records.append(SpecRecord(
                prompt_id=pid,
                prompt_text=prompt,
                spec=spec,
                source=obj.get("_source", ""),
                tier=obj.get("_tier", "diamond"),
                semantic=obj.get("_semantic", {}),
            ))

    log.info(f"Loaded {len(records)} unique Diamond specs")
    return records


# ---------------------------------------------------------------------------
# Stage 2: JUDGE — Build prompts for LLM evaluation
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are a TLA+ formal methods expert and training data quality reviewer.

Your task: evaluate a TLA+ specification on 5 dimensions, scoring each 1-5.

## Scoring Dimensions

1. **correctness** (1-5): Does the spec correctly model the described problem?
   - 5: Perfect model of the problem domain
   - 3: Mostly correct but missing edge cases or minor errors
   - 1: Fundamentally wrong model

2. **invariant_quality** (1-5): Are the invariants meaningful and complete?
   - 5: Invariants capture all key safety properties; removing any would allow bad states
   - 3: Has some invariants but misses important safety properties
   - 1: Trivial or irrelevant invariants (just TypeOK or TRUE)

3. **completeness** (1-5): Does the spec model the full problem scope?
   - 5: All described behaviors are modeled; state space is well-bounded
   - 3: Core behavior modeled but some aspects missing
   - 1: Only a trivial fragment of the problem

4. **tla_idiom** (1-5): Does the spec use proper TLA+ conventions?
   - 5: Clean, idiomatic TLA+; proper use of EXTENDS, operator definitions, fairness
   - 3: Functional but non-idiomatic or overly verbose
   - 1: Misuses TLA+ constructs, would confuse a TLA+ practitioner

5. **training_value** (1-5): How useful is this as a training example?
   - 5: Teaches generalizable patterns; prompt clearly describes problem; spec demonstrates good practices
   - 3: Acceptable example but formulaic or too simple
   - 1: Would teach bad habits or is too trivial to learn from

## Output Format

You MUST respond with ONLY a JSON object (no markdown fences, no explanation outside the JSON):
{
  "scores": {
    "correctness": <1-5>,
    "invariant_quality": <1-5>,
    "completeness": <1-5>,
    "tla_idiom": <1-5>,
    "training_value": <1-5>
  },
  "rationale": "<2-3 sentences explaining your scoring>",
  "verdict": "<keep|reject|borderline>"
}"""


def build_judge_prompt(record: SpecRecord) -> str:
    """Build the user prompt for judging a single spec."""
    return f"""## Problem Description

{record.prompt_text}

## TLA+ Specification

```tla+
{record.spec}
```

## Metadata
- Distinct states: {record.semantic.get('distinct_states', 'unknown')}
- Invariants checked: {record.semantic.get('invariants_checked', 'unknown')}
- Mutation caught: {record.semantic.get('mutation_caught', 'unknown')}

Score this specification on all 5 dimensions."""


REASONING_SYSTEM_PROMPT = """\
You are a TLA+ formal methods expert writing chain-of-thought reasoning for a training dataset.

Given a problem description and its correct TLA+ specification, write the reasoning process
that a skilled engineer would follow to derive this spec from the description. This reasoning
will be used as the "analysis" channel in training data to teach models HOW to think about
formal specification, not just WHAT to output.

Your reasoning should:
1. Identify the key state variables needed (and WHY each one is necessary)
2. Describe the state space and why it's bounded
3. Explain each action in Next (what real-world event it models)
4. Derive the safety invariants from the problem requirements
5. Explain why the invariants are sufficient to guarantee safety

Write 3-6 sentences of dense, technical reasoning. No fluff. Be specific to THIS problem.
Output ONLY the reasoning text, no JSON or formatting."""


def build_reasoning_prompt(record: SpecRecord) -> str:
    """Build the prompt for generating chain-of-thought reasoning."""
    return f"""## Problem Description

{record.prompt_text}

## Correct TLA+ Specification

```tla+
{record.spec}
```

Write the chain-of-thought reasoning for deriving this spec from the problem description."""


# ---------------------------------------------------------------------------
# Stage 2b: Load/save judgments
# ---------------------------------------------------------------------------

def load_existing_judgments() -> dict[str, dict]:
    """Load previously-saved judgments to avoid re-judging."""
    judgments: dict[str, dict] = {}
    if _JUDGMENTS_FILE.exists():
        with open(_JUDGMENTS_FILE) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    pid = obj.get("prompt_id", "")
                    if pid:
                        judgments[pid] = obj
                except json.JSONDecodeError:
                    continue
    return judgments


def save_judgment(judgment: dict):
    """Append a single judgment to the judgments file."""
    _JUDGMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_JUDGMENTS_FILE, "a") as f:
        f.write(json.dumps(judgment) + "\n")


def parse_judge_response(response: str, prompt_id: str) -> Optional[dict]:
    """Parse the JSON response from the LLM judge."""
    # Try to extract JSON from response
    response = response.strip()
    # Remove markdown fences if present
    response = re.sub(r"```(?:json)?\s*", "", response)
    response = re.sub(r"```\s*$", "", response)

    try:
        obj = json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        m = re.search(r"\{[^{}]*\"scores\"[^{}]*\{[^{}]*\}[^{}]*\}", response, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                log.warning(f"Failed to parse judge response for {prompt_id}")
                return None
        else:
            log.warning(f"No JSON found in judge response for {prompt_id}")
            return None

    scores = obj.get("scores", {})
    if not scores:
        return None

    avg = sum(scores.values()) / len(scores) if scores else 0
    return {
        "prompt_id": prompt_id,
        "scores": scores,
        "avg_score": round(avg, 2),
        "rationale": obj.get("rationale", ""),
        "verdict": obj.get("verdict", "borderline"),
        "judged_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Stage 2c: Export prompts for subagent execution
# ---------------------------------------------------------------------------

def export_judge_prompts(records: list[SpecRecord], batch_size: int = 15):
    """Export judge prompts as batch files for subagent execution."""
    existing = load_existing_judgments()
    to_judge = [r for r in records if r.prompt_id not in existing]

    if not to_judge:
        log.info("All specs already judged, nothing to export")
        return []

    _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    batches = []
    for i in range(0, len(to_judge), batch_size):
        batch = to_judge[i:i + batch_size]
        batch_id = i // batch_size
        batch_file = _PROMPTS_DIR / f"judge_batch_{batch_id:03d}.json"

        batch_data = []
        for rec in batch:
            batch_data.append({
                "prompt_id": rec.prompt_id,
                "system_prompt": JUDGE_SYSTEM_PROMPT,
                "user_prompt": build_judge_prompt(rec),
            })

        with open(batch_file, "w") as f:
            json.dump(batch_data, f, indent=2)

        batches.append((batch_id, batch_file, len(batch)))
        log.info(f"Exported batch {batch_id}: {len(batch)} specs -> {batch_file}")

    log.info(f"Exported {len(batches)} batches ({len(to_judge)} specs) to {_PROMPTS_DIR}")
    return batches


def export_reasoning_prompts(records: list[SpecRecord], batch_size: int = 10):
    """Export reasoning generation prompts for subagent execution."""
    _PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    batches = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        batch_id = i // batch_size
        batch_file = _PROMPTS_DIR / f"reason_batch_{batch_id:03d}.json"

        batch_data = []
        for rec in batch:
            batch_data.append({
                "prompt_id": rec.prompt_id,
                "system_prompt": REASONING_SYSTEM_PROMPT,
                "user_prompt": build_reasoning_prompt(rec),
            })

        with open(batch_file, "w") as f:
            json.dump(batch_data, f, indent=2)

        batches.append((batch_id, batch_file, len(batch)))

    log.info(f"Exported {len(batches)} reasoning batches ({len(records)} specs)")
    return batches


# ---------------------------------------------------------------------------
# Stage 3: FILTER
# ---------------------------------------------------------------------------

def filter_by_threshold(
    records: list[SpecRecord],
    judgments: dict[str, dict],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[SpecRecord], list[SpecRecord]]:
    """Partition records into kept and rejected based on judge scores."""
    kept: list[SpecRecord] = []
    rejected: list[SpecRecord] = []

    for rec in records:
        j = judgments.get(rec.prompt_id)
        if not j:
            log.debug(f"No judgment for {rec.prompt_id}, skipping")
            rejected.append(rec)
            continue

        rec.judgment = j
        avg = j.get("avg_score", 0)
        verdict = j.get("verdict", "borderline")

        if verdict == "reject" or avg < threshold:
            rejected.append(rec)
        else:
            kept.append(rec)

    log.info(f"Filter (threshold={threshold}): kept={len(kept)} rejected={len(rejected)}")
    return kept, rejected


# ---------------------------------------------------------------------------
# Stage 5: ASSEMBLE
# ---------------------------------------------------------------------------

from src.training.dataset_builder import _DEVELOPER_PROMPT  # single source of truth


def assemble_curated_dataset(
    records: list[SpecRecord],
    reasoning_map: dict[str, str],
) -> int:
    """Write the final curated SFT dataset."""
    count = 0
    with open(_CURATED_OUT, "w") as f:
        for rec in records:
            cot = reasoning_map.get(rec.prompt_id, "")
            if not cot:
                cot = (
                    "I'll analyze the problem requirements, identify state variables "
                    "and safety invariants, then write a verified TLA+ specification."
                )

            row = {
                "_tier": "diamond_curated",
                "_prompt_id": rec.prompt_id,
                "_source": rec.source,
                "_timestamp": datetime.now().isoformat(),
                "_semantic": rec.semantic,
                "_judgment": rec.judgment,
                "messages": [
                    {"role": "developer", "content": _DEVELOPER_PROMPT},
                    {"role": "user", "content": rec.prompt_text},
                    {"role": "assistant", "channel": "analysis", "content": cot},
                    {"role": "assistant", "channel": "final", "content": rec.spec.strip()},
                ],
            }
            f.write(json.dumps(row) + "\n")
            count += 1

    log.info(f"Assembled {count} curated specs -> {_CURATED_OUT}")
    return count


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    threshold: float = DEFAULT_THRESHOLD,
    judge_only: bool = False,
    skip_judge: bool = False,
    export_prompts: bool = False,
):
    """Run the full curation pipeline."""

    # Stage 1: LOAD
    log.info("=" * 60)
    log.info("STAGE 1: Load Diamond specs")
    log.info("=" * 60)
    records = load_diamond_specs()
    if not records:
        log.error("No specs to process")
        return

    # Stage 2: JUDGE
    log.info("=" * 60)
    log.info("STAGE 2: LLM Judge")
    log.info("=" * 60)

    if export_prompts:
        judge_batches = export_judge_prompts(records)
        log.info(f"\nExported {len(judge_batches)} judge batches.")
        log.info("Run subagents on each batch, then re-run with --skip-judge")
        return

    judgments = load_existing_judgments()
    unjudged = [r for r in records if r.prompt_id not in judgments]

    if skip_judge:
        log.info(f"Skipping judge (--skip-judge). Using {len(judgments)} existing judgments.")
    elif unjudged:
        log.info(f"{len(unjudged)} specs need judging ({len(judgments)} already judged)")
        log.info("Export prompts with --export-prompts, then run subagents.")
        log.info("Or use the subagent runner to judge interactively.")
        # Export for convenience
        export_judge_prompts(records)
        return
    else:
        log.info(f"All {len(records)} specs already judged")

    if judge_only:
        log.info("Judge-only mode, stopping here.")
        return

    # Stage 3: FILTER
    log.info("=" * 60)
    log.info("STAGE 3: Filter by quality threshold")
    log.info("=" * 60)
    kept, rejected = filter_by_threshold(records, judgments, threshold)

    # Score distribution
    all_scores = [j.get("avg_score", 0) for j in judgments.values()]
    if all_scores:
        log.info(f"Score distribution: min={min(all_scores):.1f} "
                 f"median={sorted(all_scores)[len(all_scores)//2]:.1f} "
                 f"max={max(all_scores):.1f} mean={sum(all_scores)/len(all_scores):.1f}")

    # Stage 4: REASON
    log.info("=" * 60)
    log.info("STAGE 4: Chain-of-thought reasoning")
    log.info("=" * 60)
    reasoning_map: dict[str, str] = {}

    # Check for existing reasoning in judgments
    for rec in kept:
        j = judgments.get(rec.prompt_id, {})
        cot = j.get("chain_of_thought", "")
        if cot:
            reasoning_map[rec.prompt_id] = cot

    needs_reasoning = [r for r in kept if r.prompt_id not in reasoning_map]
    if needs_reasoning:
        log.info(f"{len(needs_reasoning)} specs need reasoning ({len(reasoning_map)} already have it)")
        export_reasoning_prompts(needs_reasoning)
        log.info("Run reasoning subagents, then re-run pipeline.")
    else:
        log.info(f"All {len(kept)} specs have reasoning")

    # Stage 5: ASSEMBLE
    log.info("=" * 60)
    log.info("STAGE 5: Assemble curated dataset")
    log.info("=" * 60)
    count = assemble_curated_dataset(kept, reasoning_map)

    # Summary
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Input: {len(records)} Diamond specs")
    log.info(f"  Judged: {len(judgments)}")
    log.info(f"  Kept: {len(kept)} (threshold={threshold})")
    log.info(f"  Rejected: {len(rejected)}")
    log.info(f"  With reasoning: {len(reasoning_map)}")
    log.info(f"  Output: {count} specs -> {_CURATED_OUT}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diamond spec curation pipeline")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Minimum avg score to keep (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--judge-only", action="store_true",
                        help="Only run judging stage")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip judging, use existing judgments")
    parser.add_argument("--export-prompts", action="store_true",
                        help="Export judge prompts for subagent execution")
    args = parser.parse_args()

    run_pipeline(
        threshold=args.threshold,
        judge_only=args.judge_only,
        skip_judge=args.skip_judge,
        export_prompts=args.export_prompts,
    )


if __name__ == "__main__":
    main()
