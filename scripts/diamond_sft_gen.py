#!/usr/bin/env python3
"""
diamond_sft_gen.py — Rejection-sampling pipeline for Diamond-tier TLA+ specs.

Instead of DPO/KTO preference learning on noisy gold/silver/bronze labels,
this script generates specs and keeps ONLY those that pass the full
Diamond gate (SANY + TLC + semantic checks: distinct_states > 1,
non-trivial invariants, mutation caught).

The output is a clean SFT dataset of semantically verified specs.

Usage:
    # Audit existing gold specs (no generation, just filter)
    python -m scripts.diamond_sft_gen --audit-only

    # Generate new Diamond specs using teacher model (gpt-oss:120b via Ollama)
    python -m scripts.diamond_sft_gen --generate --target 200

    # Generate using the student model (chattla:20b)
    python -m scripts.diamond_sft_gen --generate --model chattla:20b --target 200

    # Use all available prompts, 5 attempts each
    python -m scripts.diamond_sft_gen --generate --attempts 5 --target 200
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("diamond_sft")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DESCRIPTIONS_JSON = _REPO_ROOT / "data" / "derived" / "tla_descriptions.json"
_BENCHMARK_JSON = _REPO_ROOT / "data" / "benchmarks" / "benchmark_suite.json"
_GOLD_ALL_BM = _REPO_ROOT / "data" / "processed" / "gold_all_benchmarks_sft.jsonl"
_GOLD_BM = _REPO_ROOT / "data" / "processed" / "gold_benchmark_sft.jsonl"
_TEACHER_GOLD = _REPO_ROOT / "data" / "processed" / "teacher_gold.jsonl"
_TRAIN_JSONL = _REPO_ROOT / "data" / "processed" / "train.jsonl"
_GOLD_CACHE = _REPO_ROOT / "data" / "processed" / "rl" / "gold_spec_cache.jsonl"
_DIAMOND_OUT = _REPO_ROOT / "data" / "processed" / "diamond_sft.jsonl"
_DIAMOND_LOG = _REPO_ROOT / "outputs" / "logs" / "diamond_audit.jsonl"

# Ollama defaults
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_TEACHER_MODEL", "gpt-oss:120b")


@dataclass
class DiamondResult:
    """Result of Diamond validation for a single spec."""
    prompt_id: str
    prompt_text: str
    spec: str
    sany_pass: bool = False
    tlc_tier: str = "bronze"
    is_diamond: bool = False
    distinct_states: int = 0
    action_coverage: float = 0.0
    trivial_invariant: bool = False
    mutation_tested: bool = False
    mutation_caught: bool = False
    invariants_checked: int = 0
    attempt: int = 0
    model: str = ""
    error: str = ""


def _extract_spec(raw: str) -> str:
    """Extract TLA+ module from model output, stripping markdown fences."""
    if not raw:
        return ""
    raw = re.sub(r"```(?:tla\+?|TLA\+?)?\s*\n?", "", raw)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    # Strip analysis/reasoning tags
    raw = re.sub(r"<analysis>.*?</analysis>", "", raw, flags=re.DOTALL)
    m_start = re.search(r"-{4,}\s*MODULE\s+\w+\s*-{4,}", raw)
    m_end = raw.rfind("====")
    if m_start and m_end > m_start.start():
        return raw[m_start.start(): m_end + 4].strip()
    return raw.strip()


def _get_module_name(spec: str) -> str:
    m = re.search(r"MODULE\s+(\w+)", spec)
    return m.group(1) if m else "Temp"


def validate_diamond(spec: str, prompt_id: str = "", prompt_text: str = "",
                     attempt: int = 0, model: str = "") -> DiamondResult:
    """Run full SANY + TLC + Diamond validation on a spec string."""
    from src.validators.sany_validator import validate_string as sany_validate
    from src.validators.tlc_validator import validate_string as tlc_validate

    result = DiamondResult(
        prompt_id=prompt_id, prompt_text=prompt_text[:500],
        spec=spec, attempt=attempt, model=model,
    )

    if "MODULE" not in spec:
        result.error = "no MODULE found"
        return result

    mod_name = _get_module_name(spec)

    # SANY
    try:
        sr = sany_validate(spec, module_name=mod_name)
    except Exception as e:
        result.error = f"sany_error: {e}"
        return result

    if not sr.valid:
        result.error = f"sany_fail: {'; '.join(sr.errors[:3])}"
        return result
    result.sany_pass = True

    # TLC (with semantic info computed for gold)
    try:
        tr = tlc_validate(spec, module_name=mod_name, timeout=60)
    except Exception as e:
        result.error = f"tlc_error: {e}"
        return result

    result.tlc_tier = tr.tier
    result.distinct_states = tr.semantic.distinct_states
    result.action_coverage = tr.semantic.action_coverage
    result.trivial_invariant = tr.semantic.trivial_invariant
    result.mutation_tested = tr.semantic.mutation_tested
    result.mutation_caught = tr.semantic.mutation_caught
    result.invariants_checked = tr.semantic.invariants_checked
    result.is_diamond = tr.is_diamond

    if not result.is_diamond and tr.tier == "gold":
        # Gold but not diamond — log why
        reasons = []
        if tr.semantic.distinct_states <= 1:
            reasons.append(f"states={tr.semantic.distinct_states}")
        if tr.semantic.trivial_invariant:
            reasons.append("trivial_inv")
        if tr.semantic.invariants_checked == 0:
            reasons.append("no_invariants")
        if not tr.semantic.mutation_tested:
            reasons.append("mutation_not_tested")
        elif not tr.semantic.mutation_caught:
            reasons.append("mutation_not_caught")
        result.error = f"gold_not_diamond: {', '.join(reasons)}"

    return result


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts() -> list[dict]:
    """Load all available prompts from descriptions + benchmarks."""
    prompts = []
    seen_ids = set()

    # 1. Module descriptions (205 specs from tlaplus/Examples)
    if _DESCRIPTIONS_JSON.exists():
        with open(_DESCRIPTIONS_JSON) as f:
            descs = json.load(f)
        for d in descs:
            pid = d.get("id") or d.get("module_name", "")
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            desc = d.get("description", {})
            narrative = desc.get("narrative", "") if isinstance(desc, dict) else str(desc)
            technical = desc.get("technical", "") if isinstance(desc, dict) else ""
            if not narrative:
                continue
            # Build a rich prompt from narrative + technical details
            prompt_text = narrative.strip()
            if isinstance(technical, dict):
                tech_parts = []
                for k, v in technical.items():
                    if v and k in ("variables", "actions", "invariants", "constants"):
                        tech_parts.append(f"{k}: {v}")
                if tech_parts:
                    prompt_text += "\n\nTechnical details:\n" + "\n".join(tech_parts)
            prompts.append({"id": pid, "text": prompt_text, "source": "descriptions"})

    # 2. Benchmark problems
    if _BENCHMARK_JSON.exists():
        with open(_BENCHMARK_JSON) as f:
            benchmarks = json.load(f)
        for bm in benchmarks:
            pid = bm.get("id", bm.get("name", ""))
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            desc = bm.get("description", "")
            name = bm.get("name", "")
            prompt_text = f"{name}: {desc}" if name else desc
            prompts.append({"id": pid, "text": prompt_text, "source": "benchmark"})

    log.info(f"Loaded {len(prompts)} prompts ({len([p for p in prompts if p['source'] == 'descriptions'])} descriptions, "
             f"{len([p for p in prompts if p['source'] == 'benchmark'])} benchmarks)")
    return prompts


# ---------------------------------------------------------------------------
# Audit existing data
# ---------------------------------------------------------------------------

def audit_existing_specs() -> list[DiamondResult]:
    """Run Diamond validation on all existing gold specs to establish baseline."""
    results = []

    # Collect specs from all JSONL sources
    sources = [
        (_GOLD_ALL_BM, "gold_all_benchmarks"),
        (_GOLD_BM, "gold_benchmark"),
        (_TEACHER_GOLD, "teacher_gold"),
        (_GOLD_CACHE, "gold_cache"),
        (_TRAIN_JSONL, "train"),
    ]

    specs_to_check: list[tuple[str, str, str, str]] = []  # (source, pid, prompt, spec)

    for path, source_name in sources:
        if not path.exists():
            continue
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                pid = obj.get("_prompt_id", obj.get("prompt_id", f"{source_name}_{i}"))

                # Extract spec from messages format
                msgs = obj.get("messages", [])
                spec = ""
                prompt = ""
                for m in msgs:
                    if m.get("role") == "user":
                        prompt = m.get("content", "")
                    if m.get("channel") == "final" or (m.get("role") == "assistant" and "MODULE" in m.get("content", "")):
                        spec = m.get("content", "")

                # Also check direct spec field (gold_cache format)
                if not spec:
                    spec = obj.get("spec", "")
                if not prompt:
                    prompt = obj.get("prompt_text", obj.get("prompt", ""))

                if spec and "MODULE" in spec:
                    specs_to_check.append((source_name, pid, prompt, spec))

    log.info(f"Auditing {len(specs_to_check)} existing specs against Diamond gate...")

    for i, (source, pid, prompt, spec) in enumerate(specs_to_check):
        r = validate_diamond(spec, prompt_id=f"{source}/{pid}", prompt_text=prompt)
        results.append(r)
        if (i + 1) % 25 == 0 or (i + 1) == len(specs_to_check):
            n_d = sum(1 for x in results if x.is_diamond)
            n_g = sum(1 for x in results if x.tlc_tier == "gold")
            log.info(f"  [{i+1}/{len(specs_to_check)}] diamond={n_d} gold={n_g}")

    # Summary
    diamond_count = sum(1 for r in results if r.is_diamond)
    gold_count = sum(1 for r in results if r.tlc_tier == "gold")
    sany_count = sum(1 for r in results if r.sany_pass)

    log.info(f"Audit complete: {len(results)} specs → "
             f"SANY={sany_count} TLC_gold={gold_count} DIAMOND={diamond_count}")

    # Breakdown by source
    by_source: dict[str, dict] = {}
    for r in results:
        src = r.prompt_id.split("/")[0] if "/" in r.prompt_id else "unknown"
        if src not in by_source:
            by_source[src] = {"total": 0, "sany": 0, "gold": 0, "diamond": 0}
        by_source[src]["total"] += 1
        by_source[src]["sany"] += int(r.sany_pass)
        by_source[src]["gold"] += int(r.tlc_tier == "gold")
        by_source[src]["diamond"] += int(r.is_diamond)
    for src, counts in sorted(by_source.items()):
        log.info(f"  {src}: {counts['total']} total → {counts['sany']} SANY, "
                 f"{counts['gold']} gold, {counts['diamond']} diamond")

    # Log failure reasons for gold-but-not-diamond
    gold_not_diamond = [r for r in results if r.tlc_tier == "gold" and not r.is_diamond]
    if gold_not_diamond:
        reason_counts: dict[str, int] = {}
        for r in gold_not_diamond:
            for reason in (r.error.replace("gold_not_diamond: ", "").split(", ") if r.error else ["unknown"]):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        log.info(f"Gold-not-Diamond reasons ({len(gold_not_diamond)} specs):")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            log.info(f"  {reason}: {count}")

    return results


# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str, temperature: float = 0.3,
                 max_tokens: int = 4096, timeout: int = 180) -> Optional[str]:
    """Generate a spec via Ollama."""
    import requests

    system_prompt = (
        "You are an expert TLA+ formal methods engineer. Generate complete, "
        "syntactically correct TLA+ specifications that pass both SANY (parser) "
        "and TLC (model checker) validation.\n\n"
        "Critical requirements for semantic correctness:\n"
        "- Define meaningful INVARIANTS that actually constrain behavior (not just TypeOK)\n"
        "- Ensure the state space has more than 1 reachable state\n"
        "- Use small CONSTANT values so TLC can fully explore the state space\n"
        "- Include both safety invariants and type invariants\n"
        "- The spec must model real behavior: actions should be reachable from Init via Next\n\n"
        "After the TLA+ module, append a TLC configuration block:\n"
        "  SPECIFICATION Spec\n"
        "  INVARIANT TypeOK SafetyInvariant\n"
        "  CONSTANT N = 3  (or similar small values)\n\n"
        "Output ONLY the TLA+ module and config. No markdown fences, no explanation."
    )

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "system": system_prompt,
                "prompt": f"Write a TLA+ specification for the following:\n\n{prompt}",
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        log.debug(f"Ollama call failed: {e}")
        return None


def generate_diamond_specs(
    prompts: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    attempts_per_prompt: int = 3,
    target: int = 200,
    workers: int = 2,
) -> list[DiamondResult]:
    """Generate specs via rejection sampling until we hit target Diamond count.

    For each prompt, generate up to `attempts_per_prompt` specs at increasing
    temperatures. Keep only Diamond-passing specs.
    """
    diamond_results: list[DiamondResult] = []
    all_results: list[DiamondResult] = []
    diamond_prompts: set[str] = set()  # prompts that already produced a diamond

    # Load existing diamond specs to avoid regenerating
    if _DIAMOND_OUT.exists():
        with open(_DIAMOND_OUT) as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        pid = obj.get("_prompt_id", "")
                        if pid:
                            diamond_prompts.add(pid)
                    except json.JSONDecodeError:
                        pass
        if diamond_prompts:
            log.info(f"Skipping {len(diamond_prompts)} prompts with existing Diamond specs")

    # Filter out prompts that already have diamond specs
    remaining = [p for p in prompts if p["id"] not in diamond_prompts]
    log.info(f"Generating from {len(remaining)} prompts, target={target} diamond specs, "
             f"model={model}, {attempts_per_prompt} attempts each, {workers} workers")

    total_generated = 0
    total_diamond = len(diamond_prompts)

    for prompt in remaining:
        if total_diamond >= target:
            break

        pid = prompt["id"]
        prompt_text = prompt["text"]
        best_result = None

        for attempt in range(attempts_per_prompt):
            if total_diamond >= target:
                break

            # Increase temperature on retries for diversity
            temp = 0.2 + (attempt * 0.2)

            raw = _call_ollama(prompt_text, model=model, temperature=temp)
            total_generated += 1

            if not raw:
                continue

            spec = _extract_spec(raw)
            if not spec:
                continue

            r = validate_diamond(
                spec, prompt_id=pid, prompt_text=prompt_text,
                attempt=attempt, model=model,
            )
            all_results.append(r)

            if r.is_diamond:
                diamond_results.append(r)
                total_diamond += 1
                log.info(f"  DIAMOND [{pid}] states={r.distinct_states} "
                         f"cov={r.action_coverage:.0%} mut={r.mutation_caught} "
                         f"attempt={attempt + 1} ({total_diamond}/{target})")
                break  # got diamond for this prompt, move on
            elif r.tlc_tier == "gold":
                best_result = r
                log.debug(f"  gold-not-diamond [{pid}] {r.error} attempt={attempt + 1}")
            elif r.sany_pass:
                log.debug(f"  silver [{pid}] attempt={attempt + 1}")

        if best_result and not any(dr.prompt_id == pid for dr in diamond_results):
            log.debug(f"  Best for [{pid}]: {best_result.tlc_tier} ({best_result.error})")

        # Progress log every 10 prompts
        if (remaining.index(prompt) + 1) % 10 == 0:
            log.info(f"Progress: {remaining.index(prompt) + 1}/{len(remaining)} prompts, "
                     f"{total_diamond}/{target} diamond, {total_generated} total generated")

    log.info(f"Generation complete: {total_generated} generated → {len(diamond_results)} diamond "
             f"({len(diamond_results)/max(total_generated,1):.1%} yield)")

    return diamond_results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_diamond_sft(results: list[DiamondResult], append: bool = True):
    """Save Diamond specs in SFT training format."""
    from src.training.dataset_builder import _DEVELOPER_PROMPT

    mode = "a" if append else "w"
    count = 0

    with open(_DIAMOND_OUT, mode) as f:
        for r in results:
            if not r.is_diamond:
                continue
            row = {
                "_tier": "diamond",
                "_prompt_id": r.prompt_id,
                "_source": f"diamond_gen/{r.model}",
                "_timestamp": datetime.now().isoformat(),
                "_semantic": {
                    "distinct_states": r.distinct_states,
                    "action_coverage": r.action_coverage,
                    "mutation_caught": r.mutation_caught,
                    "invariants_checked": r.invariants_checked,
                },
                "messages": [
                    {"role": "developer", "content": _DEVELOPER_PROMPT},
                    {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{r.prompt_text}"},
                    {"role": "assistant", "channel": "analysis",
                     "content": "I'll write a verified TLA+ specification with meaningful invariants that constrain behavior."},
                    {"role": "assistant", "channel": "final", "content": r.spec.strip()},
                ],
            }
            f.write(json.dumps(row) + "\n")
            count += 1

    log.info(f"Saved {count} Diamond specs to {_DIAMOND_OUT}")


def save_audit_log(results: list[DiamondResult]):
    """Save full audit results for analysis."""
    _DIAMOND_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_DIAMOND_LOG, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r), default=str) + "\n")
    log.info(f"Audit log: {_DIAMOND_LOG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diamond-tier rejection sampling for TLA+ SFT")
    parser.add_argument("--audit-only", action="store_true",
                        help="Only audit existing specs, no generation")
    parser.add_argument("--generate", action="store_true",
                        help="Generate new specs via rejection sampling")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model for generation (default: {DEFAULT_MODEL})")
    parser.add_argument("--target", type=int, default=200,
                        help="Target number of Diamond specs (default: 200)")
    parser.add_argument("--attempts", type=int, default=5,
                        help="Attempts per prompt (default: 5)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel generation workers (default: 2)")
    args = parser.parse_args()

    if not args.audit_only and not args.generate:
        args.audit_only = True  # default to audit

    # Step 1: Audit existing specs
    log.info("=" * 60)
    log.info("PHASE 1: Audit existing specs against Diamond gate")
    log.info("=" * 60)
    audit_results = audit_existing_specs()
    save_audit_log(audit_results)

    # Save any existing diamond specs
    existing_diamond = [r for r in audit_results if r.is_diamond]
    if existing_diamond:
        save_diamond_sft(existing_diamond, append=False)
        log.info(f"Found {len(existing_diamond)} existing Diamond specs!")

    if args.audit_only:
        return

    # Step 2: Generate new Diamond specs
    current_diamond = len(existing_diamond)
    remaining_target = max(0, args.target - current_diamond)

    if remaining_target == 0:
        log.info(f"Already have {current_diamond} Diamond specs, target={args.target} met!")
        return

    log.info("=" * 60)
    log.info(f"PHASE 2: Generate {remaining_target} more Diamond specs (have {current_diamond})")
    log.info("=" * 60)

    prompts = load_prompts()
    gen_results = generate_diamond_specs(
        prompts,
        model=args.model,
        attempts_per_prompt=args.attempts,
        target=args.target,  # total target including existing
        workers=args.workers,
    )

    if gen_results:
        save_diamond_sft(gen_results, append=True)

    # Final summary
    total_diamond = current_diamond + len(gen_results)
    log.info("=" * 60)
    log.info(f"FINAL: {total_diamond} Diamond specs total "
             f"({current_diamond} existing + {len(gen_results)} generated)")
    log.info(f"Output: {_DIAMOND_OUT}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
