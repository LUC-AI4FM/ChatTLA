#!/usr/bin/env python3
"""
teacher_gold_gen.py — Use a large teacher model (via Ollama) to generate verified
gold TLA+ specs with English annotations.

The teacher generates specs for benchmark problems and prompt bank entries,
validates them through SANY+TLC, and writes gold-tier examples to
data/processed/teacher_gold.jsonl in the standard ChatML training format.

Env vars:
    OLLAMA_TEACHER_URL    — Ollama base URL (default: http://localhost:11434)
    OLLAMA_TEACHER_MODEL  — model tag (default: llama3:70b)

Usage (standalone):
    python -m scripts.teacher_gold_gen --max-prompts 10

Usage (from rl_loop.py):
    from scripts.teacher_gold_gen import harvest_teacher_gold
    n_new = harvest_teacher_gold(cycle_id=42, max_prompts=5)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("teacher_gold")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEACHER_GOLD_JSONL = _REPO_ROOT / "data" / "processed" / "teacher_gold.jsonl"
_BENCHMARK_JSON = _REPO_ROOT / "data" / "benchmarks" / "benchmark_suite.json"
_PROMPT_BANK = _REPO_ROOT / "data" / "processed" / "prompt_bank.json"

# Defaults — overridden by env vars
# If TEACHER_USE_HF=1, use HuggingFace Inference API instead of local Ollama.
# This avoids loading a 60GB+ model locally and frees GPU VRAM for training.
TEACHER_URL = os.environ.get("OLLAMA_TEACHER_URL", "http://localhost:11434")
TEACHER_MODEL = os.environ.get("OLLAMA_TEACHER_MODEL", "llama3:70b")
TEACHER_USE_HF = os.environ.get("TEACHER_USE_HF", "").strip().lower() in ("1", "true", "yes")
TEACHER_HF_MODEL = os.environ.get("TEACHER_HF_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")


def _spec_hash(spec: str) -> str:
    return hashlib.sha256(spec.strip().encode()).hexdigest()[:16]


def _load_existing_hashes() -> set[str]:
    """Load hashes of already-generated teacher gold specs to avoid duplicates."""
    hashes: set[str] = set()
    if _TEACHER_GOLD_JSONL.exists():
        with open(_TEACHER_GOLD_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    msgs = obj.get("messages", [])
                    for m in msgs:
                        if m.get("channel") == "final":
                            hashes.add(_spec_hash(m["content"]))
                except Exception:
                    pass
    return hashes


_TEACHER_SYSTEM_PROMPT = (
    "You are an expert TLA+ formal methods engineer. Generate complete, "
    "syntactically correct TLA+ specifications that pass both SANY (parser) "
    "and TLC (model checker) validation. Always include:\n"
    "- EXTENDS Integers, Sequences, FiniteSets (as needed)\n"
    "- CONSTANTS and VARIABLES declarations\n"
    "- TypeOK invariant\n"
    "- Init and Next operators\n"
    "- Domain-specific safety invariants\n"
    "- A TLC configuration comment block at the end\n\n"
    "Output ONLY the TLA+ module. No markdown fences, no explanation outside the module."
)


def _call_hf_inference(
    prompt: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: int = 120,
) -> Optional[str]:
    """Call HuggingFace Inference API. Requires HF_TOKEN env var."""
    import requests

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        log.warning("[teacher] HF_TOKEN not set, cannot use HF inference")
        return None

    api_url = f"https://router.huggingface.co/hf-inference/models/{TEACHER_HF_MODEL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": TEACHER_HF_MODEL,
        "messages": [
            {"role": "system", "content": _TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning(f"[teacher] HF inference failed: {e}")
        return None


def _call_ollama(
    prompt: str,
    *,
    url: str = "",
    model: str = "",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: int = 180,
) -> Optional[str]:
    """Call local Ollama teacher model."""
    import requests

    base = url or TEACHER_URL
    mdl = model or TEACHER_MODEL

    try:
        resp = requests.post(
            f"{base}/api/generate",
            json={
                "model": mdl,
                "system": _TEACHER_SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        log.warning(f"[teacher] Ollama call failed: {e}")
        return None


def call_teacher_model(
    prompt: str,
    *,
    url: str = "",
    model: str = "",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: int = 180,
) -> Optional[str]:
    """Call teacher model via HF Inference API (if TEACHER_USE_HF=1) or local Ollama."""
    if TEACHER_USE_HF:
        return _call_hf_inference(prompt, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    return _call_ollama(prompt, url=url, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)


def extract_tla_block(text: str) -> str:
    """Extract TLA+ module from response, stripping markdown fences if present."""
    if not text:
        return ""
    # Remove markdown code fences
    text = re.sub(r"```(?:tla\+?|TLA\+?)?\s*\n?", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    # Find the module boundaries
    m_start = re.search(r"-{4,}\s*MODULE\s+\w+\s*-{4,}", text)
    m_end = text.rfind("====")
    if m_start and m_end > m_start.start():
        return text[m_start.start(): m_end + 4].strip()
    return text.strip()


def generate_and_validate_teacher_spec(
    description: str,
    prompt_id: str,
    *,
    n_attempts: int = 3,
    url: str = "",
    model: str = "",
) -> Optional[dict]:
    """
    Generate a spec with the teacher, validate with SANY+TLC.
    Returns a training-format dict if gold, else None.
    """
    from src.validators.sany_validator import validate_string as sany_validate
    from src.validators.tlc_validator import validate_string as tlc_validate
    from src.training.dataset_builder import _DEVELOPER_PROMPT

    prompt_text = (
        f"Write a TLA+ specification for the following:\n\n{description}\n\n"
        "Include a TypeOK invariant and at least one domain-specific safety invariant. "
        "Ensure the state space is bounded so TLC can verify the spec."
    )

    for attempt in range(n_attempts):
        temp = 0.1 + (attempt * 0.15)  # slightly increase temp on retries
        raw = call_teacher_model(prompt_text, url=url, model=model, temperature=temp)
        if not raw:
            continue

        spec = extract_tla_block(raw)
        if not spec or "MODULE" not in spec:
            continue

        # Extract module name
        mod_match = re.search(r"MODULE\s+(\w+)", spec)
        mod_name = mod_match.group(1) if mod_match else "Temp"

        # SANY validation
        try:
            sany_result = sany_validate(spec, module_name=mod_name)
        except Exception as e:
            log.debug(f"[teacher] SANY error: {e}")
            continue

        if not sany_result.valid:
            log.debug(f"[teacher] SANY fail for {prompt_id} attempt {attempt + 1}")
            continue

        # TLC validation
        try:
            tlc_result = tlc_validate(spec, module_name=mod_name, timeout=60)
        except Exception as e:
            log.debug(f"[teacher] TLC error: {e}")
            continue

        if tlc_result.tier != "gold":
            log.debug(f"[teacher] TLC fail for {prompt_id} attempt {attempt + 1} (tier={tlc_result.tier})")
            continue

        # Gold! Build the training example
        # Extract a brief design rationale from the teacher response (if any text before MODULE)
        pre_module = raw[:raw.find("----")] if "----" in raw else ""
        analysis = pre_module.strip() if pre_module.strip() else (
            f"I'll write a verified TLA+ specification for: {description[:200]}"
        )

        return {
            "_tier": "teacher_gold",
            "_prompt_id": prompt_id,
            "_source": "teacher",
            "_model": model or TEACHER_MODEL,
            "_timestamp": datetime.now().isoformat(),
            "_description": description,  # English annotation for English->TLA+ mapping
            "messages": [
                {"role": "developer", "content": _DEVELOPER_PROMPT},
                {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{description}"},
                {"role": "assistant", "channel": "analysis", "content": analysis[:500]},
                {"role": "assistant", "channel": "final", "content": spec.strip()},
            ],
        }

    return None


def harvest_teacher_gold(
    cycle_id: int = 0,
    max_prompts: int = 5,
    url: str = "",
    model: str = "",
) -> int:
    """
    Run teacher gold generation for a batch of prompts.
    Returns the number of new gold examples generated.
    """
    existing_hashes = _load_existing_hashes()
    new_gold = 0

    # Load benchmark problems
    prompts_to_try: list[tuple[str, str]] = []  # (prompt_id, description)

    if _BENCHMARK_JSON.exists():
        benchmarks = json.loads(_BENCHMARK_JSON.read_text())
        for bm in benchmarks:
            pid = bm.get("id", bm.get("name", ""))
            desc = bm.get("description", "")
            hints = bm.get("hints", "")
            full_desc = f"{bm.get('name', '')}: {desc}"
            if hints:
                full_desc += f"\n\nHints: {hints}"
            prompts_to_try.append((f"T_{pid}", full_desc))

    # Also load prompt bank if available
    if _PROMPT_BANK.exists():
        try:
            bank = json.loads(_PROMPT_BANK.read_text())
            for p in bank:
                pid = p.get("id", "")
                desc = p.get("description", p.get("prompt", ""))
                if pid and desc:
                    prompts_to_try.append((f"T_{pid}", desc))
        except Exception:
            pass

    # Shuffle and limit
    import random
    random.shuffle(prompts_to_try)
    prompts_to_try = prompts_to_try[:max_prompts]

    log.info(f"[teacher] Harvesting gold from {len(prompts_to_try)} prompts "
             f"(teacher={model or TEACHER_MODEL} @ {url or TEACHER_URL})")

    _TEACHER_GOLD_JSONL.parent.mkdir(parents=True, exist_ok=True)

    for pid, desc in prompts_to_try:
        result = generate_and_validate_teacher_spec(
            desc, pid, url=url, model=model,
        )
        if result is None:
            log.debug(f"[teacher] No gold for {pid}")
            continue

        # Dedup
        spec_text = ""
        for m in result.get("messages", []):
            if m.get("channel") == "final":
                spec_text = m["content"]
        h = _spec_hash(spec_text)
        if h in existing_hashes:
            log.debug(f"[teacher] Duplicate spec for {pid}, skipping")
            continue

        existing_hashes.add(h)
        with open(_TEACHER_GOLD_JSONL, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        new_gold += 1
        log.info(f"[teacher] Gold spec for {pid} (hash={h})")

        # Brief pause between teacher calls
        time.sleep(1)

    log.info(f"[teacher] Harvest complete: {new_gold} new gold examples")
    return new_gold


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Teacher gold generation for ChatTLA")
    parser.add_argument("--max-prompts", type=int, default=10,
                        help="Max prompts to try (default: 10)")
    parser.add_argument("--url", default="",
                        help=f"Ollama URL (default: {TEACHER_URL})")
    parser.add_argument("--model", default="",
                        help=f"Teacher model (default: {TEACHER_MODEL})")
    args = parser.parse_args()

    n = harvest_teacher_gold(
        cycle_id=0,
        max_prompts=args.max_prompts,
        url=args.url,
        model=args.model,
    )
    print(f"\nGenerated {n} new gold examples → {_TEACHER_GOLD_JSONL}")


if __name__ == "__main__":
    main()
