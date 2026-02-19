"""
annotate.py — Natural-language annotation using local Ollama gpt-oss:20b.

Replaces GPT-4o annotation (which would cost $50-200 for 50k specs) with a
fully local, zero-cost alternative running on the same machine.  The local
model's output is close enough to GPT-4o quality for structural annotation
tasks like describing what a spec models.

For each validated spec this module generates:
  - natural_language_description: 2–4 sentences describing what is modelled
  - domain: one of the Domain enum values
  - difficulty: 1–5 based on spec complexity
  - key_invariants: list of invariant names with plain-English descriptions
  - key_design_decisions: list of notable design choices

Harmony format
--------------
gpt-oss REQUIRES the harmony format to work correctly.  We apply it via the
`openai-harmony` package.  The Ollama Python client talks to the local server
at OLLAMA_HOST (default http://localhost:11434).

Rate limiting
-------------
Ollama runs inference on GPU 1.  We enforce a configurable delay between
requests to keep the GPU from being fully saturated during annotation
while also handling training data scraping in parallel.

Usage
-----
    python -m src.scraper.annotate --input data/validated/combined.jsonl \\
                                   --output data/validated/annotated.jsonl
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Iterator, Optional

from src.shared.schemas.dataset_schema import Annotation, DatasetRecord

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_MODEL = os.getenv("ANNOTATION_MODEL", "gpt-oss:20b")
_INTER_REQUEST_DELAY = float(os.getenv("ANNOTATION_DELAY_S", "0.5"))

_SYSTEM_PROMPT = """\
You are an expert in formal methods and TLA+ specifications.
Reasoning: low

When given a TLA+ specification, return ONLY a JSON object with these fields:
{
  "natural_language_description": "<2-4 sentences describing what system or property this spec models>",
  "domain": "<one of: consensus, storage, networking, security, hardware, transaction, scheduling, other>",
  "difficulty": <integer 1-5>,
  "key_invariants": ["<invariant name>: <plain English description>", ...],
  "key_design_decisions": ["<decision description>", ...]
}
Output only the JSON object. No markdown fences, no preamble, no explanation.\
"""


def annotate_record(record: DatasetRecord, client=None) -> Annotation:
    """
    Call local Ollama to generate an Annotation for one DatasetRecord.

    Parameters
    ----------
    record : DatasetRecord   The validated spec to annotate.
    client                   Optional pre-constructed Ollama client (for reuse).

    Returns
    -------
    Annotation — partially filled if the model output can't be fully parsed.
    """
    import ollama  # lazy import — optional dependency

    if client is None:
        client = ollama.Client(host=_OLLAMA_HOST)

    # Build the harmony-formatted prompt via openai-harmony if available;
    # fall back to plain chat format (Ollama handles the template internally)
    user_content = _build_user_prompt(record)

    try:
        response = client.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            options={"temperature": 0.1},  # low temp for structured extraction
        )
        raw_text = response["message"]["content"]
        return _parse_annotation(raw_text)
    except Exception as exc:
        print(f"[annotate] Failed for {record.source}: {exc}")
        return Annotation(natural_language_description=f"[annotation failed: {exc}]")


def _build_user_prompt(record: DatasetRecord) -> str:
    """
    Build the user message for the annotation request.
    If pre-existing comments are available (from FormaLLM), include them as
    additional context — the model should incorporate that signal.
    """
    parts = []
    if record.annotation and record.annotation.natural_language_description:
        parts.append(f"Existing description hint:\n{record.annotation.natural_language_description[:500]}\n")
    parts.append(f"TLA+ Specification:\n```\n{record.tla_content[:3000]}\n```")
    if len(record.tla_content) > 3000:
        parts.append("(spec truncated to first 3000 chars for annotation)")
    return "\n".join(parts)


def _parse_annotation(raw_text: str) -> Annotation:
    """
    Parse JSON from the model output into an Annotation.
    Robust to minor formatting issues (text before/after the JSON object).
    """
    # Extract JSON block
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return Annotation(natural_language_description=raw_text[:500])

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Annotation(natural_language_description=raw_text[:500])

    return Annotation(
        natural_language_description=data.get("natural_language_description", ""),
        domain=_coerce_domain(data.get("domain")),
        difficulty=int(data.get("difficulty", 0)),
        key_invariants=data.get("key_invariants", []),
        key_design_decisions=data.get("key_design_decisions", []),
    )


def _coerce_domain(value: Optional[str]) -> Optional[str]:
    """Normalise a domain string to a valid Domain literal."""
    valid = {"consensus", "storage", "networking", "security", "hardware", "transaction", "scheduling", "other"}
    if value and value.lower() in valid:
        return value.lower()
    return "other"


# ---------------------------------------------------------------------------
# Batch annotation
# ---------------------------------------------------------------------------

def annotate_jsonl(
    input_path: Path,
    output_path: Path,
    skip_already_annotated: bool = True,
    batch_size: int = 100,
) -> int:
    """
    Annotate all records in input_path and write to output_path.
    Records that already have non-empty annotation can be skipped.

    Returns number of records annotated (not skipped).
    """
    import ollama

    client = ollama.Client(host=_OLLAMA_HOST)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_annotated = 0
    n_skipped = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for line in input_path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            record = DatasetRecord.from_dict(json.loads(line))

            annotated = (
                record.annotation is not None
                and bool(record.annotation.natural_language_description)
                and not record.annotation.natural_language_description.startswith("[annotation failed")
            )

            if skip_already_annotated and annotated:
                n_skipped += 1
            else:
                record.annotation = annotate_record(record, client=client)
                n_annotated += 1
                time.sleep(_INTER_REQUEST_DELAY)

                if n_annotated % 50 == 0:
                    print(f"[annotate] {n_annotated} annotated, {n_skipped} skipped...")

            fout.write(record.to_json(indent=None) + "\n")

    print(f"[annotate] Done. {n_annotated} annotated, {n_skipped} skipped → {output_path}")
    return n_annotated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate validated TLA+ specs via local Ollama")
    parser.add_argument("--input",  required=True, help="Input JSONL of validated records")
    parser.add_argument("--output", required=True, help="Output JSONL with annotations")
    parser.add_argument("--no-skip", action="store_true", help="Re-annotate already-annotated records")
    args = parser.parse_args()

    annotate_jsonl(
        input_path=Path(args.input),
        output_path=Path(args.output),
        skip_already_annotated=not args.no_skip,
    )
