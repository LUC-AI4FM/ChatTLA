"""Full-spec TLA+ training dataset for GRPO.

Loads NL problem descriptions from:
  1. data/diamond_gen_topics.json — 200 topics across 10 domains
  2. data/processed/diamond_sft.jsonl — diamond-tier specs (use the NL description)
  3. data/processed/train.jsonl — main training set (spec_generation tasks only)

Each prompt asks the model to write a complete TLA+ spec from scratch, matching
the deployment task. This replaces the per-action harness approach that failed
on the 20B model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEVELOPER_PROMPT = """\
You are ChatTLA, an expert at writing verified TLA+ formal specifications.
When asked to write a TLA+ spec, follow these rules exactly:
1. Start the module with ---- MODULE <ModuleName> ----
2. End with ====
3. Include EXTENDS, VARIABLES, Init, Next, and Spec operators
4. Define Spec == Init /\\ [][Next]_vars (with vars == <<v1, v2, ...>>)
5. Output only valid TLA+ code. No markdown fences, no explanation outside the spec.

TLC runtime rules (your spec MUST pass the TLC model checker):
6. CONSTANTS must be finite and enumerable. Never use \\in Nat or \\in Int unbounded — always use bounded ranges like 0..N where N is a CONSTANT.
7. Every disjunct in Next must specify ALL variables: either prime them (x' = ...) or use UNCHANGED <<x>>. A partial UNCHANGED causes TLC errors.
8. Init must assign every variable a concrete finite value. Never leave a variable unconstrained.
9. TypeOK must be an invariant that constrains every variable to a finite set (e.g., x \\in 0..N, state \\in {"idle", "active"}).
10. Avoid deadlock: if the system can terminate, add a Terminating == /\\ UNCHANGED vars disjunct to Next.
Reasoning: none\
"""


@dataclass
class FullSpecExample:
    prompt_id: str
    prompt: list[dict[str, str]]
    nl_description: str
    module_name: str
    domain: str


def _extract_nl_from_messages(messages: list[dict]) -> str | None:
    """Extract the NL description from a harmony-formatted messages list."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Strip the "Write a TLA+ specification for:" prefix if present
            content = re.sub(
                r"^Write a TLA\+\s*(formal\s+)?specification\s*(for\s*(the following)?:?\s*)?",
                "", content, flags=re.IGNORECASE,
            ).strip()
            if content:
                return content
    return None


def load_fullspec_prompts(
    include_topics: bool = True,
    include_diamond_sft: bool = True,
    include_train: bool = False,
    max_per_source: int | None = None,
) -> list[FullSpecExample]:
    """Build a list of GRPO-ready full-spec training examples.

    Parameters
    ----------
    include_topics      : Load from diamond_gen_topics.json (200 topics)
    include_diamond_sft : Load from diamond_sft.jsonl descriptions
    include_train       : Load from train.jsonl (spec_generation tasks only)
    max_per_source      : Cap per source (for memory/debug)
    """
    examples: list[FullSpecExample] = []
    seen_ids: set[str] = set()

    def _add(prompt_id: str, nl: str, module: str, domain: str) -> None:
        if prompt_id in seen_ids:
            return
        seen_ids.add(prompt_id)
        # gpt-oss harmony format uses "developer" role, not "system".
        # "system" injects an unwanted GPT preamble via the chat template.
        prompt = [
            {"role": "developer", "content": _DEVELOPER_PROMPT},
            {"role": "user", "content": f"Write a TLA+ specification for the following:\n\n{nl}"},
        ]
        examples.append(FullSpecExample(
            prompt_id=prompt_id,
            prompt=prompt,
            nl_description=nl,
            module_name=module,
            domain=domain,
        ))

    # Source 1: 200 topics from diamond_gen_topics.json
    if include_topics:
        topics_path = _REPO_ROOT / "data" / "diamond_gen_topics.json"
        if topics_path.is_file():
            data = json.loads(topics_path.read_text(encoding="utf-8"))
            count = 0
            for batch in data.get("batches", []):
                domain = ""
                for key in batch:
                    if key != "topics":
                        domain = batch[key] if isinstance(batch[key], str) else key
                        break
                for topic in batch.get("topics", []):
                    module = topic.get("module", "Unknown")
                    desc = topic.get("desc", "")
                    if desc:
                        _add(f"topic_{module}", desc, module, domain)
                        count += 1
                        if max_per_source and count >= max_per_source:
                            break
                if max_per_source and count >= max_per_source:
                    break

    # Source 2: diamond SFT specs (use the NL description, not the spec)
    if include_diamond_sft:
        for name in ("diamond_sft.jsonl", "diamond_curated.jsonl"):
            sft_path = _REPO_ROOT / "data" / "processed" / name
            if not sft_path.is_file():
                continue
            count = 0
            with sft_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    messages = row.get("messages", [])
                    nl = _extract_nl_from_messages(messages)
                    if not nl:
                        continue
                    pid = row.get("_prompt_id", row.get("module", f"dsft_{count}"))
                    module = row.get("module", "Spec")
                    if not module or module == "Spec":
                        m = re.search(r"MODULE\s+(\w+)", str(row.get("spec", "")))
                        module = m.group(1) if m else f"Spec{count}"
                    _add(f"dsft_{pid}", nl, module, row.get("batch", "unknown"))
                    count += 1
                    if max_per_source and count >= max_per_source:
                        break

    # Source 3: main training set (spec_generation tasks only)
    if include_train:
        train_path = _REPO_ROOT / "data" / "processed" / "train.jsonl"
        if train_path.is_file():
            count = 0
            with train_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Only spec_generation tasks
                    task = row.get("_task", "spec_generation")
                    if task != "spec_generation":
                        continue
                    messages = row.get("messages", [])
                    nl = _extract_nl_from_messages(messages)
                    if not nl:
                        continue
                    pid = row.get("_prompt_id", f"train_{count}")
                    module = "Spec"
                    m = re.search(r"MODULE\s+(\w+)", str(row))
                    if m:
                        module = m.group(1)
                    _add(f"train_{pid}", nl, module, row.get("_domain", "unknown"))
                    count += 1
                    if max_per_source and count >= max_per_source:
                        break

    return examples
