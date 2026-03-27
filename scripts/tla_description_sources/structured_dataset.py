"""
Structured TLA+ dataset records for regeneration training.

The model returns JSON matching the schema in STRUCTURED_SYSTEM_PROMPT.
HARVEST_BASELINE_STRATEGY documents the non-LLM harvest path; see also
data/derived/tla_descriptions_audit.json → generation_strategy.
"""

from __future__ import annotations

import json
from typing import Any, Optional

# System prompt: rules + exact output shape (model fills id/module_name/description).
STRUCTURED_SYSTEM_PROMPT = """You are a formal methods expert. Analyze the TLA+ spec and describe it as the paper's authors would. Output will be fed to an AI to regenerate the spec.

RULES
- Author voice: explain protocol intent, not just mechanics
- Inline TLA+ syntax for variables, values, transitions (no raw code blocks)
- No padding — every sentence must serve reconstruction
- Explain nondeterminism: what real behavior does it model?
- If algorithm is unnamed or PDF unavailable, infer and state so
- Output must be valid JSON

OUTPUT FORMAT
Return a single JSON object with this exact structure:
{ "id": "<module_name_lowercase>_001", "module_name": "<ExactTLAModuleName>", "description": { "narrative": "<3-5 sentence paper-style abstract: problem, protocol intuition, liveness guarantees. No variables, no syntax>", "technical": { "algorithm": "<Name — one-sentence purpose>", "constants_and_processes": "<Definitions + brief rationale>", "variables": [ { "name": "<varName>", "type": "<TLA+ type>", "role": "<one-line author-voice role>" } ], "init": "<Compact TLA+-style init, one entry per variable>", "actions": [ {"name": "<ActionName(i)>", "intent": "<one-sentence protocol purpose>", "pre": "<TLA+-style precondition>", "post": "<TLA+-style postcondition, branches where applicable>" } ], "next_and_fairness": "<Action composition + fairness justification>","invariants_and_properties": [ { "name": "<Name>", "assertion": "<TLA+-style>", "purpose": "<one-line why it matters>" } ],"critical_design_decisions": [ "<bullet: non-obvious choice and its purpose>" ] } } }

Use the exact module name given in the user message. The id must be <that_name_lower>_001.
"""

# Baseline harvest (no LLM): what fills `tla_descriptions.json` when `--llm` is off.
# Kept next to STRUCTURED_SYSTEM_PROMPT so “strategy” is one place for humans and for
# `tla_descriptions_audit.json` → `generation_strategy`.
HARVEST_BASELINE_STRATEGY = """\
Baseline dataset (no `--llm`): official sources per module in the coarse list:
- tlaplus/Examples: README.md curated titles, each folder’s manifest.json (authors, paper/PDF URLs),
  first `(* … *)` comment block after MODULE, optional PDF text (first pages via pypdf cache).
- tlaplus/tlapm: `library/*.tla` for TLAPS proof modules not shipped under Examples.

Technical fields: SANY XMLExporter on each `.tla` when possible; regex/static extraction fallback;
`--no-static-extract` disables both. Narrative merges harvested prose with SANY/static structure.
Optional `--llm` replaces the normalized `description` with output matching STRUCTURED_SYSTEM_PROMPT
(author voice, same JSON schema as rows committed without LLM but aimed at reconstruction).
"""


def dataset_record_id(module_name: str) -> str:
    return f"{module_name.lower()}_001"


def empty_technical_shell() -> dict[str, Any]:
    return {
        "algorithm": "",
        "constants_and_processes": "",
        "variables": [],
        "init": "",
        "actions": [],
        "next_and_fairness": "",
        "invariants_and_properties": [],
        "critical_design_decisions": [],
    }


def empty_description_from_harvest(prose: str, note: str = "") -> dict[str, Any]:
    """When --llm is off: narrative from harvested prose; technical left empty / noted."""
    tech = empty_technical_shell()
    if note:
        tech["algorithm"] = note
    return {"narrative": prose.strip() or "(no harvested text)", "technical": tech}


def _coerce_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


def normalize_technical(tech: Any) -> dict[str, Any]:
    """Ensure all expected keys exist; coerce list/dict shapes loosely."""
    base = empty_technical_shell()
    if not isinstance(tech, dict):
        return base
    base["algorithm"] = _coerce_str(tech.get("algorithm"))
    base["constants_and_processes"] = _coerce_str(tech.get("constants_and_processes"))
    base["init"] = _coerce_str(tech.get("init"))
    base["next_and_fairness"] = _coerce_str(tech.get("next_and_fairness"))

    vars_raw = tech.get("variables")
    if isinstance(vars_raw, list):
        out_v: list[dict[str, str]] = []
        for item in vars_raw:
            if isinstance(item, dict):
                out_v.append(
                    {
                        "name": _coerce_str(item.get("name")),
                        "type": _coerce_str(item.get("type")),
                        "role": _coerce_str(item.get("role")),
                    }
                )
        base["variables"] = out_v

    act_raw = tech.get("actions")
    if isinstance(act_raw, list):
        out_a: list[dict[str, str]] = []
        for item in act_raw:
            if isinstance(item, dict):
                out_a.append(
                    {
                        "name": _coerce_str(item.get("name")),
                        "intent": _coerce_str(item.get("intent")),
                        "pre": _coerce_str(item.get("pre")),
                        "post": _coerce_str(item.get("post")),
                    }
                )
        base["actions"] = out_a

    inv_raw = tech.get("invariants_and_properties")
    if isinstance(inv_raw, list):
        out_i: list[dict[str, str]] = []
        for item in inv_raw:
            if isinstance(item, dict):
                out_i.append(
                    {
                        "name": _coerce_str(item.get("name")),
                        "assertion": _coerce_str(item.get("assertion")),
                        "purpose": _coerce_str(item.get("purpose")),
                    }
                )
        base["invariants_and_properties"] = out_i

    crit = tech.get("critical_design_decisions")
    if isinstance(crit, list):
        base["critical_design_decisions"] = [_coerce_str(x) for x in crit]

    return base


def normalize_description(desc: Any) -> dict[str, Any]:
    """Normalize LLM or partial description object."""
    if not isinstance(desc, dict):
        return empty_description_from_harvest("", "Invalid description object.")
    nar = desc.get("narrative", "")
    if not isinstance(nar, str):
        nar = _coerce_str(nar)
    tech = normalize_technical(desc.get("technical"))
    return {"narrative": nar.strip(), "technical": tech}


def extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    """Parse first JSON object from model output (robust to preamble / trailing text)."""
    start = text.find("{")
    if start == -1:
        return None
    try:
        dec = json.JSONDecoder()
        obj, _ = dec.raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def validate_structured_description(desc: Any) -> list[str]:
    """Return list of validation error strings (empty if OK enough to store)."""
    errs: list[str] = []
    if not isinstance(desc, dict):
        return ["description is not an object"]
    if "narrative" not in desc:
        errs.append("missing description.narrative")
    if "technical" not in desc:
        errs.append("missing description.technical")
        return errs
    tech = desc["technical"]
    if not isinstance(tech, dict):
        errs.append("description.technical is not an object")
        return errs
    for key in (
        "algorithm",
        "constants_and_processes",
        "variables",
        "init",
        "actions",
        "next_and_fairness",
        "invariants_and_properties",
        "critical_design_decisions",
    ):
        if key not in tech:
            errs.append(f"missing description.technical.{key}")
    return errs


def build_llm_user_message(
    module_name: str,
    tla_text: str,
    *,
    readme_title: str = "",
    authors: Optional[list[str]] = None,
    sources: Optional[list[str]] = None,
    header_comment: str = "",
    pdf_excerpt: str = "",
    max_tla_chars: int = 28000,
) -> str:
    authors = authors or []
    sources = sources or []
    parts: list[str] = [
        f"Target MODULE name (exact): {module_name}",
        f"Dataset id must be: {dataset_record_id(module_name)}",
    ]
    if readme_title:
        parts.append(f"README / spec title: {readme_title}")
    if authors:
        parts.append("Authors: " + ", ".join(authors))
    if sources:
        parts.append("Reference URLs: " + "; ".join(sources[:8]))
    if header_comment:
        parts.append("Official module header comment:\n" + header_comment[:8000])
    if pdf_excerpt:
        parts.append("Excerpt from linked paper (PDF text, if any):\n" + pdf_excerpt[:12000])
    truncated = False
    if len(tla_text) > max_tla_chars:
        tla_show = tla_text[:max_tla_chars]
        truncated = True
    else:
        tla_show = tla_text
    parts.append("TLA+ module:\n" + tla_show)
    if truncated:
        parts.append(f"(TLA+ truncated to first {max_tla_chars} characters.)")
    return "\n\n".join(parts)


def call_ollama_structured(
    user_message: str,
    *,
    host: str,
    model: str,
    temperature: float = 0.15,
) -> tuple[Optional[dict[str, Any]], str]:
    """
    Returns (parsed_json_or_none, raw_text) from Ollama chat.
    """
    import ollama  # lazy

    client = ollama.Client(host=host)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={
            "temperature": temperature,
            "num_predict": 8192,
        },
    )
    raw = response["message"]["content"]
    parsed = extract_first_json_object(raw)
    return parsed, raw


def merge_llm_record(
    module_name: str,
    coarse_id: int,
    parsed: Optional[dict[str, Any]],
    raw_fallback: str,
) -> tuple[dict[str, Any], list[str]]:
    """
    Normalize LLM output into our description object; return (description_dict, warnings).
    """
    warns: list[str] = []
    if not parsed:
        warns.append("llm_parse_failed")
        return (
            empty_description_from_harvest(
                f"[LLM output not JSON]\n{raw_fallback[:2000]}",
                "LLM did not return valid JSON; re-run with --llm.",
            ),
            warns,
        )
    desc = parsed.get("description")
    if not isinstance(desc, dict):
        warns.append("missing_top_level_description")
        bad = empty_description_from_harvest(str(parsed)[:3000], "Malformed LLM response.")
        return bad, warns

    desc = normalize_description(desc)
    errs = validate_structured_description(desc)
    warns.extend(errs)
    _ = module_name, coarse_id  # reserved for future id enforcement
    return desc, warns
