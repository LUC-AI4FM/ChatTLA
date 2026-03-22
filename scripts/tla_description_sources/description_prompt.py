"""
Shared helpers: condense tla_descriptions.json rows for prompts / SFT user text.
"""

from __future__ import annotations

import re
from typing import Any, Optional


def strip_source_context(narrative: str) -> str:
    """Remove appended harvest block so RL prompts stay shorter."""
    if "[Source context for reconstruction" in narrative:
        narrative = narrative.split("[Source context for reconstruction", 1)[0].strip()
    return narrative.strip()


def condense_description_row(
    row: dict[str, Any],
    *,
    max_narrative_chars: int = 1200,
    max_actions: int = 8,
    max_next_chars: int = 1200,
    max_init_chars: int = 800,
) -> str:
    """
    Build a single user-facing block from one tla_descriptions.json row
    (description.narrative + description.technical).
    """
    desc = row.get("description") or {}
    if isinstance(desc, str):
        return desc[: max_narrative_chars * 2]

    nar = strip_source_context(str(desc.get("narrative", "")))
    if len(nar) > max_narrative_chars:
        nar = nar[: max_narrative_chars - 3] + "..."

    tech = desc.get("technical") or {}
    if not isinstance(tech, dict):
        return nar

    parts: list[str] = [f"## Protocol / intent\n{nar}"]

    algo = str(tech.get("algorithm", "")).strip()
    if algo:
        parts.append(f"## Algorithm (title)\n{algo}")

    consts = str(tech.get("constants_and_processes", "")).strip()
    if consts:
        parts.append(f"## Constants / parameters\n{consts[:600]}")

    vars_ = tech.get("variables")
    if isinstance(vars_, list) and vars_:
        names = []
        for v in vars_[:40]:
            if isinstance(v, dict) and v.get("name"):
                names.append(str(v["name"]))
            elif isinstance(v, str):
                names.append(v)
        if names:
            parts.append("## State variables\n" + ", ".join(names))

    init = str(tech.get("init", "")).strip()
    if init:
        if len(init) > max_init_chars:
            init = init[: max_init_chars - 3] + "..."
        parts.append(f"## Init (SANY-style summary)\n{init}")

    actions = tech.get("actions")
    if isinstance(actions, list) and actions:
        lines = []
        for a in actions[:max_actions]:
            if not isinstance(a, dict):
                continue
            nm = str(a.get("name", ""))
            intent = str(a.get("intent", "")).replace("\n", " ")[:200]
            post = str(a.get("post", "")).replace("\n", " ")[:300]
            if nm:
                lines.append(f"- **{nm}**: {intent}\n  transition sketch: `{post}`")
        if lines:
            parts.append("## Actions (names + intent)\n" + "\n".join(lines))

    nxf = str(tech.get("next_and_fairness", "")).strip()
    if nxf:
        if len(nxf) > max_next_chars:
            nxf = nxf[: max_next_chars - 3] + "..."
        parts.append(f"## Next / Spec / fairness (summary)\n{nxf}")

    invs = tech.get("invariants_and_properties")
    if isinstance(invs, list) and invs:
        inv_names = []
        for inv in invs[:12]:
            if isinstance(inv, dict) and inv.get("name"):
                inv_names.append(str(inv["name"]))
        if inv_names:
            parts.append("## Key invariants / properties (names)\n" + ", ".join(inv_names))

    return "\n\n".join(parts).strip()


def benchmark_context_block(
    descriptions_by_module: dict[str, dict[str, Any]],
    module_name: str,
    *,
    max_chars: int = 4500,
) -> Optional[str]:
    """Return condensed text for injection, or None if module missing."""
    row = descriptions_by_module.get(module_name)
    if not row:
        return None
    text = condense_description_row(row)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "\n… [truncated]"
    return text


def load_descriptions_index(path) -> dict[str, dict[str, Any]]:
    """module_name -> full row."""
    import json
    from pathlib import Path

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in data:
        mn = row.get("module_name")
        if mn:
            out[str(mn)] = row
    return out
