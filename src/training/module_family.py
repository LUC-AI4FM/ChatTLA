"""
TLA+ module topology helpers for training data hygiene.

Many tlaplus/Examples (and FormaLLM) specs exist as a *family*:

  Consensus.tla          — full abstract spec
  MCConsensus.tla        — tiny TLC wrapper: EXTENDS Consensus, TLC + CONSTANTS

Training on MC* shims as if they were the full protocol poisons SFT:
the user/description asks for a deep protocol, but the label is a short
INSTANCE/MV-constant shell.  This module detects those shims so dataset
builders can skip them (train on the parent module’s file instead).

This is intentionally heuristic — not a full PlusCal dependency graph.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Standard / library modules commonly listed on EXTENDS lines (not “the” protocol).
_STD_EXTENDS: frozenset[str] = frozenset({
    "Integers", "Naturals", "Reals", "Sequences", "FiniteSets", "Bags",
    "TLC", "TLCExt", "TLAPS", "TLAPSExt", "RealTime", "Randomization",
    "Functions", "SequencesExt", "FiniteSetsExt", "Folds", "BagsExt",
    "Json", "IOUtils", "DyadicRationals", "SVG", "FiniteSetTheorems",
    "NaturalsInduction", "WellFoundedInduction", "Toolbox", "MCTest",
})


def parse_module_name(tla: str) -> str | None:
    m = re.search(r"----\s*MODULE\s+(\w+)\s*----", tla, re.IGNORECASE)
    return m.group(1) if m else None


def _first_extends_line(tla: str) -> str | None:
    for line in tla.splitlines():
        s = line.split("\\*", 1)[0].strip()
        if re.match(r"^(?:LOCAL\s+)?EXTENDS\b", s, re.IGNORECASE):
            return s
    return None


def _tokens_from_extends_rest(rest: str) -> list[str]:
    """Comma-separated module names after EXTENDS … (one logical line)."""
    rest = rest.strip().split("\\*", 1)[0].strip()
    parts = [p.strip() for p in rest.replace("\n", " ").split(",")]
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        tok = p.split()[0]
        if tok.upper() in ("LOCAL", "INSTANCE", "MODULE"):
            break
        out.append(tok)
    return out


def parse_all_extends_modules(tla: str) -> tuple[str, ...]:
    """
    Every module named on any EXTENDS line (including LOCAL EXTENDS), in order.
    Heuristic line-based scan; does not join broken lines where EXTENDS continues
    on the next physical line without a keyword.
    """
    names: list[str] = []
    for line in tla.splitlines():
        s = line.split("\\*", 1)[0].strip()
        m = re.match(r"^(?:LOCAL\s+)?EXTENDS\s+(.+)$", s, re.IGNORECASE)
        if not m:
            continue
        names.extend(_tokens_from_extends_rest(m.group(1)))
    return tuple(names)


@lru_cache(maxsize=1024)
def parse_extends_modules(tla: str) -> tuple[str, ...]:
    """
    Modules named on the first EXTENDS line (comma-separated).
    Ignores comment-only lines; does not follow INSTANCE.
    """
    line = _first_extends_line(tla)
    if not line:
        return ()
    m = re.match(r"^(?:LOCAL\s+)?EXTENDS\s+(.+)$", line, re.IGNORECASE)
    if not m:
        return ()
    return tuple(_tokens_from_extends_rest(m.group(1)))


def _scrub_line_comments(tla: str) -> str:
    return "\n".join(line.split("\\*", 1)[0] for line in tla.splitlines())


def parse_instance_module_names(tla: str) -> tuple[str, ...]:
    """
    Module names appearing after INSTANCE (heuristic word scan on \\* stripped lines).
    Does not parse WITH-clauses or parameterized INSTANCES like INSTANCE M!Foo.
    """
    scrubbed = _scrub_line_comments(tla)
    found = re.findall(r"\bINSTANCE\s+(\w+)\b", scrubbed, flags=re.IGNORECASE)
    return tuple(found)


def referenced_protocol_modules(tla: str) -> frozenset[str]:
    """
    Non-library modules referenced via EXTENDS or INSTANCE — typically live in other .tla files.
    """
    refs: set[str] = set()
    for m in parse_all_extends_modules(tla):
        if m not in _STD_EXTENDS:
            refs.add(m)
    for m in parse_instance_module_names(tla):
        if m not in _STD_EXTENDS:
            refs.add(m)
    return frozenset(refs)


def missing_context_module_names(
    tla: str,
    *,
    defined_modules: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """
    Protocol-level modules referenced in this snippet whose definitions are not
    in the provided bundle. When ``defined_modules`` is None, only the primary
    ``---- MODULE Name ----`` is assumed present (single-file agent view).
    """
    primary = parse_module_name(tla)
    if defined_modules is not None:
        defined = defined_modules
    else:
        defined = frozenset({primary}) if primary else frozenset()
    missing = sorted(referenced_protocol_modules(tla) - defined)
    return tuple(missing)


def format_spec_context_gap_notice(
    tla: str,
    *,
    defined_modules: frozenset[str] | None = None,
) -> str | None:
    """
    Short markdown block to prepend when an LLM sees a single .tla file so it does
    not invent definitions from unseen EXTENDS / INSTANCE targets. Returns None if
    there is nothing to flag.
    """
    miss = missing_context_module_names(tla, defined_modules=defined_modules)
    if not miss:
        return None
    listed = ", ".join(miss)
    return (
        "## Context gap (do not invent hidden definitions)\n\n"
        "This snippet references the following module(s) via **EXTENDS** or **INSTANCE**, "
        "but their **full definitions are not included** here. "
        "Do not assume operators, CONSTANTS, or VARIABLES from them unless stated in this file; "
        "treat them as opaque imports, ask for the other file(s), or say you cannot see them.\n\n"
        f"**Referenced but not in view:** {listed}"
    )


def _non_comment_non_blank_lines(tla: str) -> int:
    n = 0
    for line in tla.splitlines():
        s = line.strip()
        if not s or s.startswith("\\*") or s.startswith("(*"):
            continue
        n += 1
    return n


def mc_stripped_core_name(module_name: str) -> str | None:
    """
    If module_name is `MCFoo` with Foo starting with uppercase, return `Foo`.
    Otherwise None (not an MC-prefixed convention we handle).
    """
    if len(module_name) < 4 or not module_name.startswith("MC"):
        return None
    rest = module_name[2:]
    if not rest or not rest[0].isupper():
        return None
    return rest


def is_model_check_shim(module_name: str | None, tla: str) -> bool:
    """
    True if this looks like a TLC model-checking wrapper around another module.

    Pattern: MODULE MC<Core> extends <Core> (plus TLC/libs), few concrete lines.
    Example: MCConsensus EXTENDS Consensus, TLC — label is useless for “write Paxos”.

    Secondary pattern: MCKVS extends KeyValueStore (name prefix does not match) but file
    is a tiny TLC shell — still poison for description→spec SFT.
    """
    if not module_name:
        module_name = parse_module_name(tla)
    if not module_name:
        return False
    if not module_name.startswith("MC") or len(module_name) < 4:
        return False

    extended = parse_extends_modules(tla)
    userish = [m for m in extended if m not in _STD_EXTENDS and m != "TLC"]
    nlines = _non_comment_non_blank_lines(tla)

    # Large MC* modules may embed real modeling (keep them)
    if nlines > 160:
        return False

    core = mc_stripped_core_name(module_name)

    # Primary: MC<Core> extends …, <Core>, … (e.g. MCCRDT extends CRDT)
    if core and core in userish:
        return True

    # Fallback: tiny TLC-tagged MC* shell where the protocol lives in another module
    # (e.g. MCKVS extends KeyValueStore — “KVS” ≠ “KeyValueStore”).
    if "TLC" in extended and nlines <= 15 and userish:
        return True

    return False


def family_dir_from_source(source: str, metadata: dict | None) -> str | None:
    """
    Best-effort directory key for grouping sibling .tla from one example tree.
    """
    meta = metadata or {}
    tla_file = meta.get("tla_file") or meta.get("path")
    if isinstance(tla_file, str) and "/" in tla_file:
        p = tla_file.rsplit("/", 1)[0]
        return p
    if source.startswith("github:") or source.startswith("formalllm:"):
        parts = source.split(":")
        if len(parts) >= 2:
            return ":".join(parts[:-1])
    return None
