#!/usr/bin/env python3
"""build_tlaplus_examples_sft.py — Fork A corpus builder.

Reads the labeled tlaplus/examples dump produced by the scraper
(data/processed/tlaplus_examples_labeled.jsonl) and emits two
validator-segregated SFT corpora:

  data/processed/tlc_target_sft.jsonl     — specs that pass TLC
  data/processed/tlaps_target_sft.jsonl   — specs that have verified TLAPS proofs

Excludes any spec whose module name collides with the 30 diamond_eval_holdout
modules so downstream evals remain unseen.

Also writes a per-corpus summary JSON with counts and source-spec manifests.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LABELED = _REPO_ROOT / "data" / "processed" / "tlaplus_examples_labeled.jsonl"
_HOLDOUT = _REPO_ROOT / "data" / "processed" / "diamond_eval_holdout.jsonl"
_OUT_TLC = _REPO_ROOT / "data" / "processed" / "tlc_target_sft.jsonl"
_OUT_TLAPS = _REPO_ROOT / "data" / "processed" / "tlaps_target_sft.jsonl"
_SUMMARY = _REPO_ROOT / "data" / "processed" / "tlaplus_examples_sft_summary.json"

DEVELOPER_PROMPT = """You are ChatTLA, an expert at writing verified TLA+ formal specifications.
When asked to write a TLA+ spec, follow these rules exactly:
1. Start the module with ---- MODULE <ModuleName> ----
2. End with ====
3. Include EXTENDS, VARIABLES, Init, Next, and Spec operators
4. After the TLA+ module, append a TLC configuration block:
   SPECIFICATION Spec
   INVARIANT TypeOK   (if TypeOK is defined)
5. Output only valid TLA+ code. No markdown fences, no explanation outside the spec.
Reasoning: medium"""

_MOD_RE = re.compile(r"----\s*MODULE\s+(\w+)")
_HTML_RE = re.compile(r"<[^>]+>")
_AUTHOR_RE = re.compile(
    r"^(authored by|by [A-Z]|this spec was written|copyright|\(c\)|\d{4}\s|.+@.+\.|.*inc\.|.*ltd|.*llc|see file|released under)",
    re.I,
)
_BAD_PREFIX_RE = re.compile(
    r"^(this (file |repository )?contains|this directory|spec for |\.\w+$)",
    re.I,
)


def _load_holdout_modules() -> set[str]:
    mods = set()
    if not _HOLDOUT.exists():
        return mods
    with _HOLDOUT.open() as f:
        for line in f:
            rec = json.loads(line)
            if "module" in rec:
                mods.add(rec["module"])
    return mods


def _clean_text(s: str) -> str:
    s = _HTML_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(".")


def _is_usable(text: str) -> bool:
    """Reject author/bio/boilerplate; require substantive task-like content."""
    if len(text) < 50:
        return False
    low = text.lower()
    if _AUTHOR_RE.match(low):
        return False
    if _BAD_PREFIX_RE.match(low):
        return False
    return True


def _extract_block_comments(src: str) -> list[str]:
    """Return balanced (* ... *) block contents in document order.

    Consecutive blocks (TLA+ header-box style uses one comment per line) are
    joined when their content strings are individually short — the typical
    result is a single coherent paragraph describing the module.
    """
    raw: list[str] = []
    i, n = 0, len(src)
    while i < n - 1:
        if src[i:i + 2] == "(*":
            depth = 1
            i += 2
            start = i
            while i < n - 1 and depth > 0:
                if src[i:i + 2] == "(*":
                    depth += 1
                    i += 2
                elif src[i:i + 2] == "*)":
                    depth -= 1
                    if depth == 0:
                        raw.append(src[start:i])
                        i += 2
                        break
                    i += 2
                else:
                    i += 1
        else:
            i += 1

    out: list[str] = []
    buf: list[str] = []
    for block in raw:
        text = block.strip().strip("*-").strip()
        if not text:
            if buf:
                out.append(" ".join(buf))
                buf = []
            continue
        if len(text) < 200:
            buf.append(text)
        else:
            if buf:
                out.append(" ".join(buf))
                buf = []
            out.append(text)
    if buf:
        out.append(" ".join(buf))
    return out


def _extract_line_comment_header(src: str) -> str:
    buf: list[str] = []
    for ln in src.splitlines()[:80]:
        s = ln.strip()
        if s.startswith("\\*"):
            t = s[2:].strip()
            if _AUTHOR_RE.match(t.lower()):
                continue
            buf.append(t)
        elif buf and s == "":
            continue
        elif buf:
            break
    return _clean_text(" ".join(buf))


def _resolve_description(rec: dict) -> str | None:
    """Prefer repo description; fall back to in-source block/line comments."""
    desc = _clean_text(rec.get("description") or "")
    if _is_usable(desc):
        return desc

    src = rec.get("tla_source", "")
    for block in _extract_block_comments(src):
        text = _clean_text(block.lstrip("*-").strip())
        if _is_usable(text):
            return text[:800]

    header = _extract_line_comment_header(src)
    if _is_usable(header):
        return header[:800]

    return None


def _resolve_module(rec: dict) -> str | None:
    """Pick the canonical module name. Prefer a non-MC module; else first one."""
    mods = _MOD_RE.findall(rec.get("tla_source", ""))
    if not mods:
        return None
    non_mc = [m for m in mods if not m.startswith("MC")]
    return (non_mc or mods)[0]


def _user_turn(module: str, description: str) -> str:
    return (
        f"Write a TLA+ specification named `{module}` for the following problem:\n\n"
        f"{description}\n"
    )


def _analysis_for(module: str, description: str) -> str:
    return (
        f"The problem asks for {description.rstrip('.')}. "
        f"I'll model it as TLA+ module {module} with explicit VARIABLES, "
        "an Init predicate, a Next-state action, and a safety invariant "
        "expressed in terms of those variables."
    )


def _record(rec: dict, tier: str, module: str, description: str) -> dict:
    return {
        "_tier": tier,
        "_prompt_id": f"tlaplus_examples/{rec['spec_name']}",
        "_source": "tlaplus_examples_v2",
        "_module": module,
        "_timestamp": datetime.utcnow().isoformat(),
        "_features": rec.get("features", {}),
        "_spec_path": rec.get("spec_path", ""),
        "messages": [
            {"role": "developer", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": _user_turn(module, description)},
            {
                "role": "assistant",
                "channel": "analysis",
                "content": _analysis_for(module, description),
            },
            {"role": "assistant", "channel": "final", "content": rec.get("tla_source", "")},
        ],
    }


def main() -> int:
    if not _LABELED.exists():
        print(f"[error] labeled corpus missing: {_LABELED}", file=sys.stderr)
        print(
            "        run the scraper first to emit tlaplus_examples_labeled.jsonl",
            file=sys.stderr,
        )
        return 1

    holdout = _load_holdout_modules()
    print(f"[info] loaded {len(holdout)} holdout module names to exclude")

    n_total = 0
    n_holdout_skipped = 0
    n_no_source = 0
    n_no_module = 0
    n_no_description = 0
    n_duplicate = 0
    tlc_rows: list[dict] = []
    tlaps_rows: list[dict] = []
    tlc_specs: list[str] = []
    tlaps_specs: list[str] = []
    seen_tlc: set[str] = set()
    seen_tlaps: set[str] = set()

    with _LABELED.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_total += 1

            spec_name = rec.get("spec_name", "")
            tla_source = rec.get("tla_source", "")
            features = rec.get("features", {}) or {}

            source_mods = set(_MOD_RE.findall(tla_source))
            source_mods.add(spec_name)
            if source_mods & holdout:
                n_holdout_skipped += 1
                continue

            if not tla_source.strip():
                n_no_source += 1
                continue

            module = _resolve_module(rec)
            if not module:
                n_no_module += 1
                continue

            description = _resolve_description(rec)
            if not description:
                n_no_description += 1
                continue

            if features.get("tlc_pass"):
                key = (module, len(tla_source))
                if key in seen_tlc:
                    n_duplicate += 1
                else:
                    seen_tlc.add(key)
                    tlc_rows.append(_record(rec, "tlc_target", module, description))
                    tlc_specs.append(spec_name)
            if features.get("tlaps_pass"):
                key = (module, len(tla_source))
                if key in seen_tlaps:
                    n_duplicate += 1
                else:
                    seen_tlaps.add(key)
                    tlaps_rows.append(_record(rec, "tlaps_target", module, description))
                    tlaps_specs.append(spec_name)

    with _OUT_TLC.open("w") as f:
        for r in tlc_rows:
            f.write(json.dumps(r) + "\n")
    with _OUT_TLAPS.open("w") as f:
        for r in tlaps_rows:
            f.write(json.dumps(r) + "\n")

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "source_rows": n_total,
        "holdout_skipped": n_holdout_skipped,
        "empty_source_skipped": n_no_source,
        "no_module_skipped": n_no_module,
        "no_description_skipped": n_no_description,
        "duplicates_skipped": n_duplicate,
        "tlc_target": {"count": len(tlc_rows), "specs": sorted(tlc_specs)},
        "tlaps_target": {"count": len(tlaps_rows), "specs": sorted(tlaps_specs)},
    }
    with _SUMMARY.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] source rows:             {n_total}")
    print(f"[ok] holdout collisions:      {n_holdout_skipped}")
    print(f"[ok] empty sources skipped:   {n_no_source}")
    print(f"[ok] no-module skipped:       {n_no_module}")
    print(f"[ok] no-description skipped:  {n_no_description}")
    print(f"[ok] duplicates skipped:      {n_duplicate}")
    print(f"[ok] tlc_target_sft.jsonl:    {len(tlc_rows)} rows -> {_OUT_TLC}")
    print(f"[ok] tlaps_target_sft.jsonl:  {len(tlaps_rows)} rows -> {_OUT_TLAPS}")
    print(f"[ok] summary:                 {_SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
