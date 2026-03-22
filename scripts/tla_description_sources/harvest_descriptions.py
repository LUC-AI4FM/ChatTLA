#!/usr/bin/env python3
"""
Harvest TLA+ module metadata and build a regeneration-oriented dataset JSON.

Each row uses id=<module>_001, module_name, coarse_id, and description.{narrative,technical}.
Without --llm: programmatic parsing (tla_static_extract.py) fills description.technical from
Init/Next/definitions; narrative blends that analysis with harvested prose. With --llm, Ollama
fills the same schema (structured_dataset.py). Use --no-static-extract to skip parsing when not using --llm.

Tracks git state so you can re-run after upstream changes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from pdf_extract import collect_pdf_excerpts_for_sources
from structured_dataset import (
    build_llm_user_message,
    call_ollama_structured,
    dataset_record_id,
    empty_description_from_harvest,
    merge_llm_record,
)
from sany_extract import extract_with_sany
from tla_static_extract import extract_structured_description, merge_harvest_prose_into_narrative

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_COARSE = _REPO_ROOT / "data" / "tla-compents-coarse.json"
_DEFAULT_OUT = _REPO_ROOT / "data" / "derived" / "tla_descriptions.json"
_DEFAULT_AUDIT = _REPO_ROOT / "data" / "derived" / "tla_descriptions_audit.json"
_PDF_CACHE = _REPO_ROOT / "data" / "derived" / ".pdf_cache"
_EXTERNAL = _REPO_ROOT / "data" / "external" / "tlaplus-examples"
_TLAPM = _REPO_ROOT / "data" / "external" / "tlapm"
_GIT_URL = "https://github.com/tlaplus/Examples.git"
_TLAPM_GIT_URL = "https://github.com/tlaplus/tlapm.git"
_DEFAULT_LLM_MODEL = os.getenv("TLA_DATASET_LLM_MODEL", "gpt-oss:20b")
_DEFAULT_LLM_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _git_meta(repo_path: Path) -> dict[str, str]:
    r = _run(["git", "-C", str(repo_path), "rev-parse", "HEAD"])
    commit = r.stdout.strip() if r.returncode == 0 else "unknown"
    r = _run(["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"])
    branch = r.stdout.strip() if r.returncode == 0 else "unknown"
    r = _run(["git", "-C", str(repo_path), "log", "-1", "--format=%ci"])
    date = r.stdout.strip() if r.returncode == 0 else ""
    return {"commit": commit, "branch": branch, "commit_date": date}


def ensure_examples_clone(update: bool, verbose: bool) -> dict[str, str]:
    """Clone or pull tlaplus/Examples; return git metadata."""
    meta: dict[str, str] = {}
    if _EXTERNAL.exists() and (_EXTERNAL / ".git").exists():
        if update:
            r = _run(["git", "-C", str(_EXTERNAL), "pull", "--ff-only"])
            if r.returncode != 0 and verbose:
                print("git pull failed:", r.stderr[-500:], file=sys.stderr)
        r = _run(["git", "-C", str(_EXTERNAL), "rev-parse", "HEAD"])
        meta["commit"] = r.stdout.strip() if r.returncode == 0 else "unknown"
        r = _run(["git", "-C", str(_EXTERNAL), "rev-parse", "--abbrev-ref", "HEAD"])
        meta["branch"] = r.stdout.strip() if r.returncode == 0 else "unknown"
        r = _run(["git", "-C", str(_EXTERNAL), "log", "-1", "--format=%ci"])
        meta["commit_date"] = r.stdout.strip() if r.returncode == 0 else ""
    else:
        _EXTERNAL.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Cloning {_GIT_URL} -> {_EXTERNAL}")
        r = _run(
            ["git", "clone", "--depth", "1", _GIT_URL, str(_EXTERNAL)]
        )
        if r.returncode != 0:
            raise RuntimeError(f"git clone failed: {r.stderr}")
        r = _run(["git", "-C", str(_EXTERNAL), "rev-parse", "HEAD"])
        meta["commit"] = r.stdout.strip()
        r = _run(["git", "-C", str(_EXTERNAL), "rev-parse", "--abbrev-ref", "HEAD"])
        meta["branch"] = r.stdout.strip()
        r = _run(["git", "-C", str(_EXTERNAL), "log", "-1", "--format=%ci"])
        meta["commit_date"] = r.stdout.strip()
    return meta


def ensure_tlapm_clone(update: bool, verbose: bool) -> dict[str, str]:
    """Clone or pull tlaplus/tlapm (TLAPS proof library .tla files)."""
    if _TLAPM.exists() and (_TLAPM / ".git").exists():
        if update:
            r = _run(["git", "-C", str(_TLAPM), "pull", "--ff-only"])
            if r.returncode != 0 and verbose:
                print("tlapm git pull failed:", r.stderr[-300:], file=sys.stderr)
        return _git_meta(_TLAPM)
    _TLAPM.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Cloning {_TLAPM_GIT_URL} -> {_TLAPM}")
    r = _run(["git", "clone", "--depth", "1", _TLAPM_GIT_URL, str(_TLAPM)])
    if r.returncode != 0:
        raise RuntimeError(f"tlapm git clone failed: {r.stderr}")
    return _git_meta(_TLAPM)


def parse_readme_titles(readme_path: Path) -> dict[str, str]:
    """
    Map spec directory name (e.g. 'Paxos', 'transaction_commit') -> title from README table.
    """
    text = readme_path.read_text(encoding="utf-8", errors="replace")
    # | [Title](specifications/DirName) | Authors | ...
    pat = re.compile(
        r"\|\s*\[([^\]]+)\]\(specifications/([^)]+)\)\s*\|",
        re.MULTILINE,
    )
    out: dict[str, str] = {}
    for m in pat.finditer(text):
        title, dirname = m.group(1).strip(), m.group(2).strip()
        # Normalize: strip trailing slash
        dirname = dirname.rstrip("/")
        out[dirname] = title
    return out


def load_all_manifests(root: Path) -> list[tuple[Path, dict]]:
    rows = []
    for p in root.glob("specifications/*/manifest.json"):
        try:
            rows.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except json.JSONDecodeError:
            continue
    return rows


def module_to_spec_dir(tla_path: Path, specs_root: Path) -> str:
    rel = tla_path.relative_to(specs_root)
    parts = rel.parts
    return parts[0] if parts else ""


def build_module_index(
    examples_root: Path,
    readme_titles: dict[str, str],
    manifests: list[tuple[Path, dict]],
) -> dict[str, dict[str, Any]]:
    """
    module_name (no .tla) -> best metadata from manifests + paths.
    """
    specs = examples_root / "specifications"
    index: dict[str, dict[str, Any]] = {}

    for manifest_path, data in manifests:
        spec_dir = manifest_path.parent.name
        title = readme_titles.get(spec_dir, "")
        authors = data.get("authors") or []
        sources = data.get("sources") or []
        for mod in data.get("modules", []):
            p = mod.get("path", "")
            if not p.endswith(".tla"):
                continue
            name = Path(p).stem
            full_path = examples_root / p
            if name not in index or len(title) > len(index[name].get("readme_title", "")):
                index[name] = {
                    "module": name,
                    "tla_path": str(full_path.relative_to(examples_root)),
                    "spec_directory": spec_dir,
                    "readme_title": title,
                    "authors": authors,
                    "sources": sources,
                    "manifest": str(manifest_path.relative_to(examples_root)),
                    "github_dir_url": f"https://github.com/tlaplus/Examples/tree/master/specifications/{spec_dir}",
                    "github_raw_url": f"https://raw.githubusercontent.com/tlaplus/Examples/master/{p}",
                }
    # Also index any .tla not in manifest (rare)
    for tla in specs.rglob("*.tla"):
        name = tla.stem
        if name in index:
            continue
        spec_dir = tla.parent.name
        index[name] = {
            "module": name,
            "tla_path": str(tla.relative_to(examples_root)),
            "spec_directory": spec_dir,
            "readme_title": readme_titles.get(spec_dir, ""),
            "authors": [],
            "sources": [],
            "manifest": "",
            "github_dir_url": f"https://github.com/tlaplus/Examples/tree/master/specifications/{spec_dir}",
            "github_raw_url": f"https://raw.githubusercontent.com/tlaplus/Examples/master/{tla.relative_to(examples_root)}",
        }
    return index


def extract_tla_header_comment(tla_path: Path) -> str:
    """First multi-line (* ... *) block after MODULE line (official in-file description)."""
    try:
        text = tla_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    # Skip ---- MODULE Name ----
    i = text.find("MODULE")
    if i == -1:
        return ""
    rest = text[i:]
    m = re.search(r"\(\*+", rest)
    if not m:
        return ""
    start = m.start()
    depth = 0
    j = start
    while j < len(rest):
        if rest[j : j + 2] == "(*":
            depth += 1
            j += 2
            continue
        if rest[j : j + 2] == "*)":
            depth -= 1
            j += 2
            if depth == 0:
                block = rest[start:j]
                inner = re.sub(r"^\(\*+\s*|\s*\*+\)$", "", block, flags=re.MULTILINE)
                lines = []
                for line in inner.splitlines():
                    line = re.sub(r"^\s*\(\*|\*\)\s*$", "", line)
                    line = line.strip()
                    if line.startswith("(*") or line.endswith("*)"):
                        line = line.strip("(*)")
                    lines.append(line)
                return "\n".join(l for l in lines if l).strip()
        else:
            j += 1
    return ""


def compose_description(
    meta: dict[str, Any],
    header_comment: str,
    exclude_urls_from_refs: Optional[set[str]] = None,
) -> tuple[str, str, list[str]]:
    """
    Returns (description, confidence, provenance_tags).

    When PDF text was extracted for a source URL, pass that URL in
    ``exclude_urls_from_refs`` so we don't duplicate the raw PDF link in
    ``References:`` (the excerpt carries the paper content).
    """
    parts: list[str] = []
    tags: list[str] = []
    exclude_urls_from_refs = exclude_urls_from_refs or set()
    if meta.get("readme_title"):
        parts.append(meta["readme_title"])
        tags.append("readme_table")
    if meta.get("authors"):
        parts.append("Authors: " + ", ".join(meta["authors"]))
        tags.append("manifest_authors")
    if header_comment:
        # Prefer comment as primary if readme is generic
        hc = header_comment[:4000]
        if not parts:
            parts.append(hc)
            tags.append("tla_header_only")
        else:
            parts.append("Module header (excerpt):\n" + hc[:2000])
            tags.append("tla_header")
    if meta.get("sources"):
        refs = [u for u in meta["sources"][:5] if u not in exclude_urls_from_refs]
        if refs:
            parts.append("References: " + "; ".join(refs))
            tags.append("manifest_sources")
    text = "\n\n".join(parts).strip()
    conf = "high" if meta.get("readme_title") and header_comment else (
        "medium" if meta.get("readme_title") or header_comment else "low"
    )
    return text, conf, tags


def harvest(
    coarse_path: Path,
    out_path: Path,
    audit_path: Path,
    update_repo: bool,
    verbose: bool,
    skip_pdf: bool = False,
    no_static_extract: bool = False,
    use_llm: bool = False,
    llm_model: str = "",
    llm_host: str = "",
    max_tla_chars: int = 28000,
    llm_delay_s: float = 0.0,
    limit: Optional[int] = None,
) -> None:
    git_meta = ensure_examples_clone(update_repo, verbose)
    tlapm_meta = ensure_tlapm_clone(update_repo, verbose)
    readme = _EXTERNAL / "README.md"
    if not readme.exists():
        raise FileNotFoundError(readme)

    readme_titles = parse_readme_titles(readme)
    manifests = load_all_manifests(_EXTERNAL)
    index = build_module_index(_EXTERNAL, readme_titles, manifests)

    coarse = json.loads(coarse_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    stats = {
        "coarse_entries": len(coarse),
        "resolved_from_tlaplus_examples": 0,
        "resolved_from_tlapm_library": 0,
        "missing_file": 0,
        "pdf_excerpt_rows": 0,
        "pdf_fetch_errors": 0,
        "pdf_chars_total": 0,
        "llm_rows": 0,
        "llm_parse_failures": 0,
        "sany_extract_rows": 0,
        "sany_fallback_static": 0,
        "static_extract_rows": 0,
    }

    llm_model = llm_model or _DEFAULT_LLM_MODEL
    llm_host = llm_host or _DEFAULT_LLM_HOST

    for row_i, row in enumerate(coarse):
        if limit is not None and row_i >= limit:
            break
        spec_name = row["Specification"]
        entry: dict[str, Any] = {
            "id": dataset_record_id(spec_name),
            "module_name": spec_name,
            "coarse_id": row["id"],
            "description": {},
            "confidence": "none",
            "provenance": [],
            "upstream": {
                "tlaplus_examples": {
                    "repo": "tlaplus/Examples",
                    **git_meta,
                },
                "tlaplus_tlapm": {
                    "repo": "tlaplus/tlapm",
                    **tlapm_meta,
                },
            },
        }

        meta = index.get(spec_name)
        if not meta:
            tla = None
            for p in (_EXTERNAL / "specifications").rglob(f"{spec_name}.tla"):
                tla = p
                break
        else:
            tla = _EXTERNAL / meta["tla_path"]

        if not meta and tla is not None:
            spec_dir = tla.parent.name
            meta = {
                "module": spec_name,
                "tla_path": str(tla.relative_to(_EXTERNAL)),
                "spec_directory": spec_dir,
                "readme_title": readme_titles.get(spec_dir, ""),
                "authors": [],
                "sources": [],
                "manifest": "",
                "github_dir_url": f"https://github.com/tlaplus/Examples/tree/master/specifications/{spec_dir}",
                "github_raw_url": f"https://raw.githubusercontent.com/tlaplus/Examples/master/{tla.relative_to(_EXTERNAL)}",
            }

        source_kind = "examples"
        if not tla or not tla.exists():
            # Official TLAPS / proof library modules live in tlaplus/tlapm/library/
            tlapm_tla = _TLAPM / "library" / f"{spec_name}.tla"
            if tlapm_tla.exists():
                tla = tlapm_tla
                source_kind = "tlapm"
                rel = tla.relative_to(_TLAPM)
                meta = {
                    "readme_title": f"TLAPS proof library: {spec_name}",
                    "authors": [],
                    "sources": [
                        "https://github.com/tlaplus/tlapm/blob/main/library/README.md",
                    ],
                    "github_dir_url": "https://github.com/tlaplus/tlapm/tree/main/library",
                    "github_raw_url": f"https://raw.githubusercontent.com/tlaplus/tlapm/main/{rel.as_posix()}",
                }
                stats["resolved_from_tlapm_library"] += 1
            else:
                entry["description"] = empty_description_from_harvest(
                    f"No {spec_name}.tla in tlaplus/Examples@{git_meta.get('commit', '?')[:8]} "
                    f"or tlaplus/tlapm/library@{tlapm_meta.get('commit', '?')[:8]}.",
                    "Module file not found upstream.",
                )
                entry["confidence"] = "none"
                entry["provenance"] = ["not_found"]
                stats["missing_file"] += 1
                results.append(entry)
                continue

        if source_kind == "examples":
            stats["resolved_from_tlaplus_examples"] += 1
        header = extract_tla_header_comment(tla)
        try:
            tla_text = tla.read_text(encoding="utf-8", errors="replace")
        except OSError:
            tla_text = ""

        pdf_excerpt = ""
        pdf_details: list[dict[str, Any]] = []
        exclude_pdf_urls: set[str] = set()
        if (
            not skip_pdf
            and meta
            and meta.get("sources")
            and source_kind == "examples"
        ):
            pdf_excerpt, pdf_details = collect_pdf_excerpts_for_sources(
                meta["sources"],
                _PDF_CACHE,
                max_pdfs=2,
                max_pages=3,
                max_chars_per_pdf=3500,
                verbose=verbose,
            )
            for d in pdf_details:
                st = str(d.get("status", ""))
                ch = int(d.get("chars") or 0)
                if ch >= 40 and st in ("ok", "cache_hit"):
                    exclude_pdf_urls.add(str(d["url"]))
                if st.startswith("error:"):
                    stats["pdf_fetch_errors"] += 1
            if pdf_excerpt:
                stats["pdf_excerpt_rows"] += 1
                stats["pdf_chars_total"] += len(pdf_excerpt)

        desc, conf, tags = compose_description(
            meta or {}, header, exclude_urls_from_refs=exclude_pdf_urls
        )
        if pdf_excerpt:
            desc = (
                desc
                + "\n\n---\nFrom paper (PDF excerpt, first pages):\n"
                + pdf_excerpt
            )
            tags = tags + ["pdf_text_excerpt"]
        if source_kind == "tlapm":
            tags = ["tlapm_proof_library"] + tags
            entry["source_repository"] = "tlaplus/tlapm"
        else:
            entry["source_repository"] = "tlaplus/Examples"

        llm_warnings: list[str] = []
        if use_llm:
            stats["llm_rows"] += 1
            try:
                user_msg = build_llm_user_message(
                    spec_name,
                    tla_text,
                    readme_title=(meta or {}).get("readme_title", "") or "",
                    authors=(meta or {}).get("authors"),
                    sources=(meta or {}).get("sources"),
                    header_comment=header,
                    pdf_excerpt=pdf_excerpt,
                    max_tla_chars=max_tla_chars,
                )
                parsed, raw = call_ollama_structured(
                    user_msg, host=llm_host, model=llm_model
                )
                description_obj, llm_warnings = merge_llm_record(
                    spec_name, row["id"], parsed, raw
                )
                if "llm_parse_failed" in llm_warnings:
                    stats["llm_parse_failures"] += 1
            except Exception as exc:
                if verbose:
                    print(f"  LLM failed for {spec_name}: {exc}", file=sys.stderr, flush=True)
                description_obj = empty_description_from_harvest(
                    desc, f"LLM call failed: {exc!s}"
                )
                llm_warnings = [f"exception:{exc!s}"]
                stats["llm_parse_failures"] += 1
            if llm_delay_s > 0:
                time.sleep(llm_delay_s)
        elif not no_static_extract and tla and tla.exists():
            sany_include = [tla.parent]
            # Add tlapm/library for specs that EXTEND TLAPS
            tlapm_lib = _TLAPM / "library"
            if tlapm_lib.is_dir():
                sany_include.append(tlapm_lib)
            # Add sibling spec dirs (some specs EXTEND modules in parent/neighbor dirs)
            spec_root = tla.parent.parent
            if spec_root.is_dir():
                for peer in spec_root.iterdir():
                    if peer.is_dir() and peer != tla.parent:
                        sany_include.append(peer)

            sany_desc, sany_status = extract_with_sany(
                tla,
                module_name=spec_name,
                readme_title=(meta or {}).get("readme_title", "") or "",
                header_comment=header,
                harvest_prose=desc,
                include_dirs=sany_include,
            )
            if sany_desc is not None:
                description_obj = sany_desc
                stats["sany_extract_rows"] += 1
                tags = tags + ["sany_ast"]
                entry["sany_status"] = sany_status
                if verbose:
                    print(f"  {spec_name}: SANY {sany_status}", flush=True)
            elif tla_text.strip():
                stats["sany_fallback_static"] += 1
                description_obj = extract_structured_description(
                    tla_text,
                    module_name=spec_name,
                    readme_title=(meta or {}).get("readme_title", "") or "",
                    header_comment=header,
                    harvest_prose=desc,
                )
                description_obj = merge_harvest_prose_into_narrative(description_obj, desc)
                tags = tags + ["static_extract"]
                entry["sany_status"] = sany_status
                if verbose:
                    print(f"  {spec_name}: SANY failed ({sany_status}), fell back to static", flush=True)
            else:
                description_obj = empty_description_from_harvest(desc, f"SANY failed: {sany_status}; empty .tla.")
                entry["sany_status"] = sany_status
        else:
            description_obj = empty_description_from_harvest(
                desc,
                "Extraction disabled or no .tla file; use --llm for model-generated schema.",
            )

        entry["description"] = description_obj
        entry["confidence"] = conf
        entry["provenance"] = tags
        entry["paths"] = {
            "local_tla": str(tla.relative_to(_REPO_ROOT)),
            "github_directory": meta.get("github_dir_url", "") if meta else "",
            "github_raw": meta.get("github_raw_url", "") if meta else "",
            "manifest": meta.get("manifest", "") if meta else "",
        }
        if meta and meta.get("sources"):
            entry["official_sources"] = meta["sources"]
        if meta and meta.get("authors"):
            entry["authors"] = meta["authors"]
        if pdf_details:
            entry["pdf_excerpt_meta"] = pdf_details
        if use_llm:
            entry["llm"] = {
                "model": llm_model,
                "host": llm_host,
                "warnings": llm_warnings,
            }

        results.append(entry)

    stats["output_rows"] = len(results)

    _DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    audit = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "git": {"tlaplus_examples": git_meta, "tlaplus_tlapm": tlapm_meta},
        "coarse_file": str(coarse_path.relative_to(_REPO_ROOT)),
        "examples_clone": str(_EXTERNAL.relative_to(_REPO_ROOT)),
        "stats": stats,
        "readme_titles_parsed": len(readme_titles),
        "manifests_loaded": len(manifests),
        "modules_indexed": len(index),
        "how_to_refresh": [
            f"cd {_EXTERNAL} && git pull && cd {_TLAPM} && git pull",
            f"python3 {Path(__file__).relative_to(_REPO_ROOT)} --update-repo",
        ],
        "notes": [
            "Each row: id=<module>_001, module_name, coarse_id (index from tla-compents-coarse), description.{narrative,technical}.",
            "Use --llm + Ollama to generate full author-style narrative and technical reconstruction fields.",
            "Without --llm, SANY XMLExporter parses the .tla AST (via tla2tools.jar); falls back to regex-based static extraction if SANY fails. --no-static-extract skips both.",
            "Harvest sources: README title + manifest authors/sources + first TLA comment; PDF text via pypdf cache.",
            "Modules not in Examples resolve from tlaplus/tlapm/library/ (TLAPS proof library).",
        ],
        "llm": {
            "enabled": use_llm,
            "model": llm_model if use_llm else None,
            "host": llm_host if use_llm else None,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if verbose:
        print(json.dumps(audit["stats"], indent=2))
        print("Wrote", out_path)
        print("Wrote", audit_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest TLA+ descriptions from tlaplus/Examples")
    ap.add_argument("--coarse", type=Path, default=_DEFAULT_COARSE)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--audit", type=Path, default=_DEFAULT_AUDIT)
    ap.add_argument("--update-repo", action="store_true", help="git pull in data/external/tlaplus-examples")
    ap.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Do not download PDFs or extract text (faster, offline; References keep raw URLs).",
    )
    ap.add_argument(
        "--no-static-extract",
        action="store_true",
        help="Do not run programmatic TLA+ parsing to fill technical fields (only if not using --llm).",
    )
    ap.add_argument(
        "--llm",
        action="store_true",
        help="Call Ollama to fill description.narrative + description.technical (dataset schema for spec regeneration).",
    )
    ap.add_argument(
        "--llm-model",
        default="",
        help=f"Ollama model tag (default env TLA_DATASET_LLM_MODEL or {_DEFAULT_LLM_MODEL!r}).",
    )
    ap.add_argument(
        "--llm-host",
        default="",
        help=f"Ollama base URL (default env OLLAMA_HOST or {_DEFAULT_LLM_HOST!r}).",
    )
    ap.add_argument(
        "--max-tla-chars",
        type=int,
        default=28000,
        help="Max TLA+ source characters sent to the LLM (default 28000).",
    )
    ap.add_argument(
        "--llm-delay-s",
        type=float,
        default=0.0,
        help="Sleep between LLM requests (rate-limit GPU).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N coarse rows (debug).",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    harvest(
        args.coarse,
        args.out,
        args.audit,
        args.update_repo,
        args.verbose,
        skip_pdf=args.skip_pdf,
        no_static_extract=args.no_static_extract,
        use_llm=args.llm,
        llm_model=args.llm_model,
        llm_host=args.llm_host,
        max_tla_chars=args.max_tla_chars,
        llm_delay_s=args.llm_delay_s,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
