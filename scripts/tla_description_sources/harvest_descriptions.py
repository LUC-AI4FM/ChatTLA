#!/usr/bin/env python3
"""
Harvest official TLA+ module descriptions from tlaplus/Examples + in-file comments.

Tracks git state so you can re-run after upstream changes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_COARSE = _REPO_ROOT / "data" / "tla-compents-coarse.json"
_DEFAULT_OUT = _REPO_ROOT / "data" / "derived" / "tla_descriptions.json"
_DEFAULT_AUDIT = _REPO_ROOT / "data" / "derived" / "tla_descriptions_audit.json"
_EXTERNAL = _REPO_ROOT / "data" / "external" / "tlaplus-examples"
_TLAPM = _REPO_ROOT / "data" / "external" / "tlapm"
_GIT_URL = "https://github.com/tlaplus/Examples.git"
_TLAPM_GIT_URL = "https://github.com/tlaplus/tlapm.git"


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


def compose_description(meta: dict[str, Any], header_comment: str) -> tuple[str, str, list[str]]:
    """
    Returns (description, confidence, provenance_tags).
    """
    parts: list[str] = []
    tags: list[str] = []
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
        parts.append("References: " + "; ".join(meta["sources"][:5]))
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
    }

    for row in coarse:
        spec_name = row["Specification"]
        entry: dict[str, Any] = {
            "id": row["id"],
            "specification": spec_name,
            "description": "",
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
                entry["description"] = (
                    f"No {spec_name}.tla in tlaplus/Examples@{git_meta.get('commit', '?')[:8]} "
                    f"or tlaplus/tlapm/library@{tlapm_meta.get('commit', '?')[:8]}."
                )
                entry["confidence"] = "none"
                entry["provenance"] = ["not_found"]
                stats["missing_file"] += 1
                results.append(entry)
                continue

        if source_kind == "examples":
            stats["resolved_from_tlaplus_examples"] += 1
        header = extract_tla_header_comment(tla)
        desc, conf, tags = compose_description(meta or {}, header)
        if source_kind == "tlapm":
            tags = ["tlapm_proof_library"] + tags
            entry["source_repository"] = "tlaplus/tlapm"
        else:
            entry["source_repository"] = "tlaplus/Examples"
        entry["description"] = desc
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

        results.append(entry)

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
            "Descriptions combine README table title + manifest authors/sources + first TLA comment block.",
            "Modules not in Examples are resolved from tlaplus/tlapm/library/ (TLAPS proof library).",
            "Re-run after upstream changes; both git commits are recorded in each row and in audit.",
        ],
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
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    harvest(args.coarse, args.out, args.audit, args.update_repo, args.verbose)


if __name__ == "__main__":
    main()
