"""
publish_hf.py — Upload GGUF + Modelfile (+ optional README) to Hugging Face Hub.

Called after merge + GGUF conversion (e.g. from scripts/rl_loop.py).
Requires HF_TOKEN with write access to the model repo.

Versioning
----------
State file: data/benchmarks/hf_publish_state.json
  {"last_published_version": 11}

Each successful upload bumps to v(last+1), uploads:
  gguf/chattla-20b-v{N}-Q8_0.gguf
  gguf/Modelfile   (FROM ./chattla-20b-v{N}-Q8_0.gguf)
Optional:
  README.md        (from outputs/hf_readme/README.md, version + benchmark summary patched)

Usage
-----
  HF_TOKEN=... python -m src.training.publish_hf
  python -m src.training.publish_hf --dry-run
  python -m src.training.publish_hf --skip-readme
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GGUF_DIR = _REPO_ROOT / "outputs" / "gguf"
_MERGED_MODEL_DIR = _REPO_ROOT / "outputs" / "merged_model"
_STATE_PATH = _REPO_ROOT / "data" / "benchmarks" / "hf_publish_state.json"
_README_TEMPLATE = _REPO_ROOT / "outputs" / "hf_readme" / "README.md"
_DEFAULT_REPO = "EricSpencer00/chattla-20b"
# Match public model card v11; first automated publish becomes v12 unless overridden.
_INITIAL_VERSION = 11


def _load_state() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"last_published_version": _INITIAL_VERSION}


def _save_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def latest_full_benchmark_stats() -> dict | None:
    """Parse newest outputs/benchmark_results_*_full_*.csv for SANY/TLC rates."""
    pattern = "benchmark_results_*_full_*.csv"
    candidates = sorted(
        _REPO_ROOT.glob(f"outputs/{pattern}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            with path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            n = len(rows)

            def _ok(row: dict, key: str) -> bool:
                v = row.get(key, "")
                return str(v).strip() in ("1", "True", "true", "yes")

            sany = sum(1 for r in rows if _ok(r, "sany_pass"))
            tlc = sum(1 for r in rows if _ok(r, "tlc_pass"))
            structs = []
            for r in rows:
                try:
                    structs.append(float(r.get("structural_score", 0) or 0))
                except (TypeError, ValueError):
                    pass
            avg_s = sum(structs) / len(structs) if structs else 0.0
            st = path.stat()
            return {
                "n": n,
                "sany": sany,
                "tlc": tlc,
                "avg_struct": avg_s,
                "source_csv": str(path.name),
                "source_path": str(path),
                "mtime": st.st_mtime,
            }
        except (OSError, csv.Error):
            continue
    return None


def full_benchmark_fresh_enough(max_age_hours: float) -> tuple[bool, str]:
    """Require a parsed full-suite CSV newer than max_age_hours (wall clock)."""
    if max_age_hours is None or max_age_hours <= 0:
        return True, "no freshness requirement"
    stats = latest_full_benchmark_stats()
    if not stats:
        return False, "no outputs/benchmark_results_*_full_*.csv found — run full benchmark first"
    age_h = (time.time() - stats["mtime"]) / 3600.0
    if age_h > max_age_hours:
        return (
            False,
            f"newest full benchmark {stats['source_csv']} is {age_h:.1f}h old (limit {max_age_hours}h)",
        )
    return True, f"full benchmark OK ({stats['source_csv']}, {age_h:.1f}h old)"


def _patch_readme(text: str, version: int, stats: dict | None) -> str:
    """Bump visible v(N-1) → v{version} and optionally refresh benchmark summary line."""
    prev = version - 1
    text = re.sub(rf"\(v{prev}\)", f"(v{version})", text)
    text = re.sub(rf"\bv{prev}\b", f"v{version}", text)
    text = re.sub(
        rf"ChatTLA-20b \(v{version}\) \(v{version}\)",
        f"ChatTLA-20b (v{version})",
        text,
    )

    if stats:
        new_summary = (
            f"**SANY pass: {stats['sany']}/{stats['n']} ({100*stats['sany']/max(stats['n'],1):.0f}%) · "
            f"TLC pass: {stats['tlc']}/{stats['n']} ({100*stats['tlc']/max(stats['n'],1):.0f}%) · "
            f"Avg structural: {stats['avg_struct']:.2f}**"
        )
        text = re.sub(r"\*\*SANY pass:[^\n]+\*\*", new_summary, text, count=1)
        note = f"\n\n*Auto-updated from `{stats['source_csv']}` (full benchmark suite).*"
        if note.strip() not in text:
            text = text.replace(new_summary, new_summary + note, 1)

    return text


def publish(
    repo_id: str = _DEFAULT_REPO,
    quant: str = "Q8_0",
    *,
    skip_readme: bool = False,
    dry_run: bool = False,
    cycle_id: int | None = None,
    version_override: int | None = None,
    require_fresh_full_benchmark_hours: float | None = None,
    upload_merged_model: bool = False,
) -> int | None:
    """
    Upload artifacts. Returns new version number on success, None on skip/failure.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[publish_hf] ERROR: pip install huggingface_hub", file=sys.stderr)
        return None

    gguf_local = _GGUF_DIR / f"chattla-20b-{quant}.gguf"
    if not gguf_local.is_file():
        print(f"[publish_hf] ERROR: GGUF not found: {gguf_local}", file=sys.stderr)
        return None

    state = _load_state()
    last = int(state.get("last_published_version", _INITIAL_VERSION))
    new_ver = version_override if version_override is not None else last + 1

    path_in_repo = f"gguf/chattla-20b-v{new_ver}-{quant}.gguf"
    msg_parts = [f"v{new_ver}: ChatTLA GGUF ({quant})"]
    if cycle_id is not None:
        msg_parts.append(f"RL cycle {cycle_id}")
    commit_message = " — ".join(msg_parts)

    # Modelfile: relative GGUF name for Hub users (same folder as Modelfile)
    from src.inference.convert_to_gguf import MODELFILE_TEMPLATE

    rel_gguf = f"./chattla-20b-v{new_ver}-{quant}.gguf"
    modelfile_body = MODELFILE_TEMPLATE.format(gguf_path=rel_gguf)

    print(f"[publish_hf] Repo={repo_id} version=v{new_ver} file={path_in_repo}")

    ok_fresh, fresh_msg = full_benchmark_fresh_enough(require_fresh_full_benchmark_hours or 0)
    print(f"[publish_hf] Benchmark freshness: {fresh_msg}")
    if not ok_fresh:
        if dry_run:
            print("[publish_hf] WARN: would abort real publish — stale/missing full benchmark", file=sys.stderr)
        else:
            print(f"[publish_hf] ABORT: {fresh_msg}", file=sys.stderr)
            return None

    if dry_run:
        print("[publish_hf] Dry run — no upload")
        return new_ver

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[publish_hf] SKIP: HF_TOKEN not set", file=sys.stderr)
        return None

    api = HfApi(token=token)

    api.upload_file(
        path_or_fileobj=str(gguf_local),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"[publish_hf] Uploaded {path_in_repo}")

    with tempfile.NamedTemporaryFile("w", suffix="Modelfile", delete=False, encoding="utf-8") as tf:
        tf.write(modelfile_body)
        mf_tmp = tf.name
    try:
        api.upload_file(
            path_or_fileobj=mf_tmp,
            path_in_repo="gguf/Modelfile",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"v{new_ver}: Modelfile → {path_in_repo.split('/')[-1]}",
        )
        print("[publish_hf] Uploaded gguf/Modelfile")
    finally:
        Path(mf_tmp).unlink(missing_ok=True)

    if not skip_readme and _README_TEMPLATE.is_file():
        stats = latest_full_benchmark_stats()
        raw = _README_TEMPLATE.read_text(encoding="utf-8")
        patched = _patch_readme(raw, new_ver, stats)
        with tempfile.NamedTemporaryFile("w", suffix="README.md", delete=False, encoding="utf-8") as tf:
            tf.write(patched)
            rd_tmp = tf.name
        try:
            api.upload_file(
                path_or_fileobj=rd_tmp,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"v{new_ver}: README (benchmarks from latest full CSV if present)",
            )
            print("[publish_hf] Uploaded README.md")
        finally:
            Path(rd_tmp).unlink(missing_ok=True)
    elif not skip_readme:
        print("[publish_hf] WARN: outputs/hf_readme/README.md missing — skip README")

    if upload_merged_model and _MERGED_MODEL_DIR.is_dir():
        cfg = _MERGED_MODEL_DIR / "config.json"
        if cfg.is_file():
            print("[publish_hf] Uploading merged BF16 folder (~tens of GB) → merged_bf16/ …")
            api.upload_folder(
                folder_path=str(_MERGED_MODEL_DIR),
                path_in_repo="merged_bf16",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"v{new_ver}: merged BF16 weights (pytorch_model.bin + config)",
                ignore_patterns=[".git*", "*.tmp"],
            )
            print("[publish_hf] Uploaded merged_bf16/")
        else:
            print("[publish_hf] WARN: merged_model/ missing config.json — skip merged upload")
    elif upload_merged_model:
        print("[publish_hf] WARN: outputs/merged_model not found — skip merged upload")

    state["last_published_version"] = new_ver
    state["last_repo"] = repo_id
    state["last_gguf_path_in_repo"] = path_in_repo
    if cycle_id is not None:
        state["last_cycle_id"] = cycle_id
    _save_state(state)
    print(f"[publish_hf] Done. State saved → {_STATE_PATH}")
    return new_ver


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Publish ChatTLA GGUF to Hugging Face Hub")
    parser.add_argument("--repo", default=_DEFAULT_REPO)
    parser.add_argument("--quant", default="Q8_0")
    parser.add_argument("--skip-readme", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cycle-id", type=int, default=None)
    parser.add_argument("--version", type=int, default=None, help="Force version number (default: last+1)")
    parser.add_argument(
        "--require-fresh-full-benchmark-hours",
        type=float,
        default=0,
        help="Abort if no full benchmark CSV or newest is older than this many hours (0=disabled)",
    )
    parser.add_argument(
        "--upload-merged-model",
        action="store_true",
        help="Also upload outputs/merged_model/ to merged_bf16/ (~40GB+; slow)",
    )
    args = parser.parse_args()

    # Optional: load .env from repo root
    env_path = _REPO_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    req_h = (
        None
        if args.require_fresh_full_benchmark_hours <= 0
        else args.require_fresh_full_benchmark_hours
    )
    v = publish(
        repo_id=args.repo,
        quant=args.quant,
        skip_readme=args.skip_readme,
        dry_run=args.dry_run,
        cycle_id=args.cycle_id,
        version_override=args.version,
        require_fresh_full_benchmark_hours=req_h,
        upload_merged_model=args.upload_merged_model,
    )
    if args.dry_run:
        return 0
    return 0 if v is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
