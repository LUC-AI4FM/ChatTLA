#!/usr/bin/env python3
"""Run multi-model Diamond benchmark across prompt regimes.

This is a pure inference experiment:
- no training
- no LoRA changes
- no model merges

For each (model, regime, problem) triple:
  1) generate a spec
  2) score with the existing Diamond validator
  3) append one CSV row (flush + fsync after each row)

The runner is resumable: if results.csv already exists, completed triples are
loaded and skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diamond_sft_gen import validate_diamond
from src.inference.ollama_client import ChatTLAClient
from src.inference.piecewise_gen import generate_piecewise


MODELS = [
    "chattla:20b",
    "gpt-oss:20b",
    "llama3.1:70b",
    "qwen2.5:72b",
]

REGIMES = [
    "single_shot",
    "piecewise",
]

CSV_FIELDS = [
    "model",
    "regime",
    "problem_id",
    "problem_name",
    "d1_sany",
    "d2_reach",
    "d3_nonvac",
    "d4_mutate",
    "tier",
    "error_msg",
    "elapsed_s",
]


def find_holdout_path(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"holdout file not found: {p}")

    candidates = [
        REPO_ROOT / "data" / "benchmarks" / "diamond_holdout.jsonl",
        REPO_ROOT / "data" / "processed" / "diamond_eval_holdout.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p

    for p in (REPO_ROOT / "data").rglob("*diamond*holdout*.jsonl"):
        if p.is_file():
            return p

    raise FileNotFoundError(
        "Could not locate a 30-problem Diamond holdout JSONL. "
        "Tried standard paths and data/**/diamond*holdout*.jsonl."
    )


def load_holdout(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"holdout is empty: {path}")
    return rows


def problem_id(rec: dict, index_1based: int) -> str:
    return str(
        rec.get("problem_id")
        or rec.get("id")
        or rec.get("module")
        or f"P{index_1based:03d}"
    )


def problem_name(rec: dict) -> str:
    return str(
        rec.get("problem_name")
        or rec.get("name")
        or rec.get("module")
        or rec.get("topic_desc")
        or ""
    )


def load_available_ollama_models() -> set[str]:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception as e:
        print(f"[warn] could not query ollama list: {e}", file=sys.stderr)
        return set()

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        print(f"[warn] ollama list failed: {tail}", file=sys.stderr)
        return set()

    models: set[str] = set()
    for line in proc.stdout.splitlines()[1:]:  # skip header
        line = line.rstrip()
        if not line.strip():
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if parts:
            models.add(parts[0])
    return models


def ensure_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()


def load_completed(path: Path) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not path.exists() or path.stat().st_size == 0:
        return done

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = (row.get("model") or "").strip()
            regime = (row.get("regime") or "").strip()
            pid = (row.get("problem_id") or "").strip()
            if model and regime and pid:
                done.add((model, regime, pid))
    return done


def run_single_shot(client: ChatTLAClient, rec: dict, module_name: str) -> str:
    prompt = str(rec.get("topic_desc") or rec.get("description") or "")
    return client.generate_spec(prompt, module_name=module_name)


def run_piecewise(rec: dict, pid: str, module_name: str, model_tag: str) -> str:
    prompt = str(rec.get("topic_desc") or rec.get("description") or "")
    pw = generate_piecewise(
        problem_id=pid,
        nl_description=prompt,
        module_name=module_name,
        model_tag=model_tag,
    )
    return pw.spec


def score_diamond(spec: str, pid: str, prompt_text: str, model: str) -> tuple[int, int, int, int, int, str]:
    """Return (d1, d2, d3, d4, tier, error_msg)."""
    try:
        r = validate_diamond(
            spec,
            prompt_id=pid,
            prompt_text=prompt_text,
            model=model,
        )
    except Exception as e:
        return 0, 0, 0, 0, 0, f"validator_crash: {e}"

    d1 = int(bool(r.sany_pass))
    d2 = int(bool(d1 and r.tlc_tier == "gold" and r.distinct_states > 1))
    d3 = int(bool(d2 and r.invariants_checked > 0 and not r.trivial_invariant))
    d4 = int(bool(d3 and r.mutation_tested and r.mutation_caught))

    if d4:
        tier = 4
    elif d3:
        tier = 3
    elif d2:
        tier = 2
    elif d1:
        tier = 1
    else:
        tier = 0

    error_msg = ""
    if not d1:
        error_msg = r.error or "sany_fail"
    return d1, d2, d3, d4, tier, error_msg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-model Diamond benchmark")
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "diamond_multimodel" / "results.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--holdout",
        default=None,
        help="Optional holdout JSONL path (auto-detected if omitted)",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=MODELS,
        help="Model tags to evaluate",
    )
    p.add_argument(
        "--regimes",
        nargs="*",
        default=REGIMES,
        choices=REGIMES,
        help="Prompt regimes to evaluate",
    )
    p.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Optional cap for smoke testing (default: all holdout problems)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    holdout_path = find_holdout_path(args.holdout)
    holdout = load_holdout(holdout_path)
    if args.max_problems is not None:
        holdout = holdout[: max(args.max_problems, 0)]
    out_csv = Path(args.output)

    ensure_csv_header(out_csv)
    completed = load_completed(out_csv)

    available = load_available_ollama_models()
    selected_models: list[str] = []
    for m in args.models:
        if available and m not in available:
            print(f"[warn] model not available in Ollama, skipping: {m}")
            continue
        selected_models.append(m)

    if not selected_models:
        print("[error] no selected models are available; nothing to run", file=sys.stderr)
        return 1

    print(f"[info] holdout: {holdout_path} ({len(holdout)} problems)")
    print(f"[info] output:  {out_csv}")
    print(f"[info] models:  {selected_models}")
    print(f"[info] regimes: {args.regimes}")
    print(f"[info] resume:  {len(completed)} completed triples already in CSV")

    written = 0
    skipped = 0
    client_cache: dict[str, ChatTLAClient] = {}

    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)

        for model in selected_models:
            if "single_shot" in args.regimes and model not in client_cache:
                client_cache[model] = ChatTLAClient(model=model, reasoning="medium")

            for regime in args.regimes:
                for i, rec in enumerate(holdout, start=1):
                    pid = problem_id(rec, i)
                    pname = problem_name(rec)
                    key = (model, regime, pid)
                    if key in completed:
                        skipped += 1
                        continue

                    module_name = str(rec.get("module") or pid)
                    prompt_text = str(rec.get("topic_desc") or rec.get("description") or "")

                    t0 = time.monotonic()
                    generation_error = ""
                    spec = ""

                    try:
                        if regime == "single_shot":
                            spec = run_single_shot(client_cache[model], rec, module_name)
                        elif regime == "piecewise":
                            spec = run_piecewise(rec, pid, module_name, model)
                        else:
                            raise ValueError(f"unknown regime: {regime}")
                    except Exception as e:
                        generation_error = f"generation_failed: {e}"

                    d1, d2, d3, d4, tier, error_msg = score_diamond(
                        spec,
                        pid=pid,
                        prompt_text=prompt_text,
                        model=model,
                    )

                    if generation_error:
                        if error_msg:
                            error_msg = f"{generation_error}; {error_msg}"
                        else:
                            error_msg = generation_error

                    elapsed = round(time.monotonic() - t0, 2)

                    row = {
                        "model": model,
                        "regime": regime,
                        "problem_id": pid,
                        "problem_name": pname,
                        "d1_sany": d1,
                        "d2_reach": d2,
                        "d3_nonvac": d3,
                        "d4_mutate": d4,
                        "tier": tier,
                        "error_msg": error_msg,
                        "elapsed_s": elapsed,
                    }

                    writer.writerow(row)
                    f.flush()
                    os.fsync(f.fileno())

                    completed.add(key)
                    written += 1

                    print(f"[{model}/{regime}] {pid} ... tier={tier} ({elapsed:.2f}s)")

    print(
        f"[done] wrote {written} new rows, skipped {skipped} existing rows, "
        f"total completed triples now {len(completed)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
