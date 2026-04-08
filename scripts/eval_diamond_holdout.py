#!/usr/bin/env python3
"""eval_diamond_holdout.py — measure diamond pass-rate on the held-out 30 specs.

Loads data/processed/diamond_eval_holdout.jsonl, prompts the model with each
spec's topic_desc, runs the resulting output through the Diamond validator
(scripts.diamond_sft_gen.validate_diamond), and prints a tier breakdown.

Usage:
    python3 scripts/eval_diamond_holdout.py --model chattla:20b
    python3 scripts/eval_diamond_holdout.py --model chattla:20b --no-rag
    python3 scripts/eval_diamond_holdout.py --model chattla:20b --out outputs/eval/holdout_pre.json
    python3 scripts/eval_diamond_holdout.py --model chattla:20b-v2 --out outputs/eval/holdout_post.json
    python3 scripts/eval_diamond_holdout.py --compare outputs/eval/holdout_pre.json outputs/eval/holdout_post.json

Output JSON shape (per-model file):
  {
    "model": "...", "rag_k": <int>, "n": 30,
    "diamond": <int>, "gold": <int>, "silver": <int>, "bronze": <int>,
    "per_spec": [
       {"module": "...", "batch": "...", "tier": "...",
        "is_diamond": bool, "distinct_states": int, "mutation_caught": bool,
        "error": "...", "generated_chars": int}, ...
    ]
  }
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_HOLDOUT = _REPO_ROOT / "data" / "processed" / "diamond_eval_holdout.jsonl"


def _load_holdout() -> list[dict]:
    if not _HOLDOUT.exists():
        sys.exit(f"missing {_HOLDOUT}; run carve_diamond_holdout.py first")
    out = []
    with _HOLDOUT.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _eval_one(
    client,
    validate_diamond,
    rec: dict,
    rag_k: int,
    self_correct: bool = False,
    max_retries: int = 3,
) -> dict:
    from time import monotonic
    t0 = monotonic()
    try:
        if self_correct:
            # validate_and_generate ignores rag_k (RAG is applied at the
            # underlying generate_spec call inside the loop). It runs SANY
            # then TLC, feeding parse errors and counterexamples back into
            # the model for up to max_retries rounds, plus a deterministic
            # Python fixer pass and PlusCal stripping. This is the v9 path
            # that hit 80% SANY / 25% TLC on the old 20-problem suite.
            gen, _tier_hint = client.validate_and_generate(
                rec["topic_desc"], max_retries=max_retries,
            )
        else:
            gen = client.generate_spec(
                rec["topic_desc"], module_name=rec["module"], rag_k=rag_k,
            )
    except Exception as e:
        return {
            "module": rec["module"],
            "batch": rec.get("batch", ""),
            "tier": "gen_error",
            "is_diamond": False,
            "error": f"generate_failed: {e}",
            "generated_chars": 0,
            "elapsed": monotonic() - t0,
        }

    try:
        r = validate_diamond(gen, prompt_id=rec["module"])
    except Exception as e:
        return {
            "module": rec["module"],
            "batch": rec.get("batch", ""),
            "tier": "validator_crash",
            "is_diamond": False,
            "error": f"validator_crash: {e}",
            "generated_chars": len(gen),
            "elapsed": monotonic() - t0,
        }

    return {
        "module": rec["module"],
        "batch": rec.get("batch", ""),
        "tier": r.tlc_tier,
        "is_diamond": bool(r.is_diamond),
        "distinct_states": r.distinct_states,
        "invariants_checked": r.invariants_checked,
        "mutation_caught": r.mutation_caught,
        "trivial_invariant": r.trivial_invariant,
        "error": r.error,
        "generated_chars": len(gen),
        "elapsed": round(monotonic() - t0, 2),
    }


def cmd_run(args: argparse.Namespace) -> int:
    from src.inference.ollama_client import ChatTLAClient
    from scripts.diamond_sft_gen import validate_diamond

    holdout = _load_holdout()
    sc_note = f" self_correct={args.max_retries}" if args.self_correct else ""
    print(f"[eval] {len(holdout)} holdout specs; model={args.model}; rag_k={args.rag_k}{sc_note}")
    client = ChatTLAClient(model=args.model)

    results: list[dict] = []
    for i, rec in enumerate(holdout, 1):
        r = _eval_one(
            client, validate_diamond, rec,
            rag_k=args.rag_k,
            self_correct=args.self_correct,
            max_retries=args.max_retries,
        )
        results.append(r)
        mark = "D" if r["is_diamond"] else (
            "G" if r["tier"] == "gold" else
            "S" if r["tier"] == "silver" else
            "B" if r["tier"] == "bronze" else "?"
        )
        sys.stdout.write(f"  [{i:2d}/{len(holdout)}] {mark}  {r['batch']:30s} {r['module']:30s} "
                         f"({r.get('elapsed', 0)}s)\n")
        sys.stdout.flush()

    n = len(results)
    n_diamond = sum(1 for r in results if r["is_diamond"])
    by_tier: dict[str, int] = {}
    for r in results:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1

    summary = {
        "model": args.model,
        "rag_k": args.rag_k,
        "self_correct": bool(args.self_correct),
        "max_retries": args.max_retries if args.self_correct else 0,
        "n": n,
        "diamond": n_diamond,
        "diamond_rate": round(n_diamond / n, 3) if n else 0.0,
        "by_tier": by_tier,
        "per_spec": results,
    }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"[eval] wrote {out}")

    print()
    print(f"[eval] {args.model}: DIAMOND {n_diamond}/{n}  ({summary['diamond_rate']*100:.1f}%)  by_tier={by_tier}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    print(f"A: {a['model']}: diamond {a['diamond']}/{a['n']} ({a['diamond_rate']*100:.1f}%)  by_tier={a['by_tier']}")
    print(f"B: {b['model']}: diamond {b['diamond']}/{b['n']} ({b['diamond_rate']*100:.1f}%)  by_tier={b['by_tier']}")
    delta = b["diamond"] - a["diamond"]
    pct = (b["diamond_rate"] - a["diamond_rate"]) * 100
    print(f"DELTA: {'+' if delta >= 0 else ''}{delta} specs  ({'+' if pct >= 0 else ''}{pct:.1f} pp)")

    a_by = {r["module"]: r for r in a["per_spec"]}
    b_by = {r["module"]: r for r in b["per_spec"]}
    flips_pos = []
    flips_neg = []
    for mod in a_by:
        if mod not in b_by:
            continue
        if not a_by[mod]["is_diamond"] and b_by[mod]["is_diamond"]:
            flips_pos.append(mod)
        if a_by[mod]["is_diamond"] and not b_by[mod]["is_diamond"]:
            flips_neg.append(mod)
    print()
    print(f"Diamond gains (A->B fail->pass): {len(flips_pos)}")
    for m in flips_pos:
        print(f"  + {m}")
    print(f"Diamond regressions (A->B pass->fail): {len(flips_neg)}")
    for m in flips_neg:
        print(f"  - {m}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=False)

    pr = sub.add_parser("run", help="evaluate a model")
    pr.add_argument("--model", required=True, help="Ollama model tag, e.g. chattla:20b")
    pr.add_argument("--rag-k", type=int, default=2, help="0 to disable RAG")
    pr.add_argument("--no-rag", action="store_const", const=0, dest="rag_k")
    pr.add_argument("--out", default=None, help="write per-model JSON summary here")
    pr.add_argument(
        "--self-correct", action="store_true",
        help="route generation through validate_and_generate (SANY+TLC error-feedback loop)",
    )
    pr.add_argument(
        "--max-retries", type=int, default=3,
        help="self-correction retry budget per spec (only with --self-correct)",
    )
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser("compare", help="diff two run summaries")
    pc.add_argument("a", help="earlier (baseline) summary JSON")
    pc.add_argument("b", help="later (post-train) summary JSON")
    pc.set_defaults(func=cmd_compare)

    # convenience: bare invocation `--model X --out Y` defaults to "run"
    p.add_argument("--model", help=argparse.SUPPRESS)
    p.add_argument("--rag-k", type=int, default=2, help=argparse.SUPPRESS)
    p.add_argument("--no-rag", action="store_const", const=0, dest="rag_k", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    p.add_argument("--self-correct", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--max-retries", type=int, default=3, help=argparse.SUPPRESS)

    args = p.parse_args()
    if args.cmd is None:
        if not args.model:
            p.error("either subcommand 'run'/'compare' or --model required")
        args.func = cmd_run
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
