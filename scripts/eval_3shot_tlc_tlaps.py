#!/usr/bin/env python3
"""eval_3shot_tlc_tlaps.py — chattla:20b single-shot + 3-retry self-correct loop
that feeds BOTH TLC and TLAPS errors back into the model.

Hard 30-second wall clock per LLM response (httpx timeout on the ollama client).
Live streaming of progress to stdout for direct monitoring.

Usage:
    python3 scripts/eval_3shot_tlc_tlaps.py [N]   # default N=5 from diamond holdout
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402
import ollama  # noqa: E402

from src.inference.ollama_client import (  # noqa: E402
    ChatTLAClient,
    _build_harmony_prompt,
    _extract_tla,
    _sanitize_spec,
)
from src.training.dataset_builder import _DEVELOPER_PROMPT  # noqa: E402
from src.validators.sany_validator import validate_string as sany_validate  # noqa: E402
from src.validators.tlc_validator import validate_string as tlc_validate  # noqa: E402
from src.validators.tlaps_validator import validate_string as tlaps_validate  # noqa: E402

MODEL = "chattla:20b"
TIMEOUT_S = 30
MAX_RETRIES = 3
HOLDOUT = REPO / "data" / "processed" / "diamond_eval_holdout.jsonl"


def log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def make_client() -> ChatTLAClient:
    c = ChatTLAClient(model=MODEL)
    # Replace the underlying ollama client with one that has a hard 30s
    # request-level timeout. httpx will raise on any call that runs longer.
    c._client = ollama.Client(host="http://localhost:11434", timeout=TIMEOUT_S)
    return c


def _tail(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else "…" + s[-n:]


def build_feedback(sany_r, tlc_r, tlaps_r) -> str:
    """Build a feedback string from validator raw_output tails.

    The validator dataclass `.errors` / `.tlc_violations` summary fields
    drop critical context (line numbers, expected tokens, the violating
    state trace). The model can only self-correct if it sees that detail,
    so we feed back raw_output tails directly.
    """
    parts: list[str] = []
    if sany_r is not None and not sany_r.valid:
        parts.append("SANY parse errors:\n" + _tail(sany_r.raw_output, 1500))
    if tlc_r is not None and tlc_r.tier != "gold":
        parts.append(f"TLC ({tlc_r.tier}) output:\n" + _tail(tlc_r.raw_output, 1500))
    if tlaps_r is not None and tlaps_r.tier in ("partial", "unproved", "parse_error"):
        parts.append(
            f"TLAPS ({tlaps_r.tier}) output:\n" + _tail(tlaps_r.raw_output, 800)
        )
    return "\n\n".join(parts)


def self_correct(
    client: ChatTLAClient,
    nl_description: str,
    module_name: str,
    buggy_spec: str,
    errors: str,
    attempt: int,
) -> str:
    """Local self-correct call.

    Two non-obvious changes vs the stock `_self_correct_sany`:

    1. We pass the original natural-language description into the retry
       prompt. The stock helper omits it, which means the model has to
       infer the requirement from the buggy spec alone — and tends to
       just regurgitate that spec verbatim.

    2. On the final retry (attempt == MAX_RETRIES) we drop the buggy
       spec entirely and ask for a fresh write conditioned only on the
       description + error summary. Empirically the buggy spec sitting
       in context dominates the distribution and collapses retries into
       byte-identical copies even at temperature 0.85.

    Temperature schedule: 0.50 → 0.70 → 0.90 across attempts 1..3.
    """
    temperature = 0.50 + 0.20 * (attempt - 1)
    developer_content = f"{_DEVELOPER_PROMPT}\nReasoning: high"

    if attempt >= MAX_RETRIES:
        # Fresh write — no buggy spec in context
        user_content = (
            f"Write a TLA+ specification for the following system. "
            f"Use module name: {module_name}\n\n"
            f"Requirement:\n{nl_description}\n\n"
            f"Earlier attempts produced specs with these problems — "
            f"AVOID them all:\n{errors}\n\n"
            "Output ONLY a pure-TLA+ module (no PlusCal, no markdown)."
        )
    else:
        user_content = (
            f"You are revising a TLA+ specification (self-correct attempt {attempt}).\n\n"
            f"Original requirement:\n{nl_description}\n\n"
            f"Module name: {module_name}\n\n"
            f"Validation errors from the previous attempt:\n{errors}\n\n"
            f"Previous (buggy) spec:\n{buggy_spec}\n\n"
            "Fix every error listed above. The corrected spec MUST satisfy "
            "the original requirement. Make real changes — do not return a "
            "byte-identical copy. Output ONLY the corrected pure-TLA+ module "
            "(no PlusCal, no markdown)."
        )
    prompt = _build_harmony_prompt(developer_content, user_content)
    response = client._client.generate(
        model=client.model,
        prompt=prompt,
        raw=True,
        options={
            "temperature": temperature,
            "repeat_penalty": 1.3,
            "num_predict": 4096,
            "top_k": 40,
            "top_p": 0.9,
            "seed": 1000 + attempt,  # vary seed across retries
            "stop": ["<|return|>", "<|end|>", "<|start|>"],
        },
    )
    raw = "---- MODULE" + response["response"]
    if "====" not in raw:
        raw += "\n===="
    return _sanitize_spec(_extract_tla(raw))


SHOT_DUMP_DIR = REPO / "outputs" / "eval" / "3shot_dumps"


def _dump_shot(module: str, attempt: int, spec: str, sany_r, tlc_r, tlaps_r) -> None:
    SHOT_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    p = SHOT_DUMP_DIR / f"{module}_shot{attempt}.tla"
    p.write_text(spec)
    meta = SHOT_DUMP_DIR / f"{module}_shot{attempt}.diag.txt"
    parts = []
    if sany_r is not None:
        parts.append(f"=== SANY (valid={sany_r.valid}) ===\n{sany_r.raw_output}")
    if tlc_r is not None:
        parts.append(
            f"=== TLC (tier={tlc_r.tier} states={getattr(tlc_r, 'distinct_states', 0)}) ===\n"
            f"{tlc_r.raw_output}"
        )
    if tlaps_r is not None:
        parts.append(f"=== TLAPS (tier={tlaps_r.tier}) ===\n{tlaps_r.raw_output}")
    meta.write_text("\n\n".join(parts))


def run_one(client: ChatTLAClient, rec: dict) -> dict:
    desc = rec["topic_desc"]
    module = rec["module"]
    log("")
    log("=" * 72)
    log(f"SPEC: {module}   batch={rec.get('batch', '')}")
    log(f"DESC: {desc[:140]}{'…' if len(desc) > 140 else ''}")
    log("-" * 72)

    spec: str = ""
    prev_spec: str = ""
    last_sany = last_tlc = last_tlaps = None

    # 1 single-shot + MAX_RETRIES retries
    for attempt in range(MAX_RETRIES + 1):
        t0 = time.monotonic()
        try:
            if attempt == 0:
                log(f"[shot 0] generate_spec (single-shot)…")
                spec = client.generate_spec(desc, module_name=module, rag_k=2)
            else:
                feedback = build_feedback(last_sany, last_tlc, last_tlaps)
                fresh = " [FRESH]" if attempt >= MAX_RETRIES else ""
                log(f"[shot {attempt}] self-correct with feedback "
                    f"({len(feedback)} chars, temp={0.50 + 0.20 * (attempt - 1):.2f}){fresh}…")
                spec = self_correct(client, desc, module, spec, feedback, attempt)
        except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            elapsed = time.monotonic() - t0
            log(f"  ✗ KILLED after {elapsed:.1f}s ({type(e).__name__})")
            return {
                "module": module, "batch": rec.get("batch", ""),
                "result": "timeout", "killed_at_attempt": attempt,
                "elapsed_at_kill": round(elapsed, 1),
            }
        except Exception as e:
            elapsed = time.monotonic() - t0
            log(f"  ✗ gen_error after {elapsed:.1f}s: {e}")
            return {
                "module": module, "batch": rec.get("batch", ""),
                "result": "gen_error", "error": str(e),
            }
        gen_elapsed = time.monotonic() - t0
        identical = (attempt > 0 and spec == prev_spec)
        log(f"  generated {len(spec)} chars in {gen_elapsed:.1f}s"
            + ("  ⚠ IDENTICAL to previous shot" if identical else ""))
        prev_spec = spec

        # SANY
        sany_r = sany_validate(spec, module_name=module)
        last_sany = sany_r
        if not sany_r.valid:
            err_count = len(sany_r.errors)
            log(f"  SANY : FAIL ({err_count} errors)")
            if sany_r.errors:
                log(f"    e.g. {sany_r.errors[0][:120]}")
            last_tlc = None
            last_tlaps = None
            _dump_shot(module, attempt, spec, sany_r, None, None)
            continue
        log("  SANY : PASS")

        # TLC
        tlc_r = tlc_validate(spec, module_name=module)
        last_tlc = tlc_r
        log(f"  TLC  : tier={tlc_r.tier}  states={getattr(tlc_r, 'distinct_states', 0)}  "
            f"mut_caught={getattr(tlc_r, 'mutation_caught', False)}")
        if getattr(tlc_r, "tlc_violations", None):
            log(f"    violation: {tlc_r.tlc_violations[0][:120]}")

        # TLAPS
        tlaps_r = tlaps_validate(spec, module_name=module)
        last_tlaps = tlaps_r
        log(f"  TLAPS: tier={tlaps_r.tier}  "
            f"obligations={tlaps_r.obligations_proved}/{tlaps_r.obligations_total}")

        _dump_shot(module, attempt, spec, sany_r, tlc_r, tlaps_r)
        tlc_ok = tlc_r.tier == "gold"
        tlaps_ok = tlaps_r.tier in ("proved", "no_theorems")
        if tlc_ok and tlaps_ok:
            log(f"  ✓ FIXED at shot {attempt}")
            return {
                "module": module, "batch": rec.get("batch", ""),
                "result": "fixed", "attempts_used": attempt + 1,
                "tlc_tier": tlc_r.tier, "tlaps_tier": tlaps_r.tier,
            }

        # Anything left to feed back?
        feedback_preview = build_feedback(None, tlc_r, tlaps_r)
        if not feedback_preview:
            # SANY-passing silver with no actionable feedback — stop retrying.
            log("  (no actionable TLC/TLAPS feedback; stopping retries)")
            return {
                "module": module, "batch": rec.get("batch", ""),
                "result": "no_feedback_stop", "attempts_used": attempt + 1,
                "tlc_tier": tlc_r.tier, "tlaps_tier": tlaps_r.tier,
            }

    # exhausted retries
    log(f"  ✗ EXHAUSTED retries ({MAX_RETRIES + 1} shots)")
    return {
        "module": module, "batch": rec.get("batch", ""),
        "result": "unfixed", "attempts_used": MAX_RETRIES + 1,
        "tlc_tier": getattr(last_tlc, "tier", None),
        "tlaps_tier": getattr(last_tlaps, "tier", None),
        "sany_valid": getattr(last_sany, "valid", False),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("n", nargs="?", type=int, default=5,
                        help="Number of holdout specs to evaluate (default 5)")
    parser.add_argument("--holdout", default=str(HOLDOUT),
                        help="Path to holdout JSONL file")
    parser.add_argument("--model", default=MODEL,
                        help="Ollama model name (default chattla:20b)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (auto-generated if omitted)")
    args = parser.parse_args()

    model_name = args.model
    holdout_path = Path(args.holdout)
    holdout = [json.loads(l) for l in holdout_path.read_text().splitlines() if l.strip()]
    holdout = holdout[:args.n]

    log(f"[eval] model={model_name}  n={len(holdout)}  holdout={holdout_path.name}  "
        f"per-call-timeout={TIMEOUT_S}s  max_retries={MAX_RETRIES}")
    client = ChatTLAClient(model=model_name)
    client._client = ollama.Client(host="http://localhost:11434", timeout=TIMEOUT_S)

    results: list[dict] = []
    t_all = time.monotonic()
    for i, rec in enumerate(holdout, 1):
        log(f"\n### [{i}/{len(holdout)}]")
        results.append(run_one(client, rec))

    log("\n" + "=" * 72)
    log("SUMMARY")
    log("=" * 72)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["result"]] = counts.get(r["result"], 0) + 1
    for k, v in counts.items():
        log(f"  {k:20s} {v}")
    fixed = sum(1 for r in results if r["result"] == "fixed")
    log(f"\n  FIXED {fixed}/{len(results)}  "
        f"({fixed / len(results) * 100:.0f}%)  "
        f"total_wall={time.monotonic() - t_all:.0f}s")

    # Per-domain breakdown
    domain_results: dict[str, dict] = {}
    for r in results:
        domain = r.get("batch", "unknown")
        if domain not in domain_results:
            domain_results[domain] = {"total": 0, "fixed": 0}
        domain_results[domain]["total"] += 1
        if r["result"] == "fixed":
            domain_results[domain]["fixed"] += 1
    if domain_results:
        log("\nPer-domain breakdown:")
        for domain, stats in sorted(domain_results.items()):
            log(f"  {domain:30s} {stats['fixed']}/{stats['total']}")

    if args.output:
        out = Path(args.output)
    else:
        out = REPO / "outputs" / "eval" / "3shot_tlc_tlaps.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": model_name, "n": len(results), "timeout_s": TIMEOUT_S,
         "max_retries": MAX_RETRIES, "holdout": str(holdout_path),
         "results": results, "counts": counts,
         "by_domain": domain_results},
        indent=2,
    ))
    log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
