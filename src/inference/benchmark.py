"""
benchmark.py — Evaluate ChatTLA on the 20-problem handcrafted benchmark suite.

Runs TLA+ spec generation for each benchmark problem against:
  1. Base gpt-oss:20b (baseline)
  2. ChatTLA fine-tuned model (chattla:20b)

Scores are aggregated across three dimensions:
  - sany_pass   : SANY parsing succeeds (syntax correct TLA+)
  - tlc_pass    : TLC model-checks with no violations
  - structural  : Heuristic structural rubric (has required operators, etc.)

Results are written to outputs/benchmark_results.csv and logged to MLflow.
The notebook notebooks/03_evaluation.ipynb reads this CSV for plotting.

Usage
-----
    # Compare base vs fine-tuned:
    python -m src.inference.benchmark

    # Fine-tuned model only:
    python -m src.inference.benchmark --model chattla:20b

    # With TLC self-correction (up to 3 retries on TLC failure):
    python -m src.inference.benchmark --self-correct
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import mlflow

_REPO_ROOT     = Path(__file__).resolve().parents[2]
_BENCH_JSON    = _REPO_ROOT / "data" / "benchmarks" / "benchmark_suite.json"
_RESULTS_CSV   = _REPO_ROOT / "outputs" / "benchmark_results.csv"

_MODELS = {
    "base":      "gpt-oss:20b",
    "chattla":   "chattla:20b",
}

_CSV_FIELDS = [
    "model", "benchmark_id", "name", "domain", "difficulty",
    "sany_pass", "tlc_pass", "structural_score",
    "tlc_tier", "runtime_s", "generated_spec",
    # Per-component verdicts (DeepSeek-Prover-V2 inspired denser scoring).
    # Each is 0/1 except partial_credit which is the weighted mean.
    "init_present", "next_present", "init_level_ok", "next_level_ok",
    "invariants_declared", "tlc_depth1_ok", "partial_credit",
    "expected_invariant_overlap",  # how many of the benchmark's expected invariants the generated spec named
    "plan_used",                    # 1 when plan-then-spec was used and stage 1 succeeded
]


def score_structural(spec: str, expected_invariants: list[str]) -> float:
    """
    Heuristic structural rubric — 0.0 to 1.0.
    Checks for: module delimiters, EXTENDS, VARIABLES, Init, Next, Spec,
    TypeOK, and at least one expected invariant name.
    """
    checks = [
        bool(re.search(r"----\s*MODULE", spec)),
        bool(re.search(r"====", spec)),
        bool(re.search(r"\bEXTENDS\b", spec)),
        bool(re.search(r"\bVARIABLES\b", spec)),
        bool(re.search(r"^Init\s*==", spec, re.MULTILINE)),
        bool(re.search(r"^Next\s*==", spec, re.MULTILINE)),
        bool(re.search(r"^Spec\s*==", spec, re.MULTILINE)),
        bool(re.search(r"^TypeOK\s*==", spec, re.MULTILINE)),
        any(re.search(rf"^{re.escape(inv)}\s*==", spec, re.MULTILINE) for inv in expected_invariants),
    ]
    return sum(checks) / len(checks)


_TIER_RANK = {"gold": 3, "silver": 2, "bronze": 1}


def _run_single_attempt(
    problem: dict,
    client,
    use_self_correct: bool,
    use_plan: bool = False,
):
    """
    Run one generation attempt for a benchmark problem.

    Returns a dict with: spec, tier, structural, semantic (or None),
    plan_used (bool), expected_invariant_overlap (int).
    """
    from src.validators.tlc_validator import validate_string

    description = problem["description"]
    if problem.get("hints"):
        description += f"\n\nHints: {problem['hints']}"

    plan_used = False
    plan_obj = None
    semantic = None

    if use_self_correct:
        spec, tier = client.validate_and_generate(description, max_retries=3)
        # validate_and_generate doesn't return semantic; re-run validate_string
        # to capture per-component verdicts for the chosen final spec.
        m = re.search(r"----\s*MODULE\s+(\w+)", spec)
        module_name = m.group(1) if m else "Generated"
        try:
            tlc_result = validate_string(spec, module_name=module_name)
            semantic = tlc_result.semantic
        except Exception:
            semantic = None
    else:
        if use_plan:
            plan_obj, spec = client.generate_with_plan(description)
            plan_used = plan_obj is not None
        else:
            spec = client.generate_spec(description)
        m = re.search(r"----\s*MODULE\s+(\w+)", spec)
        module_name = m.group(1) if m else "Generated"
        tlc_result = validate_string(spec, module_name=module_name)
        tier = tlc_result.tier
        semantic = tlc_result.semantic

    structural = score_structural(spec, problem.get("expected_invariants", []))

    # How many of the benchmark's expected invariants does the generated spec name?
    expected_invs = problem.get("expected_invariants", []) or []
    overlap = sum(
        1 for inv in expected_invs
        if re.search(rf"^{re.escape(inv)}\s*==", spec, re.MULTILINE)
    )

    return {
        "spec": spec,
        "tier": tier,
        "structural": structural,
        "semantic": semantic,
        "plan_used": plan_used,
        "expected_invariant_overlap": overlap,
    }


def run_benchmark_problem(
    problem: dict,
    model_tag: str,
    use_self_correct: bool = False,
    attempts: int = 1,
    use_plan: bool = False,
) -> dict:
    """Run a single benchmark problem with N attempts, keeping the best result.

    Best is now scored on (tier_rank, partial_credit, structural) — partial
    credit breaks ties between two same-tier specs so an attempt that nails
    more components wins over one that just got lucky on top-level structure.
    """
    from src.inference.ollama_client import ChatTLAClient

    client = ChatTLAClient(model=model_tag, reasoning="medium")
    t0 = time.monotonic()

    best = None  # holds the best _run_single_attempt result dict

    for i in range(attempts):
        try:
            if i > 0:
                client._temp_override = 0.05 + i * 0.10
            else:
                client._temp_override = None

            cur = _run_single_attempt(problem, client, use_self_correct, use_plan=use_plan)

            cur_pc = cur["semantic"].partial_credit if cur["semantic"] else 0.0
            cur_key = (
                _TIER_RANK.get(cur["tier"], 0),
                cur_pc,
                cur["structural"],
            )
            if best is None:
                best = cur
                best_key = cur_key
            elif cur_key > best_key:
                best = cur
                best_key = cur_key

            if cur["tier"] == "gold":
                break
        except Exception:
            continue

    elapsed = time.monotonic() - t0
    client._temp_override = None

    if best is None:
        raise RuntimeError(f"All {attempts} attempts failed for {problem['id']}")

    sem = best["semantic"]
    return {
        "model":             model_tag,
        "benchmark_id":      problem["id"],
        "name":              problem["name"],
        "domain":            problem["domain"],
        "difficulty":        problem["difficulty"],
        "sany_pass":         int(best["tier"] in ("silver", "gold")),
        "tlc_pass":          int(best["tier"] == "gold"),
        "structural_score":  round(best["structural"], 3),
        "tlc_tier":          best["tier"],
        "runtime_s":         round(elapsed, 2),
        "generated_spec":    best["spec"][:8000],
        "init_present":         int(bool(sem and sem.init_present)),
        "next_present":         int(bool(sem and sem.next_present)),
        "init_level_ok":        int(bool(sem and sem.init_level_ok)),
        "next_level_ok":        int(bool(sem and sem.next_level_ok)),
        "invariants_declared":  int(bool(sem and sem.invariants_declared)),
        "tlc_depth1_ok":        int(bool(sem and sem.tlc_depth1_ok)),
        "partial_credit":       round(sem.partial_credit, 4) if sem else 0.0,
        "expected_invariant_overlap": best["expected_invariant_overlap"],
        "plan_used":             int(bool(best["plan_used"])),
    }


def run(
    models: Optional[list[str]] = None,
    output_csv: Path = _RESULTS_CSV,
    use_self_correct: bool = False,
    limit: Optional[int] = None,
    problem_ids: Optional[list[str]] = None,
    attempts: int = 1,
    use_plan: bool = False,
) -> None:
    with _BENCH_JSON.open() as f:
        problems = json.load(f)

    # optionally restrict to a subset for faster testing
    if problem_ids is not None:
        problems = [p for p in problems if p["id"] in problem_ids]
    if limit is not None:
        problems = problems[:limit]

    if models is None:
        models = list(_MODELS.values())

    mlflow.set_experiment("ChatTLA-Benchmark")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with mlflow.start_run(run_name="benchmark"):
        for model_tag in models:
            sany_passes = 0
            tlc_passes  = 0
            print(f"\n[benchmark] Model: {model_tag}")

            for problem in problems:
                att_label = f" (best of {attempts})" if attempts > 1 else ""
                print(f"  [{problem['id']}] {problem['name']}{att_label}...", end="", flush=True)
                try:
                    row = run_benchmark_problem(
                        problem, model_tag, use_self_correct,
                        attempts=attempts, use_plan=use_plan,
                    )
                    rows.append(row)
                    sany_passes += row["sany_pass"]
                    tlc_passes  += row["tlc_pass"]
                    status = (
                        f"tier={row['tlc_tier']} pc={row['partial_credit']:.2f} "
                        f"struct={row['structural_score']:.2f} t={row['runtime_s']}s"
                    )
                    if row.get("plan_used"):
                        status += " plan✓"
                    print(f" {status}")
                except Exception as exc:
                    print(f" ERROR: {exc}")
                    rows.append({k: None for k in _CSV_FIELDS} | {
                        "model": model_tag,
                        "benchmark_id": problem["id"],
                        "name": problem["name"],
                    })

            n = len(problems)
            model_rows = [r for r in rows if r.get("model") == model_tag]
            pc_mean = (
                sum((r.get("partial_credit") or 0.0) for r in model_rows) / max(1, len(model_rows))
            )
            depth1_pass = sum(int(r.get("tlc_depth1_ok") or 0) for r in model_rows)
            mlflow.log_metrics({
                f"{model_tag}/sany_pass_rate":     sany_passes / n,
                f"{model_tag}/tlc_pass_rate":      tlc_passes  / n,
                f"{model_tag}/depth1_pass_rate":   depth1_pass / n,
                f"{model_tag}/partial_credit_mean": pc_mean,
            })
            print(
                f"\n[benchmark] {model_tag}: sany={sany_passes}/{n}  "
                f"tlc={tlc_passes}/{n}  depth1={depth1_pass}/{n}  "
                f"partial_credit={pc_mean:.3f}"
            )

    # Write CSV
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[benchmark] Results written to {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ChatTLA benchmark suite")
    parser.add_argument("--model",        default=None,  help="Single model tag (default: run all)")
    parser.add_argument("--self-correct", action="store_true", help="Enable TLC self-correction loop")
    parser.add_argument("--output",       default=str(_RESULTS_CSV))
    parser.add_argument("--limit",        type=int, help="Stop after N benchmark problems (for quick smoke tests)")
    parser.add_argument("--problems",     nargs="+", help="Run only the given benchmark IDs (e.g. BM001 BM007)")
    parser.add_argument("--attempts",     type=int, default=1, help="Number of generation attempts per problem (best-of-N)")
    parser.add_argument("--use-plan",     action="store_true",
                        help="Use two-stage plan-then-spec generation (DeepSeek-Prover-V2 style). "
                             "Roughly doubles wall-time per problem; falls back to single-shot if "
                             "the planning JSON cannot be parsed.")
    args = parser.parse_args()

    model_list = [args.model] if args.model else None
    run(models=model_list, output_csv=Path(args.output), use_self_correct=args.self_correct,
        limit=args.limit, problem_ids=args.problems, attempts=args.attempts,
        use_plan=args.use_plan)
