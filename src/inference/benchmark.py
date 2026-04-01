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
    "base":      "deepseek-r1:8b",
    "chattla":   "chattla:8b",
}

_CSV_FIELDS = [
    "model", "benchmark_id", "name", "domain", "difficulty",
    "sany_pass", "tlc_pass", "structural_score",
    "tlc_tier", "runtime_s", "generated_spec",
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
) -> tuple[str, str, float]:
    """
    Run one generation attempt for a benchmark problem.
    Returns (spec, tier, structural_score).
    """
    from src.validators.tlc_validator import validate_string

    description = problem["description"]
    if problem.get("hints"):
        description += f"\n\nHints: {problem['hints']}"

    if use_self_correct:
        spec, tier = client.validate_and_generate(description, max_retries=3)
    else:
        spec = client.generate_spec(description)
        m = re.search(r"----\s*MODULE\s+(\w+)", spec)
        module_name = m.group(1) if m else "Generated"
        tlc_result = validate_string(spec, module_name=module_name)
        tier = tlc_result.tier

    structural = score_structural(spec, problem.get("expected_invariants", []))
    return spec, tier, structural


def run_benchmark_problem(
    problem: dict,
    model_tag: str,
    use_self_correct: bool = False,
    attempts: int = 1,
) -> dict:
    """Run a single benchmark problem with N attempts, keeping the best result."""
    from src.inference.ollama_client import ChatTLAClient

    client = ChatTLAClient(model=model_tag, reasoning="medium")
    t0 = time.monotonic()

    best_spec, best_tier, best_struct = None, "bronze", 0.0

    for i in range(attempts):
        try:
            # Vary temperature slightly across attempts for diversity
            if i > 0:
                client._temp_override = 0.05 + i * 0.10  # 0.15, 0.25, ...
            else:
                client._temp_override = None

            spec, tier, structural = _run_single_attempt(
                problem, client, use_self_correct
            )

            # Pick the best: highest tier, then highest structural score
            tier_rank = _TIER_RANK.get(tier, 0)
            best_rank = _TIER_RANK.get(best_tier, 0)
            if (tier_rank > best_rank) or \
               (tier_rank == best_rank and structural > best_struct):
                best_spec, best_tier, best_struct = spec, tier, structural

            # Early exit if we hit gold
            if tier == "gold":
                break
        except Exception:
            continue  # skip failed attempts, try again

    elapsed = time.monotonic() - t0
    client._temp_override = None  # cleanup

    if best_spec is None:
        # All attempts failed
        raise RuntimeError(f"All {attempts} attempts failed for {problem['id']}")

    return {
        "model":             model_tag,
        "benchmark_id":      problem["id"],
        "name":              problem["name"],
        "domain":            problem["domain"],
        "difficulty":        problem["difficulty"],
        "sany_pass":         int(best_tier in ("silver", "gold")),
        "tlc_pass":          int(best_tier == "gold"),
        "structural_score":  round(best_struct, 3),
        "tlc_tier":          best_tier,
        "runtime_s":         round(elapsed, 2),
        "generated_spec":    best_spec[:8000],  # truncate for CSV
    }


def run(
    models: Optional[list[str]] = None,
    output_csv: Path = _RESULTS_CSV,
    use_self_correct: bool = False,
    limit: Optional[int] = None,
    problem_ids: Optional[list[str]] = None,
    attempts: int = 1,
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
                    row = run_benchmark_problem(problem, model_tag, use_self_correct, attempts=attempts)
                    rows.append(row)
                    sany_passes += row["sany_pass"]
                    tlc_passes  += row["tlc_pass"]
                    status = f"tier={row['tlc_tier']} struct={row['structural_score']:.2f} t={row['runtime_s']}s"
                    print(f" {status}")
                except Exception as exc:
                    print(f" ERROR: {exc}")
                    rows.append({k: None for k in _CSV_FIELDS} | {
                        "model": model_tag,
                        "benchmark_id": problem["id"],
                        "name": problem["name"],
                    })

            n = len(problems)
            mlflow.log_metrics({
                f"{model_tag}/sany_pass_rate": sany_passes / n,
                f"{model_tag}/tlc_pass_rate":  tlc_passes  / n,
            })
            print(f"\n[benchmark] {model_tag}: sany={sany_passes}/{n}  tlc={tlc_passes}/{n}")

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
    args = parser.parse_args()

    model_list = [args.model] if args.model else None
    run(models=model_list, output_csv=Path(args.output), use_self_correct=args.self_correct,
        limit=args.limit, problem_ids=args.problems, attempts=args.attempts)
