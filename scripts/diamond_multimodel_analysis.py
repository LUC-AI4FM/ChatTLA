#!/usr/bin/env python3
"""Analyze multi-model Diamond benchmark CSV results.

Reads outputs/diamond_multimodel/results.csv and prints:
  1) Tier distribution per (model, regime) with Diamond rate + Wilson 95% CI
  2) Failure-mode breakdown per (model, regime) for non-Diamond rows
  3) Piecewise delta table per model (single_shot vs piecewise)
  4) Cross-model ranking by Diamond pass rate with model-class annotation

Also writes the same text to outputs/diamond_multimodel/analysis.txt.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "outputs" / "diamond_multimodel" / "results.csv"
DEFAULT_ANALYSIS = REPO_ROOT / "outputs" / "diamond_multimodel" / "analysis.txt"


def wilson_95(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def model_class(model: str) -> str:
    if model == "chattla:20b" or model.startswith("chattla:"):
        return "fine-tuned"
    if model == "gpt-oss:20b":
        return "base"
    return "general"


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(r: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze multi-model Diamond benchmark CSV")
    p.add_argument("--input", default=str(DEFAULT_RESULTS), help="Input results CSV")
    p.add_argument("--output", default=str(DEFAULT_ANALYSIS), help="Output analysis text file")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"Input CSV not found: {in_path}")

    rows: list[dict] = []
    with in_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    {
                        "model": (r.get("model") or "").strip(),
                        "regime": (r.get("regime") or "").strip(),
                        "problem_id": (r.get("problem_id") or "").strip(),
                        "tier": int(r.get("tier") or 0),
                    }
                )
            except ValueError:
                continue

    if not rows:
        raise SystemExit(f"No rows found in {in_path}")

    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_group[(r["model"], r["regime"])].append(r)

    report_parts: list[str] = []
    report_parts.append(f"Input: {in_path}")
    report_parts.append(f"Rows: {len(rows)}")
    report_parts.append("")

    # 1) Tier distribution table
    tier_rows: list[list[str]] = []
    tier_stats: dict[tuple[str, str], dict] = {}
    for (model, regime) in sorted(by_group.keys()):
        grp = by_group[(model, regime)]
        n = len(grp)
        counts = {i: 0 for i in range(5)}
        for r in grp:
            t = r["tier"]
            if t < 0:
                t = 0
            if t > 4:
                t = 4
            counts[t] += 1

        k = counts[4]
        p = k / n if n else 0.0
        lo, hi = wilson_95(k, n)
        tier_stats[(model, regime)] = {
            "n": n,
            "counts": counts,
            "k": k,
            "p": p,
            "lo": lo,
            "hi": hi,
            "half_width": (hi - lo) / 2.0,
        }

        tier_rows.append(
            [
                model,
                regime,
                str(n),
                str(counts[0]),
                str(counts[1]),
                str(counts[2]),
                str(counts[3]),
                str(counts[4]),
                fmt_pct(p),
                f"[{fmt_pct(lo)}, {fmt_pct(hi)}]",
            ]
        )

    report_parts.append("Tier Distribution (by model/regime)")
    report_parts.append(
        make_table(
            ["model", "regime", "n", "t0", "t1", "t2", "t3", "t4", "diamond%", "wilson95"],
            tier_rows,
        )
    )
    report_parts.append("")

    # 2) Failure mode breakdown for non-diamond specs
    failure_rows: list[list[str]] = []
    for (model, regime) in sorted(by_group.keys()):
        st = tier_stats[(model, regime)]
        counts = st["counts"]
        non = st["n"] - counts[4]
        if non <= 0:
            d1 = d2 = d3 = d4 = 0.0
        else:
            d1 = counts[0] / non
            d2 = counts[1] / non
            d3 = counts[2] / non
            d4 = counts[3] / non

        failure_rows.append(
            [
                model,
                regime,
                str(non),
                fmt_pct(d1),
                fmt_pct(d2),
                fmt_pct(d3),
                fmt_pct(d4),
            ]
        )

    report_parts.append("Failure Modes (non-Diamond only)")
    report_parts.append(
        make_table(
            ["model", "regime", "non_diamond_n", "%fail_D1", "%fail_D2", "%fail_D3", "%fail_D4"],
            failure_rows,
        )
    )
    report_parts.append("")

    # 3) Piecewise delta table
    models = sorted({r["model"] for r in rows})
    delta_rows: list[list[str]] = []
    for model in models:
        ss = tier_stats.get((model, "single_shot"))
        pw = tier_stats.get((model, "piecewise"))
        if not ss or not pw:
            delta_rows.append([model, "NA", "NA", "NA", "NA", "NA"])
            continue

        p_ss = ss["p"]
        p_pw = pw["p"]
        delta = p_pw - p_ss
        noise_floor = ss["half_width"] + pw["half_width"]
        exceeds = abs(delta) > noise_floor

        delta_rows.append(
            [
                model,
                fmt_pct(p_ss),
                fmt_pct(p_pw),
                f"{delta * 100.0:+.1f}pp",
                f"{noise_floor * 100.0:.1f}pp",
                "yes" if exceeds else "no",
            ]
        )

    report_parts.append("Piecewise Delta (Diamond pass rate)")
    report_parts.append(
        make_table(
            [
                "model",
                "single_shot",
                "piecewise",
                "delta",
                "noise_floor",
                "delta_exceeds_noise",
            ],
            delta_rows,
        )
    )
    report_parts.append("")

    # 4) Cross-model comparison ranked by Diamond pass rate, by regime
    for regime in sorted({r["regime"] for r in rows}):
        ranking: list[tuple[str, float, int, float, float]] = []
        for model in models:
            st = tier_stats.get((model, regime))
            if not st:
                continue
            ranking.append((model, st["p"], st["n"], st["lo"], st["hi"]))
        ranking.sort(key=lambda x: (-x[1], -x[2], x[0]))

        rank_rows: list[list[str]] = []
        for i, (model, p, n, lo, hi) in enumerate(ranking, start=1):
            rank_rows.append(
                [
                    str(i),
                    model,
                    model_class(model),
                    str(n),
                    fmt_pct(p),
                    f"[{fmt_pct(lo)}, {fmt_pct(hi)}]",
                ]
            )

        report_parts.append(f"Cross-Model Ranking ({regime})")
        report_parts.append(
            make_table(["rank", "model", "class", "n", "diamond%", "wilson95"], rank_rows)
        )
        report_parts.append("")

    report = "\n".join(report_parts).rstrip() + "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
