# TLA descriptions ↔ training / RL / benchmarks

This doc explains how **`data/derived/tla_descriptions.json`**, **`benchmark_to_module.json`**, and SFT / RL interact so benchmark numbers stay interpretable.

## Files

| File | Role |
|------|------|
| `data/derived/tla_descriptions.json` | Per-module structured descriptions (narrative + technical) harvested from Examples-style specs. |
| `data/benchmarks/benchmark_to_module.json` | Maps `BM001`… to a **module name** in `tlaplus/Examples` (or `null`). `holdout_module_names` lists modules tied to benchmark eval tasks. |
| `data/processed/description_sft.jsonl` | **Train** SFT examples: condensed description → local `.tla` target (built by `scripts/build_description_sft_jsonl.py`). **Excludes** holdout modules. |
| `data/processed/description_sft_holdout.jsonl` | Same format but **only** holdout rows (for ablations / contamination checks — **not** merged into `train.jsonl`). |

## Holdout (contamination control)

- Modules listed in **`holdout_module_names`** in `benchmark_to_module.json` are **not** included in `description_sft.jsonl`.
- RL **does** inject a condensed description for mapped benchmarks at **prompt** time (see `scripts/rl_loop.py` → `load_prompt_bank`). That is *test-time context*, not supervised targets on the eval prompt text in SFT.
- For strict apples-to-apples: compare runs with the same `benchmark_suite.json` and note whether description SFT + RL enrichment were enabled.

## Regenerate description SFT

From repo root:

```bash
python3 scripts/build_description_sft_jsonl.py
```

Then rebuild the main training JSONL (includes description SFT when flag is set):

```bash
python3 -m src.training.dataset_builder --sany-only --include-augmented --include-description-sft
```

The RL loop’s retrain step passes `--include-description-sft` automatically; if `description_sft.jsonl` is missing, the builder logs a warning and continues without it.

## Interpretability

- **SFT on descriptions** teaches the model to map rich text → valid specs for **non-holdout** modules.
- **Benchmark eval** still uses the benchmark task text (+ optional injected reference block in RL); improving SANY/TLC pass rates is the intended metric, not BLEU on descriptions.
