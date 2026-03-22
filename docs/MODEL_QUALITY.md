# Model quality targets (ChatTLA fine-tuning)

## What to optimize for (in order)

1. **TLC pass rate** on the **full** benchmark suite — end-to-end model checking (finite models, configs, invariants). This is the product metric for “valid + checkable” specs.
2. **SANY pass rate** — necessary gate: invalid syntax never reaches TLC.
3. **Structural heuristics** — diagnostic only; do not tune the loop primarily on structural score.

**Quick eval** (default **12** problems per cycle in `scripts/rl_loop.py`, configurable with `--quick-eval-limit`) is a **trend signal**; it is noisy. Use **full-suite** CSVs under `outputs/benchmark_results_*_full_*.csv` when comparing models after retrains. By default the loop uses **`--cycle-hours 0`** (no padding sleep between cycles); use `--cycle-hours 1.5` if you want spaced cycles.

## Training levers (implemented in-repo)

| Mechanism | Role |
|-----------|------|
| `dataset_builder --sany-only` | Drop non–SANY-valid rows from the validated corpus before building tasks. |
| **Augmented archive** | `data/processed/augmented.jsonl` is **append-only** across RL cycles — nothing is deleted when benchmarks move 30% → 50%. |
| **Best-per-prompt merge** (default) | Rebuild collapses augmented rows to **one row per `_prompt_id`** (or user-text hash), keeping the **highest tier** (gold ≈ bugfix > silver > bronze). New gold replaces old silver for the same prompt. |
| **Silver SFT** (optional) | RL can append **silver** (SANY pass, TLC fail) when that prompt has no gold yet; use `--no-silver-augmented` if TLC quality drops. **Bronze** stays off unless `--no-gold-only-augmented`. |
| **`bugfix` tier** (oversampled) | TLC-error → gold-fix examples get **2×** weight in `train.jsonl` (see `--bugfix-oversample`). |
| Description SFT | Holdout-safe description → `.tla` supervision from `build_description_sft_jsonl.py`. |
| RL prompt enrichment | Condensed descriptions for mapped benchmarks (test-time context). |
| **Difficulty cap** | Derived from recent **SANY** and **TLC** rates in `rl_history.jsonl` so hard prompts aren’t emphasized while TLC is still weak. |

## DPO / bugfix

- DPO pairs (gold vs worse) live in `data/processed/rl/dpo_pairs.jsonl` when training supports them.
- Bugfix rows teach recovery from **real TLC feedback** strings.

## Hugging Face Hub

After each successful RL retrain (merge + GGUF + Ollama), the loop runs **`src.training.publish_hf`** when `HF_TOKEN` is set. It uploads versioned `gguf/chattla-20b-vN-Q8_0.gguf`, `gguf/Modelfile`, and patches `README.md` from `outputs/hf_readme/README.md` (plus latest **full** benchmark CSV if present). Disable with `python scripts/rl_loop.py --no-publish-hf`. State file: `data/benchmarks/hf_publish_state.json`.

Public model: [EricSpencer00/chattla-20b](https://huggingface.co/EricSpencer00/chattla-20b).

## Operational checklist

- After changing `augmented.jsonl` or descriptions: rebuild with  
  `python -m src.training.dataset_builder --sany-only --include-augmented --include-description-sft --bugfix-oversample 2`  
  (omit `--no-silver-augmented` / `--no-augmented-best-per-prompt` unless you want the old behavior)
- Run a **full** benchmark (`python -m src.inference.benchmark ...` without `--limit`) before claiming a win on TLC.
- For Hub releases: ensure `HF_TOKEN` is in `.env`; optional `python -m src.training.publish_hf --dry-run` to see next version.
