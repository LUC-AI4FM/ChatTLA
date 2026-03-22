# Training pipeline audit (ChatTLA)

End-to-end view of how data becomes a deployable model, and where quality metrics apply.

## 1. Static corpus (offline)

| Step | Location | Output |
|------|----------|--------|
| Scrape / ingest | `src/scraper/*`, `data/raw/` | Raw specs |
| Validate / annotate | `pipeline.py`, `annotate.py` | `data/validated/combined.jsonl` |
| **Dataset build** | `python -m src.training.dataset_builder` | `data/processed/train.jsonl`, `eval.jsonl` |

**dataset_builder** (recommended flags for production):

- `--sany-only` — only records that pass SANY enter task variants (major SANY lever).
- `--include-augmented` — append `data/processed/augmented.jsonl` (RL/self-improve gold + **bugfix**).
- `--include-description-sft` — append `description_sft.jsonl` (holdout-safe; see `data/benchmarks/README_DESCRIPTION_INTEGRATION.md`).
- `--bugfix-oversample 2` — duplicate **bugfix** rows (TLC-error → fix signal).

**Task variants per record:** spec generation, spec completion, invariant gen (where annotations exist).

## 2. RL loop (online) — `scripts/rl_loop.py`

| Phase | What happens |
|-------|----------------|
| **1** | Sample prompts (benchmark + synthetic + `RL*`); optional condensed description injection for mapped BMs. |
| **2** | Generate with Ollama `chattla:20b`; SANY + TLC; tier gold/silver/bronze; multi-attempt. |
| **3** | **Persist:** gold SFT + DPO (gold vs worse) + **bugfix** (silver + TLC feedback → gold target) → `augmented.jsonl`, `dpo_pairs.jsonl`. Dedup by spec hash. |
| **4** | If `accumulated_new >= threshold`: **retrain** (see §3). |
| **5** | Quick benchmark (6 problems) every cycle; full suite every `BENCHMARK_EVERY_N` at **night**. |
| **6** | **Difficulty cap** from `rl_history.jsonl`: SANY unlocks harder prompts; **TLC rate caps** difficulty so hard tasks aren’t over-sampled while TLC is weak. |

**Hugging Face:** After successful merge + GGUF + Ollama, if `HF_TOKEN` is set and not `--no-publish-hf`, runs `python -m src.training.publish_hf` (versioned `gguf/chattla-20b-vN-Q8_0.gguf`, `gguf/Modelfile`, optional `README.md`). State: `data/benchmarks/hf_publish_state.json`.

## 3. Retrain subprocess chain (RL `rebuild_and_retrain`)

1. `dataset_builder` (flags above, incl. `--bugfix-oversample 2`).
2. `train.py` — LoRA SFT, harmony format, VRAM-aware `max_length`.
3. `merge_lora.py` — GPU (both cards) with CPU fallback.
4. `convert_to_gguf.py` — `llama.cpp` convert → `outputs/gguf/chattla-20b-Q8_0.gguf` + local Ollama `chattla:20b`.
5. **`publish_hf`** — Hub upload (optional).

## 4. Self-improve loop — `src/training/self_improve.py`

Separate path: rule-based fixes → augment → retrain → merge → GGUF → **publish if `HF_TOKEN`**. Does not use the same dataset_builder flags as RL by default (caller should run builder separately if needed).

## 5. Quality metrics (what to trust)

| Metric | Use |
|--------|-----|
| **Full benchmark TLC%** | Primary comparison across model versions (see `outputs/benchmark_results_*_full_*.csv`). |
| **Full benchmark SANY%** | Funnel metric. |
| **Quick 6-problem eval** | Trend only; high variance. |
| **phase1 gold / TLC counts** | Health of data collection, not same as benchmark. |
| **TLCEvalCallback (train.py)** | Training-time probe on eval JSONL; not a substitute for benchmark. |

## 6. Artifacts & secrets

| Path | Note |
|------|------|
| `outputs/checkpoints/` | LoRA adapters (gitignored). |
| `outputs/merged_model/` | Merged BF16 (gitignored). |
| `outputs/gguf/` | Local GGUF + Modelfile (gitignored). |
| `HF_TOKEN` | Write access to [EricSpencer00/chattla-20b](https://huggingface.co/EricSpencer00/chattla-20b); set in `.env` or environment. |
| `data/benchmarks/hf_publish_state.json` | **Tracked** — bump `last_published_version` if you upload manually. |

## 7. Manual commands (cheat sheet)

```bash
# Rebuild train JSONL
python -m src.training.dataset_builder --sany-only --include-augmented \
  --include-description-sft --bugfix-oversample 2

# Train
CUDA_VISIBLE_DEVICES=0,1 python -m src.training.train --epochs 1

# Merge + GGUF + Ollama
python -m src.training.merge_lora
python -m src.inference.convert_to_gguf --quant Q8_0

# Hub (after GGUF exists)
HF_TOKEN=... python -m src.training.publish_hf --dry-run   # inspect
HF_TOKEN=... python -m src.training.publish_hf
```

## 8. Implemented follow-ups

- **DPO:** `train.py --dpo-after` runs `train_dpo.run_after_sft` on **gold** rows in `dpo_pairs.jsonl` (≥2 rows). The RL loop adds `--dpo-after` automatically when enough gold pairs exist. Requires working `trl` + `rich>=14` (see `requirements.txt`).
- **Publish gate:** `publish_hf --require-fresh-full-benchmark-hours N` aborts if there is no `outputs/benchmark_results_*_full_*.csv` or the newest is older than *N* hours. RL loop: set `CHATTLA_PUBLISH_REQUIRE_BENCHMARK_HOURS`.
- **Merged weights on Hub:** `publish_hf --upload-merged-model` uploads `outputs/merged_model/` → `merged_bf16/` (~40GB+). RL loop: `CHATTLA_HF_UPLOAD_MERGED=1`.

**Bootstrap:** `./scripts/launch_rl.sh setup` (or `start`) creates `.venv`, installs `requirements.txt`, and loads `.env`. The RL tmux session prepends `.venv/bin` to `PATH` and sources `.env` before `python3 scripts/rl_loop.py`.

See also [`MODEL_QUALITY.md`](MODEL_QUALITY.md).
