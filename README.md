# ChatTLA

Fine-tuning [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) to generate verifiably correct [TLA+](https://lamport.azurewebsites.net/tla/tla.html) formal specifications.

**Key differentiator**: TLC model checking as the primary training metric — not perplexity.  A generated spec must pass the TLA+ model checker before it counts as a success.

**Environment:** `./scripts/launch_rl.sh setup` (or `start`) creates `.venv`, runs `pip install -r requirements.txt`, and loads `.env`. The tmux session uses `.venv/bin/python` and exports `.env` so `HF_TOKEN` applies to the RL loop. For an interactive shell: `source .venv/bin/activate` then `set -a && source .env && set +a`.

---
```
FormaLLM (205)  ──┐
GitHub scrape ────┼─→ validated/combined.jsonl (gold/silver/bronze)
                  │
                  ├─→ processed/augmented.jsonl (RL-generated, deduplicated)
                  │
                  ├─→ derived/tla_descriptions.json (module summaries)
                  │
                  └─→ dataset_builder.py
                      └─→ processed/train.jsonl (295 examples)
                      └─→ processed/eval.jsonl (4 examples)
                      └─→ processed/rl/dpo_pairs.jsonl (18 gold pairs)

    ↓ Training ↓

outputs/checkpoints/ (SFT LoRA adapter)
          ↓ train_dpo.py (optional) ↓
outputs/checkpoints_dpo/ (DPO LoRA adapter)
          ↓ merge_lora.py (SFT + DPO) ↓
outputs/merged_model/ (merged BF16)
          ↓ convert_to_gguf.py ↓
outputs/gguf/chattla-20b-Q8_0.gguf (21 GB)
          ↓ Register with Ollama ↓

RL Loop continuously:
  - Ollama client generates specs
  - TLC validator classifies
  - New examples → augmented.jsonl
  - At 50 gold: rebuild + retrain
```
---

## Project Structure

```
ChatTLA/
├── data/
│   ├── FormaLLM/        ← git submodule: 205 seed specs (MIT)
│   ├── raw/             ← unvalidated records per scrape source
│   ├── validated/       ← gold + silver tier specs with annotations
│   ├── rejected/        ← bronze tier (kept for analysis)
│   ├── processed/       ← train.jsonl + eval.jsonl (harmony-formatted)
│   └── benchmarks/
│       └── benchmark_suite.json  ← 20 hand-crafted eval problems
├── src/
│   ├── scraper/         ← Phase 1: data collection
│   │   ├── ingest_formalllm.py
│   │   ├── github_agent.py
│   │   ├── dedup_agent.py
│   │   ├── annotate.py
│   │   └── pipeline.py       ← run this
│   ├── validators/      ← TLC/SANY wrappers
│   │   ├── sany_validator.py
│   │   ├── tlc_validator.py
│   │   └── quality_scorer.py
│   ├── training/        ← Phase 2: fine-tuning
│   │   ├── dataset_builder.py
│   │   ├── lora_config.yaml
│   │   ├── train.py           ← run this
│   │   ├── tlc_eval_callback.py
│   │   ├── train_dpo.py
│   │   ├── merge_lora.py
│   │   └── publish_hf.py
│   ├── inference/       ← Phase 3: deployment & eval
│   │   ├── ollama_client.py
│   │   ├── benchmark.py
│   │   └── convert_to_gguf.py
│   └── shared/
│       ├── tlc/tla2tools.jar  ← bundled TLC v1.8.0
│       └── schemas/dataset_schema.py
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_finetuning.ipynb
│   └── 03_evaluation.ipynb
└── outputs/
    ├── checkpoints/     ← LoRA adapter checkpoints
    ├── merged_model/    ← merged BF16 model (post merge_lora.py)
    ├── gguf/            ← Q4_K_M + Q8_0 GGUF files
    └── logs/            ← timestamped pipeline run logs
```

---

## Setup

```bash
# 1. Clone and initialise submodules
git clone <repo>
git submodule update --init --recursive     # pulls FormaLLM into data/FormaLLM/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env with GitHub tokens (for scraping)
cp .env.example .env
# Edit .env: set GITHUB_TOKEN_1, OLLAMA_HOST, etc.

# 4. Verify Java (required for TLC/SANY)
java -version    # needs Java 8+

# 5. Pull base model via Ollama
ollama pull gpt-oss:20b
```

---

## Phases

### Phase 1 — Data Collection

```bash
# FormaLLM seed only (fast, no GitHub key needed):
python -m src.scraper.pipeline --dry-run

# Full scrape (several hours, needs GITHUB_TOKEN_1):
python -m src.scraper.pipeline

# No annotation (no local Ollama needed):
python -m src.scraper.pipeline --no-github --no-annotate
```

**Output**: `data/validated/combined.jsonl`

**Data quality targets**:
- FormaLLM seed: 205 records, mostly gold-tier
- After GitHub scrape: ≥10k gold, ≥50k silver

### Phase 2 — Fine-Tuning

```bash
# Validate setup (10 steps, fast):
CUDA_VISIBLE_DEVICES=1 python -m src.training.train --smoke-test

# Full training (several hours on GPU 1):
CUDA_VISIBLE_DEVICES=1 python -m src.training.train

# Monitor with MLflow:
mlflow ui --port 5000

# Optional DPO refinement (requires gold preference pairs):
python -m src.training.train_dpo --checkpoint outputs/checkpoints/checkpoint-155

# After training — merge LoRA into base (auto-detects DPO checkpoint):
CUDA_VISIBLE_DEVICES=0,1 python -m src.training.merge_lora
```

**Hardware**: Quadro RTX 8000 (49 GB), device index 1. Merge uses both GPUs.
**Primary metric**: `tlc/tlc_clean_rate` > 0.70 on eval set.

### Phase 3 — Inference & Evaluation

```bash
# Convert to GGUF and register with Ollama:
python -m src.inference.convert_to_gguf

# Test inference:
ollama run chattla:20b "Write a TLA+ spec for two-phase commit."

# Run 20-problem benchmark (base vs fine-tuned):
python -m src.inference.benchmark

# Quick spec generation from Python:
python -m src.inference.ollama_client "A distributed read-write lock." --validate
```

---

## Model Details

| Property | Value |
|----------|-------|
| Base model | `openai/gpt-oss-20b` |
| Architecture | MoE, 21B total / 3.6B active |
| Quantisation | MXFP4 → dequantised to BF16 for training |
| PEFT | LoRA rank 8, all-linear + MoE expert layers (blocks 7, 15, 23) |
| Prompt format | gpt-oss [harmony](https://github.com/openai/harmony) (required) |
| License | Apache 2.0 |
| GPU | Quadro RTX 8000 × 1 (GPU index 1), 49 GB VRAM |

---

## Validation Tiers

| Tier | Criterion | Training use |
|------|-----------|-------------|
| **gold** | TLC model-checks with zero violations | Primary training data |
| **silver** | SANY parses (syntactically valid TLA+) | Training data (labelled) |
| **bronze** | SANY fails | Rejected; kept in `data/rejected/` for error analysis |

---

## Benchmark Suite

20 hand-crafted TLA+ problems in `data/benchmarks/benchmark_suite.json`:

| Domain | Count | Examples |
|--------|-------|---------|
| Consensus | 4 | Paxos, Raft leader election, Bakery, Dekker's |
| Scheduling | 4 | Mutex, Dining Philosophers, Peterson's, Token Ring |
| Transactions | 2 | Two-Phase Commit, Snapshot Isolation |
| Storage | 5 | KV Store, Allocator, G-Counter CRDT, Producer-Consumer, RW Lock |
| Networking | 5 | Chandy-Lamport, Gossip, BRP, Publish-Subscribe, Clock Sync |

Difficulty: 2 (beginner) — 5 (research-grade).

---

## Research Notes

- **Model quality targets** (TLC → SANY → structure): see [`docs/MODEL_QUALITY.md`](docs/MODEL_QUALITY.md). Full pipeline map: [`docs/TRAINING_PIPELINE_AUDIT.md`](docs/TRAINING_PIPELINE_AUDIT.md). Training uses SANY-filtered corpus data, gold-only RL SFT, **2×** oversampling of TLC **bugfix** examples (`--bugfix-oversample`), and a TLC-aware RL difficulty cap. **Hugging Face:** set `HF_TOKEN` so the RL loop uploads versioned GGUF + Modelfile to [EricSpencer00/chattla-20b](https://huggingface.co/EricSpencer00/chattla-20b) after each retrain (`--no-publish-hf` to skip).
- **Self-annotation**: We use local Ollama `gpt-oss:20b` (not GPT-4o) for NL annotation of specs — zero cost, fully air-gapped.  See `src/scraper/annotate.py`.
- **TLC as reward signal**: `TLCEvalCallback` runs TLC at every `eval_steps` and logs `tlc_clean_rate` to MLflow.  This is the experiment's primary metric.
- **Harmony format** is mandatory for gpt-oss — applied at `dataset_builder.py` time so every JSONL record is already formatted.
- **Deduplication**: MinHash LSH (Jaccard ≥ 0.8) via `datasketch` prevents near-duplicate specs from inflating training set quality.
- **License hygiene**: GitHub scraper filters to MIT/Apache/BSD/ISC/Unlicense only.  GPL specs are kept in a separate `data/rejected/` bucket.
