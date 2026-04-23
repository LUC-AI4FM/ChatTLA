# ChatTLA

Fine-tuning [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) to generate verifiably correct [TLA+](https://lamport.azurewebsites.net/tla/tla.html) formal specifications.

**Key differentiator**: TLC model checking is the primary training metric — not perplexity. A generated spec must pass the TLA+ model checker before it counts as a success.

**Model:** [EricSpencer00/chattla-20b](https://huggingface.co/EricSpencer00/chattla-20b) (Apache 2.0)

---

## Quick Start

**Ollama (recommended):**
```bash
ollama run EricSpencer00/chattla-20b
```

**Python (transformers):**
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="EricSpencer00/chattla-20b", device_map="auto")
result = pipe([{"role": "user", "content": "Write a TLA+ spec for two-phase commit."}], max_new_tokens=1024)
print(result[0]["generated_text"][-1]["content"])
```

**GGUF / llama.cpp:**
```bash
huggingface-cli download EricSpencer00/chattla-20b gguf/chattla-20b-v15-Q8_0.gguf --local-dir ./chattla
./llama-cli -m chattla/gguf/chattla-20b-v15-Q8_0.gguf -n 1024 --temp 0.4
```

---

## Benchmark Results (v15)

| Tier | Rate | Criterion |
|------|------|-----------|
| Diamond | 9/30 (30%) | Parses + model-checks + non-trivial invariants |
| Gold    | 9/30 (30%) | Parses + model-checks cleanly |

Evaluated on a 30-spec held-out suite with up to 3 self-correction attempts via TLC feedback.

---

## Project Structure

```
ChatTLA/
├── data/
│   ├── FormaLLM/        ← git submodule: seed specs (MIT)
│   ├── processed/       ← train.jsonl + eval.jsonl (harmony-formatted)
│   └── benchmarks/
│       └── benchmark_suite.json  ← 30 hand-crafted eval problems
├── src/
│   ├── scraper/         ← Phase 1: data collection
│   ├── validators/      ← TLC/SANY wrappers
│   ├── training/        ← Phase 2: fine-tuning
│   └── inference/       ← Phase 3: deployment & eval
├── scripts/             ← RL loop, pipeline, and training scripts
├── configs/
│   └── accelerate_fsdp.yaml
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_finetuning.ipynb
│   └── 03_evaluation.ipynb
└── outputs/
    ├── eval/            ← benchmark result JSON files
    └── diamond_gen/     ← curated diamond-tier training data
```

---

## Setup

```bash
# 1. Clone and initialise submodules
git clone https://github.com/LUC-AI4FM/ChatTLA.git
cd ChatTLA
git submodule update --init --recursive

# 2. Create virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env: set HF_TOKEN (for publishing), GITHUB_TOKEN_1 (for scraping)

# 4. Verify Java (required for TLC/SANY validator)
java -version    # needs Java 8+
```

---

## Phases

### Phase 1 — Data Collection

```bash
# FormaLLM seed only (fast, no GitHub token needed):
python -m src.scraper.pipeline --no-github --no-annotate

# Full scrape (several hours, needs GITHUB_TOKEN_1):
python -m src.scraper.pipeline
```

**Output:** `data/validated/combined.jsonl`

### Phase 2 — Fine-Tuning

```bash
# Smoke test (10 steps):
python -m src.training.train --smoke-test

# Full training:
python -m src.training.train

# Monitor with MLflow:
mlflow ui --port 5000

# Merge LoRA into base:
python -m src.training.merge_lora
```

**Primary metric:** `tlc/tlc_clean_rate` > 0.70 on eval set.

### Phase 3 — Inference & Evaluation

```bash
# Convert to GGUF and register with Ollama:
python -m src.inference.convert_to_gguf

# Run 30-problem benchmark:
python -m src.inference.benchmark

# Quick spec generation:
python -m src.inference.ollama_client "A distributed read-write lock." --validate
```

---

## Model Details

| Property | Value |
|----------|-------|
| Base model | `openai/gpt-oss-20b` |
| Architecture | MoE, 21B total / 3.6B active |
| Quantisation | MXFP4 → dequantised to BF16 for training |
| PEFT | LoRA rank 8, all-linear + MoE expert layers |
| Prompt format | [harmony](https://github.com/openai/harmony) (required) |
| License | Apache 2.0 |

---

## Validation Tiers

| Tier | Criterion | Training use |
|------|-----------|-------------|
| **diamond** | TLC model-checks with non-trivial invariants | Primary GRPO reward signal |
| **gold** | TLC model-checks with zero violations | Primary SFT data |
| **silver** | SANY parses (syntactically valid) | Training data (labelled) |
| **bronze** | SANY fails | Rejected; kept for error analysis |

---

## Benchmark Suite

30 hand-crafted TLA+ problems in `data/benchmarks/benchmark_suite.json`:

| Domain | Count | Examples |
|--------|-------|---------|
| Consensus | 4 | Paxos, Raft leader election, Bakery, Dekker's |
| Scheduling | 4 | Mutex, Dining Philosophers, Peterson's, Token Ring |
| Transactions | 2 | Two-Phase Commit, Snapshot Isolation |
| Storage | 5 | KV Store, Allocator, G-Counter CRDT, Producer-Consumer, RW Lock |
| Networking | 5 | Chandy-Lamport, Gossip, BRP, Publish-Subscribe, Clock Sync |

Difficulty ranges from 2 (beginner) to 5 (research-grade).

---

## Research Notes

- **TLC as reward signal:** `TLCEvalCallback` runs TLC at every `eval_steps` and logs `tlc_clean_rate` to MLflow — the experiment's primary metric.
- **Repair-based GRPO:** The RL loop uses improvement reward (did repair fix the spec?) rather than absolute pass/fail, giving non-zero reward variance even on hard problems.
- **Self-annotation:** Local Ollama `gpt-oss:20b` annotates specs — zero cost, fully air-gapped. See `src/scraper/annotate.py`.
- **Harmony format** is mandatory for gpt-oss — applied at `dataset_builder.py` time.
- **Deduplication:** MinHash LSH (Jaccard ≥ 0.8) via `datasketch` prevents near-duplicate specs from inflating training quality.
- **License hygiene:** GitHub scraper filters to MIT/Apache/BSD/ISC/Unlicense only.

See [`docs/MODEL_QUALITY.md`](docs/MODEL_QUALITY.md) for quality targets and [`docs/TRAINING_PIPELINE_AUDIT.md`](docs/TRAINING_PIPELINE_AUDIT.md) for the full pipeline map.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | For publishing | Hugging Face Hub write token |
| `GITHUB_TOKEN_1` | For Phase 1 scraping | GitHub API token (30 req/min authenticated) |
| `OLLAMA_HOST` | Optional | Ollama daemon URL (default: `http://localhost:11434`) |
| `CHATTLA_MODEL_DIR` | Optional | External path for merged model weights (default: `outputs/`) |
| `CUDA_VISIBLE_DEVICES` | Optional | GPU assignment |

Copy `.env.example` to `.env` and fill in values. Never commit `.env`.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

Seed data from [FormaLLM](https://github.com/LUC-FMitF/FormaLLM) (MIT).
