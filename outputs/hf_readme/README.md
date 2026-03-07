---
base_model: openai/gpt-oss-20b
language:
- en
license: apache-2.0
library_name: transformers
model_name: ChatTLA-20b
tags:
- tla-plus
- formal-methods
- formal-verification
- code-generation
- trl
- sft
- generated_from_trainer
datasets:
- EricSpencer00/chattla-20b
pipeline_tag: text-generation
---

# ChatTLA-20b (v10)

ChatTLA is a fine-tuned version of [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) specialised in generating **TLA+ formal specifications** — the language used by AWS, Microsoft, and Intel to mathematically verify distributed systems.

Given a plain-English description of a concurrent or distributed system, ChatTLA outputs a complete, syntactically valid TLA+ module including `Init`, `Next`, `Spec`, `TypeOK`, and domain invariants, together with a TLC model-checker configuration block.

---

## Benchmark Results (v10, single-shot)

Evaluated on a handcrafted 20-problem suite covering distributed algorithms, concurrency primitives, and protocol specs. Three tiers:

| Tier | Meaning |
|------|---------|
| 🥇 Gold | SANY parses **and** TLC model-checks clean |
| 🥈 Silver | SANY parses, TLC finds violation or timeout |
| Bronze | SANY parse failure |

### Per-problem results

| ID | Problem | Tier | SANY | TLC | Struct | Time |
|----|---------|------|------|-----|--------|------|
| BM001 | Mutual Exclusion | Bronze | — | — | 1.00 | 3.7s |
| BM002 | Two-Phase Commit | Bronze | — | — | 1.00 | 10.8s |
| BM003 | Dining Philosophers | Bronze | — | — | 1.00 | 6.2s |
| BM004 | Lamport's Bakery Algorithm | 🥈 Silver | ✓ | — | 1.00 | 4.5s |
| BM005 | Producer-Consumer Queue | Bronze | — | — | 0.78 | 3.2s |
| BM006 | Raft Leader Election | Bronze | — | — | 1.00 | 4.1s |
| BM007 | Read-Write Lock | 🥈 Silver | ✓ | — | 1.00 | 3.8s |
| BM008 | Distributed Snapshot (Chandy-Lamport) | Bronze | — | — | 1.00 | 7.5s |
| BM009 | Token Ring | 🥇 **Gold** | ✓ | ✓ | 0.89 | 1.9s |
| BM010 | Simple Key-Value Store | Bronze | — | — | 1.00 | 1.7s |
| BM011 | Paxos Single-Decree | 🥈 Silver | ✓ | — | 0.44 | 44.6s |
| BM012 | Bounded Retransmission Protocol | Bronze | — | — | 0.78 | 8.1s |
| BM013 | Transaction Isolation (Snapshot) | Bronze | — | — | 1.00 | 8.4s |
| BM014 | Clock Synchronisation | Bronze | — | — | 0.44 | 44.7s |
| BM015 | Peterson's Algorithm | Bronze | — | — | 0.78 | 3.0s |
| BM016 | Gossip Protocol | Bronze | — | — | 1.00 | 1.7s |
| BM017 | Simple Allocator | Bronze | — | — | 0.89 | 4.0s |
| BM018 | Publish-Subscribe Broker | Bronze | — | — | 1.00 | 3.9s |
| BM019 | Dekker's Algorithm | 🥇 **Gold** | ✓ | ✓ | 1.00 | 3.9s |
| BM020 | Eventually Consistent Counter | 🥈 Silver | ✓ | — | 0.44 | 44.8s |

**SANY pass: 6/20 (30%) · TLC pass: 2/20 (10%) · Avg structural: 0.87**

### Version history (single-shot, 20 problems)

| Version | SANY | TLC | Avg Structural |
|---------|------|-----|----------------|
| v6 | 4/20 (20%) | 1/20 (5%) | 0.75 |
| v7 | 6/20 (30%) | 1/20 (5%) | 0.71 |
| v8 | 8/20 (40%) | 1/20 (5%) | 0.72 |
| v9 | 6/20 (30%) | 3/20 (15%) | 0.86 |
| v9 best-of-5 + self-correct | 16/20 (80%) | 5/20 (25%) | 0.88 |
| **v10** | **6/20 (30%)** | **2/20 (10%)** | **0.87** |

> Single-shot scores are conservative. With `--attempts 5 --self-correct` v9 reaches 80% SANY / 25% TLC.

---

## Quick Start

### Ollama (recommended)

```bash
# Pull and run directly
ollama run EricSpencer00/chattla-20b

# Or use the bundled Modelfile
curl -L https://huggingface.co/EricSpencer00/chattla-20b/resolve/main/gguf/Modelfile -o Modelfile
ollama create chattla:20b -f Modelfile
ollama run chattla:20b "Write a TLA+ spec for a token ring with N nodes."
```

### Python (transformers)

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="EricSpencer00/chattla-20b",
    device_map="auto",
)

prompt = (
    "Write a complete TLA+ specification for a two-phase commit protocol "
    "with one coordinator and N participants."
)
result = pipe([{"role": "user", "content": prompt}], max_new_tokens=1024, return_full_text=False)
print(result[0]["generated_text"])
```

### llama.cpp / GGUF

```bash
# Download GGUF
huggingface-cli download EricSpencer00/chattla-20b \
    gguf/chattla-20b-v10-Q8_0.gguf \
    --local-dir ./chattla

# Run with llama.cpp
./llama-cli -m chattla/gguf/chattla-20b-v10-Q8_0.gguf \
    -n 1024 --temp 0.4 \
    -p "Write a TLA+ spec for mutual exclusion with N processes."
```

---

## Model Details

| Property | Value |
|----------|-------|
| Base model | openai/gpt-oss-20b |
| Parameters | 20.9B |
| Architecture | GptOss (sliding + full attention) |
| Fine-tuning method | SFT (LoRA → merged) |
| Context length | 4096 (trained) / 131072 (base) |
| GGUF quantisation | Q8_0 (~21 GB) |
| Training date | March 2026 |

### System prompt

The model is prompted with:

```
You are ChatTLA, an expert at writing verified TLA+ formal specifications.
When asked to write a TLA+ spec, follow these rules exactly:
1. Start the module with ---- MODULE <ModuleName> ----
2. End with ====
3. Include EXTENDS, VARIABLES, Init, Next, and Spec operators
4. After the TLA+ module, append a TLC configuration block:
   SPECIFICATION Spec
   INVARIANT TypeOK   (if TypeOK is defined)
5. Output only valid TLA+ code. No markdown fences, no explanation outside the spec.
```

---

## Training

Fine-tuned with SFT on a curated dataset of TLA+ specifications scraped from GitHub, augmented with handcrafted examples covering distributed algorithms, concurrency primitives, and protocol verification.

### Training configuration

| Setting | Value |
|---------|-------|
| Method | SFT with LoRA |
| TRL | 0.28.0 |
| Transformers | 5.2.0 |
| PyTorch | 2.10.0 |
| Datasets | 4.5.0 |
| Tokenizers | 0.22.2 |
| Hardware | 2× Quadro RTX 8000 (48 GB each) |

---

## Files

```
EricSpencer00/chattla-20b
├── config.json              # Model architecture
├── tokenizer.json           # Tokenizer
├── tokenizer_config.json
├── chat_template.jinja      # Chat template
├── pytorch_model.bin        # Full BF16 weights (39 GB)
├── generation_config.json
└── gguf/
    ├── chattla-20b-v10-Q8_0.gguf   # Quantised GGUF for Ollama / llama.cpp
    └── Modelfile                    # Ollama Modelfile
```

---

## Intended Use

ChatTLA is designed for:
- Rapid prototyping of TLA+ specifications from natural-language system descriptions
- Educational exploration of formal methods
- Assisting engineers who are learning TLA+

**Not intended for:** safety-critical or production verification without human review. Always validate generated specs with SANY and TLC before relying on them.

---

## Citation

```bibtex
@misc{chattla2026,
  title   = {ChatTLA: Fine-tuned LLM for TLA+ Formal Specification Generation},
  author  = {Spencer, Eric},
  year    = {2026},
  url     = {https://huggingface.co/EricSpencer00/chattla-20b},
}
```
