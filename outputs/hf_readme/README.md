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

# ChatTLA-20b (v14)

ChatTLA is a fine-tuned version of [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) specialised in generating **TLA+ formal specifications** — the language used by AWS, Microsoft, and Intel to mathematically verify distributed systems.

Given a plain-English description of a concurrent or distributed system, ChatTLA outputs a complete, syntactically valid TLA+ module including `Init`, `Next`, `Spec`, `TypeOK`, and domain invariants, together with a TLC model-checker configuration block.

---

## Benchmark Results (v14, single-shot)

Evaluated on a 30-spec held-out suite spanning communication protocols, concurrency primitives, consensus, data structures, memory/caches, mutual exclusion, classical puzzles, scheduling, transactions, and workflow state machines. Tiers are defined by what the spec actually does under SANY and TLC, not just whether it parses:

| Tier | Meaning |
|------|---------|
| 💎 Diamond | Gold **and** TLC explores ≥1 distinct state, has a non-trivial invariant, and the invariant catches a mutation |
| 🥇 Gold | SANY parses **and** TLC model-checks clean |
| 🥈 Silver | SANY parses, TLC finds violation or timeout |
| Bronze | SANY parse failure |

Diamond is the headline metric: it's the only tier that proves the spec is *semantically* useful rather than just syntactically valid. v13 and earlier tracked SANY/TLC only, which made trivial `TRUE` invariants count as Gold.

### Per-spec results (30-spec holdout)

| # | Batch | Module | Tier | Diamond | States | Invs | Mut |
|---|-------|--------|------|:------:|:-----:|:----:|:---:|
|  1 | communication_protocols | AlternatingBit            | 🥇 Gold     | 💎 | 4  | 1 | ✓ |
|  2 | communication_protocols | Arp                       | Bronze      |    | 0  | 0 |   |
|  3 | communication_protocols | AtomicRegister            | Bronze      |    | 0  | 0 |   |
|  4 | concurrency_primitives  | BinarySemaphore           | 🥈 Silver   |    | 0  | 0 |   |
|  5 | concurrency_primitives  | Channel                   | 🥇 Gold     | 💎 | 4  | 2 | ✓ |
|  6 | concurrency_primitives  | CountDownLatch            | Bronze      |    | 0  | 0 |   |
|  7 | consensus_election      | AtomicCommit              | 🥇 Gold     |    | 0  | 1 | ✓ |
|  8 | consensus_election      | BullyElection             | 🥈 Silver   |    | 0  | 0 |   |
|  9 | consensus_election      | ByzantineQuorum           | Bronze      |    | 0  | 0 |   |
| 10 | data_structures         | BinaryHeap                | 🥈 Silver   |    | 0  | 0 |   |
| 11 | data_structures         | BloomCounter              | 🥈 Silver   |    | 0  | 0 |   |
| 12 | data_structures         | BloomFilter               | Bronze      |    | 0  | 0 |   |
| 13 | memory_caches           | ArenaAllocator            | 🥇 Gold     | 💎 | 4  | 1 | ✓ |
| 14 | memory_caches           | BuddyAllocator            | Bronze      |    | 0  | 0 |   |
| 15 | memory_caches           | CopyingGc                 | 🥈 Silver   |    | 0  | 0 |   |
| 16 | mutual_exclusion        | AdaptiveMutex             | Bronze      |    | 0  | 0 |   |
| 17 | mutual_exclusion        | AndersonMutex             | 🥇 Gold     | 💎 | 36 | 1 | ✓ |
| 18 | mutual_exclusion        | AravindMutex              | Bronze      |    | 0  | 0 |   |
| 19 | puzzles_classical       | BlocksWorld               | 🥈 Silver   |    | 0  | 0 |   |
| 20 | puzzles_classical       | ChessKingMoves            | Bronze      |    | 0  | 0 |   |
| 21 | puzzles_classical       | ColoredHats               | Bronze      |    | 0  | 0 |   |
| 22 | scheduling_resources    | AdmissionControl          | Bronze      |    | 0  | 0 |   |
| 23 | scheduling_resources    | BackpressureChannel       | 🥈 Silver   |    | 0  | 0 |   |
| 24 | scheduling_resources    | Bankers                   | 🥈 Silver   |    | 0  | 0 |   |
| 25 | transactions_databases  | ChainReplication          | Bronze      |    | 0  | 0 |   |
| 26 | transactions_databases  | DistributedLock           | 🥈 Silver   |    | 0  | 0 |   |
| 27 | transactions_databases  | FencingToken              | 🥈 Silver   |    | 0  | 0 |   |
| 28 | workflows_state_machines| ContentModeration         | Bronze      |    | 0  | 0 |   |
| 29 | workflows_state_machines| DocumentApproval          | Bronze      |    | 0  | 0 |   |
| 30 | workflows_state_machines| EmailVerification         | 🥈 Silver   |    | 0  | 0 |   |

**SANY pass: 16/30 (53%) · TLC pass: 5/30 (17%) · Diamond: 4/30 (13%)**

### Version history

| Version | Suite | SANY | TLC | Diamond / Notes |
|---------|-------|------|-----|-----------------|
| v6  | 20-problem handcraft     | 4/20 (20%)  | 1/20 (5%)  | — |
| v7  | 20-problem handcraft     | 6/20 (30%)  | 1/20 (5%)  | — |
| v8  | 20-problem handcraft     | 8/20 (40%)  | 1/20 (5%)  | — |
| v9  | 20-problem handcraft     | 6/20 (30%)  | 3/20 (15%) | — |
| v9 best-of-5 + self-correct | 20-problem handcraft | 16/20 (80%) | 5/20 (25%) | — |
| v10 | 20-problem handcraft     | 6/20 (30%)  | 2/20 (10%) | — |
| v11 | 20-problem handcraft     | 6/20 (30%)  | 2/20 (10%) | — |
| v13 (SFT + DPO) | 20-problem handcraft | 9/20 (45%) | 5/20 (25%) | not measured (trivial invariants counted as Gold) |
| **v14 (Diamond SFT)** | **30-spec holdout** | **16/30 (53%)** | **5/30 (17%)** | **4/30 (13%)** |

> v14 is the first checkpoint trained against the Diamond pipeline (rejection-sampled + curated specs with chain-of-thought, real semantic checks). The benchmark suite also changed from a 20-problem handcrafted set to a broader 30-spec holdout, so absolute SANY/TLC rates aren't directly comparable to v13. Diamond is the metric to track going forward.

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
    gguf/chattla-20b-v14-Q8_0.gguf \
    --local-dir ./chattla

# Run with llama.cpp
./llama-cli -m chattla/gguf/chattla-20b-v14-Q8_0.gguf \
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
| Fine-tuning method | Diamond rejection-sampling SFT (LoRA → merged) |
| Context length | 2048 (trained) / 131072 (base) |
| GGUF quantisation | Q8_0 (~21 GB) |
| Training date | April 2026 |

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

v14 was produced by the **Diamond curation pipeline**: candidate TLA+ specs are generated by an earlier checkpoint, then graded by a tlc_validator that checks SANY parsing, TLC state-space exploration, non-trivial invariants, and mutation-test sensitivity. Specs that survive grading are LLM-judged for chain-of-thought quality, leaving a curated training pool (209 raw → 73 curated for the v14 SFT round). The model is then fine-tuned with LoRA on this pool and merged.

DPO/KTO refinement was used in v11–v13 but was deprecated in the Diamond overhaul: 0/484 specs from those preference-trained checkpoints actually passed Diamond, indicating the model had learned TLA+ syntax without learning semantics. The Diamond pipeline trains directly on examples that *do* pass semantic checks.

### Training configuration

| Setting | Value |
|---------|-------|
| Method | SFT with LoRA (lora_dropout=0) |
| Max sequence length | 2048 |
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
    ├── chattla-20b-v14-Q8_0.gguf   # Quantised GGUF for Ollama / llama.cpp
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
