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
- grpo
- reinforcement-learning
- generated_from_trainer
datasets:
- EricSpencer00/chattla-20b
pipeline_tag: text-generation
---

# ChatTLA-20b (v15)

ChatTLA is a fine-tuned version of [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) specialised in generating **TLA+ formal specifications** — the language used by AWS, Microsoft, and Intel to mathematically verify distributed systems.

Given a plain-English description of a concurrent or distributed system, ChatTLA outputs a complete, syntactically valid TLA+ module including `Init`, `Next`, `Spec`, `TypeOK`, and domain invariants, together with a TLC model-checker configuration block.

---

## Benchmark Results (v15, 3-shot self-correct)

Evaluated on a 30-spec held-out suite spanning communication protocols, concurrency primitives, consensus, data structures, memory/caches, mutual exclusion, classical puzzles, scheduling, transactions, and workflow state machines. Each spec gets up to 3 self-correction attempts using TLC error feedback. Tiers are defined by what the spec actually does under SANY and TLC, not just whether it parses:

| Tier | Meaning |
|------|---------|
| 💎 Diamond | Gold **and** TLC explores ≥1 distinct state, has a non-trivial invariant, and the invariant catches a mutation |
| 🥇 Gold | SANY parses **and** TLC model-checks clean |
| 🥈 Silver | SANY parses, TLC finds violation or timeout |
| Bronze | SANY parse failure |

Diamond is the headline metric: it's the only tier that proves the spec is *semantically* useful rather than just syntactically valid.

### Per-spec results (30-spec holdout)

| # | Batch | Module | Tier | Diamond |
|---|-------|--------|------|:------:|
|  1 | communication_protocols | AlternatingBit            | Bronze      |    |
|  2 | communication_protocols | Arp                       | Bronze      |    |
|  3 | communication_protocols | AtomicRegister            | Bronze      |    |
|  4 | concurrency_primitives  | BinarySemaphore           | Bronze      |    |
|  5 | concurrency_primitives  | Channel                   | Bronze      |    |
|  6 | concurrency_primitives  | CountDownLatch            | Bronze      |    |
|  7 | consensus_election      | AtomicCommit              | Bronze      |    |
|  8 | consensus_election      | BullyElection             | 🥇 Gold     | 💎 |
|  9 | consensus_election      | ByzantineQuorum           | Bronze      |    |
| 10 | data_structures         | BinaryHeap                | Bronze      |    |
| 11 | data_structures         | BloomCounter              | 🥇 Gold     | 💎 |
| 12 | data_structures         | BloomFilter               | ⏱ Timeout   |    |
| 13 | memory_caches           | ArenaAllocator            | 🥇 Gold     | 💎 |
| 14 | memory_caches           | BuddyAllocator            | Bronze      |    |
| 15 | memory_caches           | CopyingGc                 | Bronze      |    |
| 16 | mutual_exclusion        | AdaptiveMutex             | 🥇 Gold     | 💎 |
| 17 | mutual_exclusion        | AndersonMutex             | 🥇 Gold     | 💎 |
| 18 | mutual_exclusion        | AravindMutex              | ⏱ Timeout   |    |
| 19 | puzzles_classical       | BlocksWorld               | Bronze      |    |
| 20 | puzzles_classical       | ChessKingMoves            | Bronze      |    |
| 21 | puzzles_classical       | ColoredHats               | Bronze      |    |
| 22 | scheduling_resources    | AdmissionControl          | 🥇 Gold     | 💎 |
| 23 | scheduling_resources    | BackpressureChannel       | 🥇 Gold     | 💎 |
| 24 | scheduling_resources    | Bankers                   | ⏱ Timeout   |    |
| 25 | transactions_databases  | ChainReplication          | ⏱ Timeout   |    |
| 26 | transactions_databases  | DistributedLock           | Bronze      |    |
| 27 | transactions_databases  | FencingToken              | Bronze      |    |
| 28 | workflows_state_machines| ContentModeration         | 🥇 Gold     | 💎 |
| 29 | workflows_state_machines| DocumentApproval          | 🥇 Gold     | 💎 |
| 30 | workflows_state_machines| EmailVerification         | Bronze      |    |

**Diamond: 9/30 (30%) · Gold: 9/30 (30%)**

### Per-domain breakdown

| Domain | Diamond |
|--------|:-------:|
| communication_protocols | 0/3 |
| concurrency_primitives | 0/3 |
| consensus_election | 1/3 |
| data_structures | 1/3 |
| memory_caches | 1/3 |
| mutual_exclusion | 2/3 |
| puzzles_classical | 0/3 |
| scheduling_resources | 2/3 |
| transactions_databases | 0/3 |
| workflows_state_machines | 2/3 |

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
| v14 (Diamond SFT) | 30-spec holdout (single-shot) | 16/30 (53%) | 5/30 (17%) | 4/30 (13%) |
| **v15 (Repair GRPO)** | **30-spec holdout (3-shot)** | 9/30 (30%) | 9/30 (30%) | **9/30 (30%)** |

> v15 applies repair-based GRPO (Group Relative Policy Optimization) on top of v14's Diamond SFT weights. The model learns to fix its own broken specs by training on (broken → repaired) trajectory pairs with TLC-graded improvement reward. v15 eval uses 3-shot self-correction with TLC error feedback, matching realistic usage; v14 was evaluated single-shot, so SANY/TLC rates are not directly comparable. Diamond is the metric to track going forward.

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
    gguf/chattla-20b-v15-Q8_0.gguf \
    --local-dir ./chattla

# Run with llama.cpp
./llama-cli -m chattla/gguf/chattla-20b-v15-Q8_0.gguf \
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
| Fine-tuning method | Diamond SFT (LoRA) → Repair GRPO (LoRA) → merged |
| Context length | 2048 (trained) / 131072 (base) |
| GGUF quantisation | Q8_0 (~22 GB) |
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

### Phase 1: Diamond SFT (v14)

v14 was produced by the **Diamond curation pipeline**: candidate TLA+ specs are generated by an earlier checkpoint, then graded by a tlc_validator that checks SANY parsing, TLC state-space exploration, non-trivial invariants, and mutation-test sensitivity. Specs that survive grading are LLM-judged for chain-of-thought quality, leaving a curated training pool (209 raw → 73 curated for the v14 SFT round). The model is fine-tuned with LoRA on this pool and merged.

### Phase 2: Repair GRPO (v15)

v15 applies **repair-based GRPO** (Group Relative Policy Optimization) on top of the v14 checkpoint. The key insight: instead of training on gold-standard specs alone, the model learns to *fix broken specs* using TLC error feedback as reward signal.

**Pipeline:**
1. **Trajectory collection** — the v14 model generates specs for 398 problems with up to 6 repair iterations each, producing (broken, repaired) pairs scored by a multi-stage validator (SANY → TLC → Apalache → TLAPS).
2. **Dataset filtering** — pairs are filtered to keep the "learnable middle": `min_before_score=0.10` (drop unparseable) and `max_before_score=0.80` (drop already-good), yielding ~430 gradable pairs centered on score ≈ 0.45.
3. **GRPO training** — 300 steps, 4 generations per prompt, max 384 completion tokens. The reward is the improvement delta: `after_score - before_score`, normalized by group. Learning rate 3e-6, KL penalty β=0.02, temperature 0.5.
4. **LoRA merge** — best checkpoint (around step 140–160 where reward peaked) merged back into full weights.

Reward peaked at steps 140–160 with `reward_std ≈ 0.25` (vs 0.0 in prior full-spec GRPO attempts that had zero variance). This was the first successful RL run on TLA+ spec generation.

**R2 regression and R3 (in progress).** A second flywheel round (R2) continued GRPO from v15's merged weights on a freshly harvested dataset and regressed to 6/30 (20%). Post-mortem: the Phase 2 merge deduped pairs on `(nl[:80], round(before_score, 1))`, a score-bucket width of 0.1 that collapsed most of the learnable-middle band; combined with a raised `min_before_score = 0.10`, the usable training set fell from 433 → 179 pairs, shifted hard (mean before_score 0.26 → 0.42), and the model overtrained past its 150-step peak over 300 steps. Regressions concentrated in `mutual_exclusion` and `workflows_state_machines` (2/3 lost each). R3 pulls only the data and step-budget levers: dedup key widened to `(nl[:120], round(before_score, 2))`, score floor restored to 0.02, `--max-iters` raised 6 → 9 to grow the raw pool, and `--max-steps` cut to 175 with a checkpoint picker that selects the save closest to step 150. v15 remains the production checkpoint until R3 beats 9/30.

DPO/KTO refinement was used in v11–v13 but was deprecated in the Diamond overhaul: 0/484 specs from those preference-trained checkpoints actually passed Diamond, indicating the model had learned TLA+ syntax without learning semantics.

### Training configuration

| Setting | Value |
|---------|-------|
| SFT method | LoRA (lora_dropout=0) |
| GRPO method | LoRA, 4 generations, 384 max completion |
| GRPO learning rate | 3e-6 |
| GRPO KL β | 0.02 |
| GRPO steps | 300 (best checkpoint ~150) |
| Max sequence length | 2048 |
| TRL | 0.28.0 |
| Transformers | 5.2.0 |
| PyTorch | 2.10.0 |
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
    ├── chattla-20b-v15-Q8_0.gguf   # Quantised GGUF for Ollama / llama.cpp
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
