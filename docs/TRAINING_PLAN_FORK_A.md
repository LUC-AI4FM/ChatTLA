# Training Plan — Fork A (Validator-Segregated SFT)

Date: 2026-04-19
Status: Phase 1 complete (data assembly); Phase 2 (SFT) pending

## Motivation

The audit on 2026-04-19 established that the current training corpus is validator-noisy:

- FormaLLM Alpaca task: 0/50 specs pass SANY ([outputs/audit_fm_alpaca.json](../outputs/audit_fm_alpaca.json))
- FormaLLM TLAPS scan: 12.7% fully proved, 26.5% parse errors ([outputs/tlaps_formallm_scan.json](../outputs/tlaps_formallm_scan.json))
- `tlaplus/examples` (canonical corpus): 96% TLC pass on 70 curated specs
  — not yet centered in the training data

Current best checkpoints ([docs/chattla_paper.tex](chattla_paper.tex)):

- **v14 SFT:** 16/30 SANY, 5/30 TLC, 4/30 Diamond
- **repair-GRPO r1:** 9/30 Diamond (+5 over v14)
- **repair-GRPO r2/r3:** regressed, then aborted at data gate

The RL-iteration path doesn't compound. The data path isn't exhausted.

## Plan

Rebuild the supervised corpus around validator-verified `tlaplus/examples`,
segregated by validator target, and train two task-specialized LoRA heads
from the v14 base checkpoint (not from base model — avoid catastrophic
forgetting per [feedback_training_catastrophic.md]).

### Phase 1 — Data assembly (this sprint)

1. **Scrape** `tlaplus/examples` into a labeled JSONL with per-spec SANY /
   TLC / TLAPS / Apalache flags from the upstream manifest.
   Output: [data/processed/tlaplus_examples_labeled.jsonl](../data/processed/tlaplus_examples_labeled.jsonl)

2. **Filter by target validator** into two SFT corpora, excluding the 30
   `diamond_eval_holdout` module names:
   - [data/processed/tlc_target_sft.jsonl](../data/processed/tlc_target_sft.jsonl) — `features.tlc_pass == true`
   - [data/processed/tlaps_target_sft.jsonl](../data/processed/tlaps_target_sft.jsonl) — `features.tlaps_pass == true`

   Builder: [scripts/build_tlaplus_examples_sft.py](../scripts/build_tlaplus_examples_sft.py)

3. **Concatenate with the existing incremental corpus** (required to avoid
   the v14-catastrophe failure mode — no from-scratch SFT on <300 examples):
   - `fork_a_tlc_sft.jsonl`   = diamond_sft_v4 + tlc_target (upweight 2x)
   - `fork_a_tlaps_sft.jsonl` = diamond_sft_v4 + tlaps_target (upweight 2x)

### Phase 2 — Incremental SFT

Two independent LoRA runs from the v14 merged checkpoint:

| Run | Corpus | Target | Expected checkpoint |
|-----|--------|--------|---------------------|
| sft-tlc | fork_a_tlc_sft.jsonl | TLC-clean generation | chattla-20b-tlc |
| sft-tlaps | fork_a_tlaps_sft.jsonl | Proof-carrying generation | chattla-20b-tlaps |

Hyperparams (starting point, same as v14 that worked):

- base = outputs/merged_model_v14
- LoRA r=16, α=32, dropout=0.05, target="all-linear"
- lr=2e-5, 1 epoch, max_length=2048
- per_device_bs=2, grad_accum=4, BF16
- eval every 50 steps, load_best_model_at_end on eval_loss

### Phase 3 — Eval

30-problem Diamond holdout for both heads; report SANY / TLC / Diamond /
TLAPS-proof pass rates per checkpoint. If `chattla-20b-tlc` beats v14 on
TLC pass rate, promote it as the new baseline. If `chattla-20b-tlaps`
produces any verifiable proofs, that's a new capability (v14 has zero).

## Explicitly not doing

- **No more repair-GRPO rounds** until we have more diverse seed prompts —
  the r3 data gate (0/396 Diamond trajectories) is unfixed
- **No retrain from gpt-oss-20b base** on either corpus — violates the
  incremental-training constraint that recovered v13 post-catastrophe
- **No KTO on imbalanced TLC labels** — reproduces the v10 vacuity trap

## Phase 1 actuals (2026-04-19)

From `tlaplus/examples` scrape + manifest labeling:

- `tlaplus_examples_labeled.jsonl`: **316 rows** (all spec-manifest combinations;
  29 repo specs skipped — no manifest.json)
- Upstream label distribution: 130 TLC-pass (41%), 34 TLAPS-pass (11%),
  0 Apalache-pass (not tracked in manifest schema), 35 PlusCal, 125 beginner
- 4 specs dropped as holdout collisions

Target SFT corpora:

- `tlc_target_sft.jsonl`: **129 rows**
- `tlaps_target_sft.jsonl`: **34 rows**

Combined incremental corpora (base + 2× oversample of new):

- `fork_a_tlc_sft.jsonl`:   3 205 base + 258 new = **3 463 rows**
- `fork_a_tlaps_sft.jsonl`: 3 205 base + 68 new  = **3 273 rows**

Both built by [scripts/build_fork_a_corpora.py](../scripts/build_fork_a_corpora.py);
summary in [data/processed/fork_a_corpora_summary.json](../data/processed/fork_a_corpora_summary.json).

## Open questions for Phase 2

- TLAPS target is small (34 new specs). If the SFT run shows no TLAPS-proof
  capability, the follow-up is either (a) include `tlapm` sibling repo
  examples — cloned at `data/external/tlapm/` — or (b) generate synthetic
  proofs with an LLM-teacher and gate with TLAPS.
- v14-base checkpoint path needs to be confirmed on disk
  (`outputs/checkpoints_v14_chk792/` is current best per version audit).
- Decide whether to train the two heads sequentially or in parallel on the
  2×RTX 8000 setup.
