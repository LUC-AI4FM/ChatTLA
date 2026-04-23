# Diamond SFT v3 — Polaris Job Notes

## Why
- Local RL loop is stuck at holdout TLC=5% for 40+ cycles.
- Root cause (discovered 2026-04-23): DPO pair file is 87% duplicates; only **14 of ~200 topics** ever produce gold specs → model memorizes 14 patterns, no generalization.
- `diamond_sft_v3.jsonl` contains **1053 curated specs across 326 topics**, with **170 prompt_ids the current model has never been trained on** (all `diamond_curated` tier, human-verified).
- SFT on this data directly injects broad topic coverage in one go. Faster than fighting the RL bootstrap.

## What's in the tarball
- `sft_diamond_v3.pbs` — qsub script
- `data/processed/diamond_sft_v3.jsonl` — 1053 training rows (~4.7 MB)
- `SFT_NOTES.md` — this file

## Before you submit
1. Fill in `#PBS -A <YOUR_PROJECT>` with your ALCF allocation code.
2. Decide base: `chattla-20b` (Option A, already there) OR `chattla-20b-rl-merged` (Option B, needs 41 GB rsync but preserves RL progress).
3. Confirm `conda activate chattla` works in your env (the same env your DPO job used).
4. Smoke-test first with `#PBS -q debug` and `--epochs 1` to catch path issues cheaply.

## Expected output
- `outputs/checkpoints_sft_diamond_v3/checkpoint-<N>/` — final LoRA adapter
- Training should converge at low loss (1053 rows × 3 epochs × batch 8 effective ≈ 400 grad steps)
- Job time: 30-60 min on 4× A100 for 3 epochs

## After the job lands
Rsync `checkpoints_sft_diamond_v3/` back to local. Evaluate on `diamond_eval_holdout.jsonl` (30 specs). Current baseline is TLC=5%; realistic target post-SFT is 30-50% given the training data now covers the holdout topic distribution.
