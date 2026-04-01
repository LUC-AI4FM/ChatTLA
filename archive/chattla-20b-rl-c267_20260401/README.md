# Archive: chattla:20b RL Loop (267 cycles)

**Date**: 2026-04-01
**Reason**: Model degraded below base gpt-oss:20b performance after 267 RL cycles.

## Final State
- 267 cycles completed
- Latest benchmark: 0/20 SANY pass, 0/20 TLC pass (all bronze)
- 1,218 augmented training examples, 22 DPO pairs
- 295 training examples (including augmented)
- Model quality collapsed — worse than base gpt-oss:20b

## Contents
- `augmented.jsonl` — RL-generated training data (1,218 examples)
- `dpo_pairs.jsonl` — DPO preference pairs (22 pairs)
- `state.json` — RL state (32 accumulated, 38 gold prompts)
- `train.jsonl` — Final training dataset (295 examples)
- `eval.jsonl` — Evaluation dataset (4 examples)
- `rl_loop.log` — Full loop log
- `rl_history.jsonl` — Per-cycle stats (267 cycles)
- `tlc_errors.jsonl` — Detailed TLC error log
- `benchmark_results_current.csv` — Final benchmark results

## Lessons
- RL loop drifted into "deep dark forest" — self-reinforcing bad examples
- Training on model's own poor outputs degraded quality cycle over cycle
- Need better quality gates and possibly different base (FormaLLM paper suggests DeepSeek r1:8b)
