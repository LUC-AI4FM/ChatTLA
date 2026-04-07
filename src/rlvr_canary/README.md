# RLVR canary

The TLA+ generation task is *hard*: small corpus, formal language, sparse
reward. The risk in jumping straight to RL on it is that any failure has too
many possible causes — bad reward, bad training stack, bad LoRA config, bad
base model, bad data. This canary isolates **the RL training stack** by
running it on a known-easy task with a published baseline.

## Three phases

| Phase | Base                              | Adapter | Task         | Pass criterion                                |
|-------|-----------------------------------|---------|--------------|-----------------------------------------------|
| 1     | Llama-3.2-1B-Instruct (full FT)   | none    | GSM8K        | clear upward reward curve, ≥45% pass@1        |
| 2     | Llama-3.2-1B-Instruct             | LoRA    | GSM8K        | within 5pp of Phase 1 final pass@1            |
| 3     | (TBD: Qwen3-1.7B / DeepSeek-R1-1.5B) | LoRA | TLA+ via per-action TLC | reward curve lifts at all                |

Only advance a phase after the prior phase passes. If Phase 2 regresses, the
problem is LoRA config; if Phase 3 doesn't lift, the problem is the TLA+
reward shaping (not the RL stack).

The kalomaze "RL Learning with LoRA" post (memory:
`reference_lora_rl_blogs.md`) lists specific gotchas that show up at each
phase boundary — re-read it before transitioning.

## Phase 1: run it

```bash
# Smoke test (~1 minute, no checkpoint)
python -m scripts.train_canary_gsm8k --smoke

# Real canary run
python -m scripts.train_canary_gsm8k --max-steps 300 --train-limit 2048
```

The reward function is in [`reward.py`](reward.py). Three-tier shaping:
`1.0` correct, `0.1` parses-but-wrong, `0.0` unparseable. The 0.1 step
prevents collapse to no-answer output during early training; remove it
once the model is reliably emitting `<answer>...</answer>`.

## What "passing" looks like

Watch the `reward` series in TRL's logs. A healthy Phase 1 run looks like:

  - Steps 0–10:    mean reward ≈ 0.05–0.15 (mostly format bonus)
  - Steps 30–60:   mean reward ≈ 0.25–0.40 (verifier reward starting to fire)
  - Steps 100+:    mean reward ≈ 0.45–0.65, completion length stabilizes

If the curve is flat at 0.1 for >50 steps, the verifier reward isn't
reaching the model — usually a tokenization / chat-template mismatch.
If the curve is flat at 0.0, format reward isn't firing — check the
`<answer>` tag regex against actual completions.

## Why these defaults

- **Llama-3.2-1B-Instruct**: small enough to full-FT on a single Quadro
  RTX 8000, well-supported in TRL, no surprise tokenizer behaviour. The
  1B size also echoes the FormaLLM finding (`docs/formallm.md`) that small
  reasoning-aligned models beat 70B siblings on TLA+ — same size class
  we'd target eventually.
- **GSM8K**: most-validated RLVR baseline; integer answer = trivial
  binary verifier; lots of public reward curves to compare against if
  ours looks anomalous.
- **GRPO group size 8**: TRL default and the standard in published GSM8K
  recipes. Don't drop below 4 — the advantage estimator gets noisy.
- **`beta=0.04`**: TRL default KL penalty against the reference model.
  If the model collapses to repetitive output, raise this; if it never
  improves, lower it.
- **`learning_rate=5e-6`**: standard for full-FT 1B-class on RL. Cut by
  10× when adding LoRA in Phase 2 (LoRA effective LR is ~10× higher than
  the nominal value).

## Promoting to Phase 2 (LoRA)

When Phase 1 passes, add LoRA wrapping at the marked spot in
`scripts/train_canary_gsm8k.py` (`# Phase 1 = full FT. No LoRA wrapping.`)
and lower the LR. The expected outcome per "LoRA Without Regret" is that
LoRA matches full FT within a few percentage points; if it doesn't, the
issue is almost always one of: rank too low, target modules incomplete,
or LR too high.

## Promoting to Phase 3 (TLA+)

Replace `load_gsm8k_prompts` with a loader over `data/processed/diamond_curated.jsonl`
and replace `binary_correctness_reward` with a wrapper around
`src.validators.per_action_tlc.validate_action`. Keep the same trainer,
LR, and KL — only the dataset and reward should change. If the curve
doesn't lift, the problem is the TLA+ task itself (reward sparsity,
data scarcity, or per-action harness coverage), not the RL stack.
