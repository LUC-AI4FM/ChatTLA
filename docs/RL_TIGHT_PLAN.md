# RL Tight Loop — Execution Plan

**Date**: 2026-04-22  
**Goal**: Produce paper-ready results in 2 weeks via continuous RL fine-tuning  
**Hardware**: 2x Quadro RTX 8000 (96GB total VRAM)

---

## Why the previous RL loop failed

Analysis of `outputs/logs/rl_history.jsonl` (73 cycles):

| Metric | Value | Problem |
|--------|-------|---------|
| Cycles run | 73 | Lots of generation |
| Retrains triggered | **0** | Never learned! |
| Gold specs/cycle | 1-3 | Low yield |
| DPO pairs accumulated | 64 | Never used |
| `retrain_skipped_min_data` | Most cycles | Threshold too high (25) |

**Root cause**: The threshold was 25 new SFT examples, but cycles only produced 1-3 gold specs. The model generated data but never updated weights.

---

## The new approach: `rl_tight_loop.py`

### Key changes

| Parameter | Old | New | Why |
|-----------|-----|-----|-----|
| DPO train threshold | 25 | **5** | Train early, train often |
| Cycle duration target | 4-6h | **1-2h** | Faster iteration |
| Evaluation | Noisy quick-eval | Fixed holdout | Reproducible metrics |
| Seeds | Random | **Explicit** | Reproducibility |
| State persistence | Partial | **Full JSON** | Resume safely |

### Reward signal

- **Gold** (TLC pass): reward = 1.0
- **Silver** (SANY pass only): reward = 0.3  
- **Bronze** (SANY fail): reward = 0.0

This is fully verifiable — no LLM judge, no subjective scoring.

---

## Execution timeline (2 weeks)

### Phase 1: Validate the method (Days 1-3)

**Goal**: Prove RL loop works before committing GPU-days.

```bash
# 1. Quick smoke test (5 min)
./scripts/launch_rl_tight.sh smoke

# 2. Small model experiment (few hours)
# Tests if DPO improves TLC rate on 3B model
python scripts/rl_small_model.py --cycles 20

# 3. If small model shows improvement → proceed to 20B
```

**Success criteria**: 
- Smoke test completes without errors
- Small model shows any TLC improvement after training

### Phase 2: Run the 20B loop (Days 3-10)

**Goal**: Accumulate data and train checkpoints.

```bash
# Start the loop (runs continuously in tmux)
./scripts/launch_rl_tight.sh start

# Monitor progress
./scripts/launch_rl_tight.sh status
./scripts/launch_rl_tight.sh logs

# Check metrics
./scripts/launch_rl_tight.sh metrics
```

**Expected per cycle** (~1.5h each):
- 15 prompts × 3 attempts = 45 specs generated
- ~3-5 gold specs (TLC pass)
- ~15-20 silver specs (SANY pass)
- ~1-2 new DPO pairs (gold vs worse)
- Training triggers every ~3-5 cycles

**7 days × 16 cycles/day = ~112 cycles possible**

### Phase 3: Evaluate and write (Days 10-14)

**Goal**: Produce paper figures and tables.

```bash
# Run final evaluation on best checkpoint
python scripts/rl_tight_loop.py --eval-only

# Generate comparison table
python -c "
import json
with open('outputs/logs/rl_tight/history.jsonl') as f:
    data = [json.loads(l) for l in f]

# Before training (cycle 1)
before = data[0]

# After training (best cycle)
best = max(data, key=lambda x: x['holdout_tlc_rate'])

print('| Metric | Before | After | Delta |')
print('|--------|--------|-------|-------|')
print(f'| SANY rate | {before[\"sany_rate\"]:.1%} | {best[\"sany_rate\"]:.1%} | {best[\"sany_rate\"]-before[\"sany_rate\"]:+.1%} |')
print(f'| TLC rate | {before[\"tlc_rate\"]:.1%} | {best[\"tlc_rate\"]:.1%} | {best[\"tlc_rate\"]-before[\"tlc_rate\"]:+.1%} |')
print(f'| Holdout TLC | {before[\"holdout_tlc_rate\"]:.1%} | {best[\"holdout_tlc_rate\"]:.1%} | {best[\"holdout_tlc_rate\"]-before[\"holdout_tlc_rate\"]:+.1%} |')
"
```

---

## Reproducibility checklist

For the paper to be credible, document:

- [ ] Git commit hash (logged in state.json)
- [ ] Config fingerprint (logged in state.json)
- [ ] Random seeds (explicit, starts at 42)
- [ ] Hardware (2x RTX 8000, CUDA version)
- [ ] Python environment (`pip freeze > requirements_frozen.txt`)
- [ ] Training hyperparameters (saved in config.json)
- [ ] All metrics over time (history.jsonl)

---

## Alternative paths if 20B doesn't improve

### Path A: Data scaling
If variance is the problem, more data helps:
```bash
# Use existing 64 DPO pairs from old loop
cat data/processed/rl/dpo_pairs.jsonl >> data/processed/rl_tight/dpo_pairs.jsonl

# Then restart training with more data
python scripts/rl_tight_loop.py --resume
```

### Path B: Smaller model overfit
Prove the method works on 3B, publish that:
```bash
python scripts/rl_small_model.py --cycles 50
# If 3B improves, that's a valid result
```

### Path C: Seed variance study
Run same config with different seeds, measure variance:
```bash
for seed in 42 123 456 789 1000; do
    python scripts/rl_tight_loop.py --cycles 10 --seed $seed --output-dir outputs/seed_$seed
done
# Compare variance across seeds
```

---

## Files created by this plan

| File | Purpose |
|------|---------|
| `scripts/rl_tight_loop.py` | Main RL loop (reproducible, fast) |
| `scripts/launch_rl_tight.sh` | tmux launcher and monitoring |
| `scripts/rl_small_model.py` | Small model experiment |
| `data/processed/rl_tight/` | DPO pairs, gold specs, state |
| `outputs/logs/rl_tight/` | Metrics history, logs |
| `outputs/checkpoints_rl_tight/` | Training checkpoints |

---

## Monitoring commands

```bash
# Attach to running loop
./scripts/launch_rl_tight.sh attach

# View metrics summary
./scripts/launch_rl_tight.sh metrics

# Tail logs
./scripts/launch_rl_tight.sh logs

# Check GPU usage
nvidia-smi -l 5

# Disk usage
df -h /home/espencer2
```

---

## Paper contribution framing

**If gains are significant**: 
> "We show that DPO with verifiable rewards (TLC model checking) improves TLA+ generation quality from X% to Y% TLC pass rate."

**If gains are marginal/noisy**:
> "We present a rigorous evaluation methodology for TLA+ generation and characterize the variance in fine-tuning outcomes, showing that apparent improvements of <Z% are within noise."

**If negative result**:
> "We demonstrate that current RLHF methods struggle with formal verification tasks despite using ground-truth rewards, suggesting the need for alternative approaches."

All three are publishable if the methodology is sound.

---

## Quick start

```bash
cd /home/espencer2/ChatTLA

# 1. Smoke test
./scripts/launch_rl_tight.sh smoke

# 2. Start the loop
./scripts/launch_rl_tight.sh start

# 3. Monitor
./scripts/launch_rl_tight.sh status
```
