# ChatTLA Autonomous RL Loop — Operational Guide

**Status**: ✅ RUNNING (Started 2026-03-18 13:02:07)

## Quick Summary

You now have a **fully autonomous, continuously-looping RL pipeline** that:

1. **Generates** TLA+ specs from prompts (25/hr during daytime, 40/hr at night)
2. **Validates** through SANY (syntax) + TLC (semantics) with **granular feedback**
3. **Trains** the model on gold/silver specs to learn from its own successes
4. **Retrains** every 10+ new high-quality examples
5. **Deploys** updated models via GGUF → Ollama
6. **Evaluates** every 3 cycles on the 20-problem benchmark suite
7. **Schedules intelligently**: Full speed 22:00–06:00, throttled 06:00–22:00

## Current System Health

```
Session:  chattla-rl (tmux)
GPU 0:    3212 MB (11% util) — available for other work
GPU 1:    2497 MB (1% util)  — RL loop running in reserve capacity

Cycle 1 Results:
  25 prompts → 17 SANY pass (68%) → 7 gold, 10 silver
  +17 new SFT examples (now 101 augmented total)
  Next cycle: in 76.6 min

Model: chattla:20b (fine-tuned gpt-oss-20b)
Benchmark baseline (v9_best5_sc2): SANY 15/20, TLC 5/20
```

## Usage Commands

### **Bootstrap (venv + deps + `.env`)**

On **`start`**, the script bootstraps `.venv` + `requirements.txt` + `.env` before tmux. **`restart`** runs the same full bootstrap first (E2E), then stops the old session and starts a new one. To only prepare the environment without tmux:

```bash
./scripts/launch_rl.sh setup   # .venv, pip install -r requirements.txt, load .env in this shell
```

The tmux session prepends `.venv/bin` to `PATH` and **sources `.env`** before starting `scripts/rl_loop.py` (so `HF_TOKEN` works for Hub publish).

### **Status & Monitoring**

```bash
# Show current cycle stats, GPU usage, recent logs
./scripts/launch_rl.sh status

# Tail the real-time log (Ctrl-C to exit)
./scripts/launch_rl.sh logs

# Attach to the tmux session directly
tmux attach -t chattla-rl

# Detach from tmux (Ctrl-B, D)
```

### **Control**

```bash
# Graceful shutdown (completes current phase, ~2–5 min)
./scripts/launch_rl.sh stop

# Restart the loop (stop + start)
./scripts/launch_rl.sh restart

# Check if running
tmux has-session -t chattla-rl && echo "Running" || echo "Stopped"
```

## How It Works

### **Cycle Structure** (target 1.5 hours)

Each cycle runs 4 phases in sequence:

```
Phase 1: Generate + Validate (~15–30 min)
  • Load 25 prompts (daytime) or 40 (nighttime)
  • Difficulty adapts: easy specs → hard as model improves
  • Generate with multi-attempt self-correction (up to 2 retries)
  • Validate: SANY (syntax) → TLC (semantics)
  • Extract: gold (perfect), silver (SANY ok, TLC issues), bronze (syntax fail)

Phase 2: Build Training Data (~5 min)
  • Gold/silver specs → SFT examples (positive training signal)
  • Gold + bronze pairs → DPO examples (preference learning)
  • TLC violations → error-conditioned examples (learn from mistakes)
  • Persist to augmented.jsonl and dpo_pairs.jsonl

Phase 3: Retrain + Deploy (60+ min, only if threshold met)
  • Trigger: 10+ new gold/silver examples accumulated
  • Rebuild train/eval JSONL from augmented data
  • Fine-tune on both GPUs night, GPU 1 only daytime
  • ~5–10 epochs depending on data size
  • Merge LoRA weights → GGUF conversion → Ollama register
  • New model live for next cycle

Phase 4: Benchmark Eval (~30 min, every 3 cycles)
  • Run 20-problem suite with self-correction
  • Track SANY pass rate, TLC pass rate
  • Results → outputs/benchmark_results_rl_c{N}_{timestamp}.csv

Then sleep to fill 1.5 hours, respecting daytime/nighttime schedule.
```

### **GPU Scheduling**

**Daytime (06:00–22:00)**: Conservative
- Max 25 prompts/cycle (vs 40 at night)
- Single GPU (device 1) for training
- Leave ~25% VRAM free for other users

**Nighttime (22:00–06:00)**: Full throttle
- Max 40 prompts/cycle
- Both GPUs (0,1) for training
- Use up to 90% VRAM

**Adaptive Difficulty**
- Recent SANY < 40% → focus on easy (difficulty ≤2)
- Recent SANY 40–60% → moderate (difficulty ≤3)
- Recent SANY 60–80% → harder (difficulty ≤4)
- Recent SANY > 80% → very hard (difficulty ≤5)

## Key Data Files

**Outputs** (`outputs/logs/`):
```
rl_loop.log              — Real-time event log
rl_history.jsonl         — Cycle-by-cycle stats (JSON):
                           {cycle_id, timestamp, gold_count, silver_count,
                            retrains, benchmark_sany_rate, benchmark_tlc_rate, ...}
benchmark_results_rl_c{N}_{ts}.csv  — Benchmark results (every 3 cycles)
```

**Training Data** (`data/processed/`):
```
augmented.jsonl          — All generated SFT examples (grows each cycle)
rl/dpo_pairs.jsonl       — Preference pairs: (prompt, chosen_gold, rejected_bronze)
train.jsonl              — Training set (rebuilt before each retrain)
eval.jsonl               — Eval set (rebuilt before each retrain)
```

## Understanding Cycle Statistics

From `./scripts/launch_rl.sh status`:

```
Total cycles: 1
  Gold specs:    7        ← 100% correct specs (TLA+ + TLC verified)
  Silver specs:  10       ← Syntax OK but semantic issues
  Retrains:      0        ← Full train→merge→GGUF cycles completed
  
Last cycle:    #1 at 2026-03-18T13:02:07
  SANY pass:   17/25      ← 68% syntax correctness
  TLC pass:    7/25       ← 28% full correctness ("gold rate")

Augmented examples: 101   ← Specs ready to train on
DPO pairs:          0     ← Will grow as model gets better
Training examples:  146   ← Last merged dataset
```

**Goal metrics** (from GitHub TLA+ examples acceptance):
- SANY pass rate: **target >80%** (syntax quality)
- TLC pass rate: **target >50%** (semantic correctness)

Current (Cycle 1): 68% SANY, 28% TLC — improving with each cycle as model learns.

## Monitoring in Real-Time

```bash
# Watch logs with highlighting
tail -f outputs/logs/rl_loop.log | grep -E "(tier=gold|SANY|TLC|Retrain|Benchmark)"

# Watch JSON history updates
watch -n 10 'tail -5 outputs/logs/rl_history.jsonl | jq .'

# GPU memory & utilization
watch -n 5 nvidia-smi

# Count augmented examples growing
watch -n 30 'wc -l data/processed/augmented.jsonl'
```

## Troubleshooting

### Loop stops or crashes
```bash
# Check last error in log
tail -50 outputs/logs/rl_loop.log | grep -i error

# Restart gracefully
./scripts/launch_rl.sh restart

# If completely stuck, kill and restart
tmux kill-session -t chattla-rl
./scripts/launch_rl.sh start
```

### Retrain never happens
- Threshold is 10 new examples; need to wait ~2–3 cycles
- If stuck, manually trigger:
```bash
tmux send-keys -t chattla-rl "python3 -m src.training.dataset_builder --include-augmented && python3 -m src.training.train --epochs 3" Enter
```

### GPU memory issues
- Loop auto-caps GPU usage at 75% (day) or 90% (night)
- If OOM during retrain, kill cycle and reduce prompts:
```bash
./scripts/launch_rl.sh stop
# Edit scripts/rl_loop.py: MAX_PROMPTS_DAY = 15, MAX_PROMPTS_NIGHT = 25
./scripts/launch_rl.sh start
```

### Ollama not responding
```bash
# Check Ollama is still running
ps aux | grep ollama

# Restart Ollama
pkill ollama
ollama serve &
# Wait 10s, then restart loop
./scripts/launch_rl.sh restart
```

## Hugging Face Hub (after retrain)

If `HF_TOKEN` is set (e.g. in `.env` loaded by your shell or `launch_rl.sh`), each successful retrain **merge + GGUF + Ollama** is followed by `python -m src.training.publish_hf`: versioned `gguf/chattla-20b-vN-Q8_0.gguf`, `gguf/Modelfile`, and an updated `README.md`. Next version is tracked in `data/benchmarks/hf_publish_state.json`. Use `--no-publish-hf` on the RL script to skip uploads. See `docs/TRAINING_PIPELINE_AUDIT.md`.

## Configuration (scripts/rl_loop.py)

Key parameters at the top of the file:

```python
CYCLE_HOURS        = 1.5       # Target hours per cycle
RETRAIN_THRESHOLD  = 10        # New specs before retrain
NIGHTTIME_START    = 22        # 10 PM start
NIGHTTIME_END      = 6         # 6 AM end
GPU_VRAM_CAP_DAY   = 0.75      # 75% daytime
GPU_VRAM_CAP_NIGHT = 0.90      # 90% nighttime
MAX_PROMPTS_DAY    = 25        # Specs per cycle (day)
MAX_PROMPTS_NIGHT  = 40        # Specs per cycle (night)
BENCHMARK_EVERY_N  = 3         # Run benchmark every N cycles
```

Edit to adjust behavior, then restart:
```bash
./scripts/launch_rl.sh restart
```

## Expected Improvement Trajectory

**Cycle 1 (now)**: ~28% TLC pass (baseline)
**Cycle 2–3**: +2–5% TLC as first retrain kicks in
**Cycle 5–10**: +10–20% TLC as model learns from diverse specs
**Cycle 15+**: +30–50% TLC as RL feedback compounds

**Acceleration factors**:
- Adaptive difficulty keeps model learning challenging problems
- TLC granular feedback teaches specific error patterns
- Error-conditioned examples (bug_fix) reinforce fixes
- Nightly full-speed generation provides more training signal

## Next Steps

1. **Monitor first 2–3 cycles** to verify stability
   ```bash
   ./scripts/launch_rl.sh logs
   ```

2. **After first retrain (~cycle 2)**, benchmark and compare
   ```bash
   cat outputs/benchmark_results_rl_c3_*.csv
   ```

3. **Tune schedule parameters** if needed (e.g., more prompts, lower retrain threshold)

4. **Let it run uninterrupted** — the system is designed for multi-day operation

## Architecture Highlights

- **Granular TLC feedback** instead of pass/fail: extract violation types, state traces, operator errors → teach model specific fixes
- **DPO-ready data**: (gold_spec, bronze_spec) pairs accumulate for future direct preference optimization
- **Adaptive difficulty**: starts easy, ramps up as model improves (no wasted cycles on easy specs once high SANY rate)
- **Distributed GPU scheduling**: nighttime uses both GPUs, daytime reserves GPU 0 for other users
- **Graceful shutdown**: SIGTERM completes current phase before exiting
- **Resume state**: history.jsonl persists cycle count; restart picks up where it left off

---

**Questions?** Check `./scripts/rl_loop.py` for detailed comments or review the main pipeline classes at the top of the file.

**Logs location**: `outputs/logs/rl_loop.log` and `outputs/logs/rl_history.jsonl`
