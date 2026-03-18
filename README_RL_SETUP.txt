================================================================================
  ChatTLA Autonomous RL Loop — SETUP COMPLETE ✓
================================================================================

The autonomous, continuously-looping RL pipeline is now RUNNING.

CURRENT STATUS:
  Session:     chattla-rl (tmux)
  GPU Usage:   GPU 1 (2.5% util, 2.5GB) — looping spec generation
  Cycle:       1 completed, 7 gold + 10 silver specs
  Sleep:       ~76 min until cycle 2
  
QUICK COMMANDS:
  Check status:    ./scripts/launch_rl.sh status
  View logs:       ./scripts/launch_rl.sh logs
  Stop gracefully: ./scripts/launch_rl.sh stop
  Restart:         ./scripts/launch_rl.sh restart

WHAT IT DOES:
  • Every 1.5 hours:
    1. Generate TLA+ specs (25/day, 40/night)
    2. Validate with SANY (syntax) + TLC (semantics)
    3. Collect gold/silver specs as training data
    4. When 10+ high-quality specs → retrain + redeploy
    5. Every 3 cycles: benchmark on 20-problem suite
    6. Adapt difficulty based on performance
    7. Sleep & repeat

KEY FILES:
  Loop script:        scripts/rl_loop.py
  Launcher:           scripts/launch_rl.sh
  Operations guide:   RL_OPERATIONS_GUIDE.md
  Logs:               outputs/logs/rl_loop.log
  History:            outputs/logs/rl_history.jsonl
  Training data:      data/processed/augmented.jsonl
  Generated models:   outputs/merged_model/ (after retrain)

EXPECTED IMPROVEMENT:
  Baseline (v9):      SANY 75%, TLC 25%
  After ~5 cycles:    SANY 80%, TLC 30-35%
  After ~15 cycles:   SANY 85%+, TLC 40-50%

SCHEDULE:
  Daytime (06:00-22:00):   Conservative (25 specs/cycle, GPU 1 only)
  Nighttime (22:00-06:00):  Full speed (40 specs/cycle, both GPUs)

GPU RESOURCE SHARING:
  · Leaves ~20-25% VRAM free during daytime
  · Uses ~10% of GPU 0 during daytime (other work can use it)
  · Full dual-GPU retrain power at night

BACKGROUND: This is a continuously-improving RL loop that treats TLC verification 
as the reward signal. The model learns from its own verified spec generations,
building an iterative improvement flywheel. After every 10+ successful specs, 
the model retrains on its own data, redeploys, and continues generating.

================================================================================
