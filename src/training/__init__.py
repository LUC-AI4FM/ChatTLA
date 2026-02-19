"""
training — Phase 2 fine-tuning package.

Scripts:
  dataset_builder.py   — Build train/eval JSONL from validated corpus
  train.py             — Fine-tune gpt-oss-20b on GPU 1 with LoRA
  tlc_eval_callback.py — TLC-based validation metric at each eval step
  merge_lora.py        — Merge adapter into base weights post-training
  lora_config.yaml     — LoRA hyperparameters (documented)
"""
