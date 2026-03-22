"""
training — Phase 2 fine-tuning package.

Scripts:
  dataset_builder.py   — Build train/eval JSONL from validated corpus
  train.py             — Fine-tune gpt-oss-20b on GPU 1 with LoRA
  tlc_eval_callback.py — TLC-based validation metric at each eval step
  merge_lora.py        — Merge adapter into base weights post-training
  publish_hf.py        — Versioned GGUF + Modelfile + README → Hugging Face Hub
  train_dpo.py         — Optional DPO refinement after SFT (gold dpo_pairs)
  lora_config.yaml     — LoRA hyperparameters (documented)
"""
