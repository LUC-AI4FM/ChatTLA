"""
train.py — Fine-tune gpt-oss-20b for TLA+ specification generation.

Hardware
--------
GPUs: 2x Quadro RTX 8000 (48 GB VRAM each, 96 GB total)
Both GPUs used via CUDA_VISIBLE_DEVICES=0,1 with device_map="auto".

Model loading
-------------
gpt-oss-20b uses native MXFP4-quantized MoE weights (~3.6B active params).
No additional quantization needed — native MXFP4 fits comfortably in 96GB.

LoRA strategy
-------------
target_modules="all-linear" covers attention, FFN, AND MoE expert projections.
  - r=16, alpha=32 (effective scale=2.0)
  - All transformer layers trained (no layers_to_transform restriction)
  - Dropout 0.05 for regularization on small dataset

Training setup
--------------
SFTTrainer from TRL handles the harmony-formatted JSONL natively via
the messages field.  We use:
  - per_device_train_batch_size=2, gradient_accumulation_steps=4 → effective batch = 8
  - gradient_checkpointing=True to trade compute for memory
  - max_length=2048 (TLA+ specs average 6984 chars — must NOT truncate)
  - 10 epochs for 57-example dataset with cosine LR schedule
  - BF16 mixed precision
  - load_best_model_at_end=True with eval_loss metric

MLflow experiment tracking
--------------------------
Every run logs: config, per-step loss, eval loss, sany_parse_rate,
tlc_clean_rate (from TLCEvalCallback), and hardware stats.

Usage
-----
    # Smoke test (validates setup):
    CUDA_VISIBLE_DEVICES=0,1 python -m src.training.train --smoke-test

    # Full training run:
    CUDA_VISIBLE_DEVICES=0,1 python -m src.training.train

    # Resume from checkpoint:
    CUDA_VISIBLE_DEVICES=0,1 python -m src.training.train --resume outputs/checkpoints/checkpoint-500
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Use both GPUs by default. Override with CUDA_VISIBLE_DEVICES env var if needed.
# Previous single-GPU pinning was unnecessarily limiting VRAM to 48GB.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

# Optimize CUDA memory allocation to avoid fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
import mlflow
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig

from src.training.tlc_eval_callback import TLCEvalCallback


class ClearCacheCallback(TrainerCallback):
    """Free fragmented GPU memory before eval/save to prevent OOM spikes."""

    def on_evaluate(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_save(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_REPO_ROOT    = Path(__file__).resolve().parents[2]
# diamond_sft_v3.jsonl is the post-2026-04-07 SFT corpus: existing train.jsonl
# (713 records) + the 170 new diamond-gen-v2 specs in chat-message format,
# with the 170 new specs oversampled 2x to bias the gradient toward the
# newly-introduced algorithm families. Built by scripts/build_diamond_sft_v3.py.
# Holdout (data/processed/diamond_eval_holdout.jsonl) is excluded by module
# name with a hard assertion at build time.
_TRAIN_JSONL  = _REPO_ROOT / "data" / "processed" / "diamond_sft_v4.jsonl"
_EVAL_JSONL   = _REPO_ROOT / "data" / "processed" / "eval.jsonl"
_PROVER_TRAIN_JSONL = _REPO_ROOT / "data" / "processed" / "prover_train.jsonl"
_PROVER_EVAL_JSONL  = _REPO_ROOT / "data" / "processed" / "prover_eval.jsonl"
_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "checkpoints"
_PROVER_CHECKPOINT_DIR = _REPO_ROOT / "outputs" / "checkpoints_prover"
_LORA_CFG_PATH  = Path(__file__).parent / "lora_config.yaml"

MODEL_ID = "openai/gpt-oss-20b"


def load_lora_config() -> LoraConfig:
    """Load LoRA config from lora_config.yaml."""
    with _LORA_CFG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias=cfg["bias"],
        target_modules=cfg["target_modules"],
        target_parameters=cfg.get("target_parameters"),
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(device_map: str = "auto", max_gpu_memory_mb: int | None = None,
                              base_model: str | None = None):
    """
    Load model with optional base_model override for incremental training.
    - `base_model`: path or HF ID to load instead of the default MODEL_ID.
      Use this to load from a previously merged model (e.g. outputs/merged_model/)
      so DPO builds on existing TLA+ knowledge instead of starting from scratch.
    - `device_map`: passed through to `from_pretrained` ("auto" | "cpu" | "cuda").
    - `max_gpu_memory_mb`: when provided and `device_map=="auto"`, it's used to
      cap the memory assigned to `cuda:0` by the accelerate dispatcher.

    gradient_checkpointing requires use_cache=False.
    """
    model_id = base_model or MODEL_ID

    # FSDP / distributed launch: WORLD_SIZE>1 means accelerate launched us with
    # one process per GPU. With `fsdp_cpu_ram_efficient_loading: true`, rank 0
    # loads the full model on CPU; other ranks materialize on meta. FSDP then
    # broadcasts and shards across GPUs. This avoids each rank loading the
    # full ~42 GB BF16 base on its single GPU (which OOMs on RTX 8000).
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    fsdp_mode = world_size > 1 and isinstance(device_map, str) and device_map == "auto"
    if fsdp_mode:
        device_map = "cpu"  # all ranks load on CPU; FSDP gathers to GPUs after wrap
        print(f"[train] FSDP detected (WORLD_SIZE={world_size}); rank {local_rank} "
              f"loading on CPU (FSDP will shard to GPUs)")

    print(f"[train] Loading model: {model_id} (device_map={device_map})")
    print(f"[train] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[train] VRAM available: {vram_gb:.1f} GB")
    except Exception:
        vram_gb = None
        print("[train] VRAM available: unknown")

    # gpt-oss-20b comes pre-quantized with Mxfp4Config (weight-only quantization).
    # The model's native quantization is fully compatible with LoRA fine-tuning.
    # We do not apply additional quantization.

    # Build max_memory mapping for accelerate / transformers dispatch when the
    # user requests automatic device placement.
    max_memory = None
    if isinstance(device_map, str) and device_map == "auto":
        try:
            n_gpus = torch.cuda.device_count()
            print(f"[train] Detected {n_gpus} GPU(s)")

            # total system RAM for a sensible cpu cap (fallback to 64GB)
            try:
                total_ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                total_ram_mb = int(total_ram_bytes // (1024 * 1024))
            except Exception:
                total_ram_mb = 64 * 1024

            # Build max_memory for ALL visible GPUs.  Previous code only set
            # GPU 0, causing the dispatcher to skip GPU 1 and offload to CPU
            # (meta device) — which then crashes DataParallel.
            max_memory = {"cpu": f"{total_ram_mb}MB"}
            for gpu_id in range(n_gpus):
                total_vram_mb = int(torch.cuda.get_device_properties(gpu_id).total_memory // (1024 * 1024))

                # Try to read *currently free* VRAM
                free_vram_mb = None
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(gpu_id)
                    free_vram_mb = int(free_bytes // (1024 * 1024))
                except Exception:
                    pass

                # User-provided cap or 90% of total (was 60% — too conservative)
                user_cap_mb = max_gpu_memory_mb or int(total_vram_mb * 0.90)

                # If we know free VRAM, cap to 90% of free to avoid overcommit
                if free_vram_mb is not None:
                    cap_mb = int(min(user_cap_mb, int(free_vram_mb * 0.90)))
                else:
                    cap_mb = user_cap_mb

                # Floor: never go below 4GB
                cap_mb = max(4096, cap_mb)
                max_memory[gpu_id] = f"{cap_mb}MB"
                print(f"[train] GPU {gpu_id}: total={total_vram_mb}MB free={free_vram_mb}MB -> cap={cap_mb}MB")

            print(f"[train] max_memory for dispatch: {max_memory}")
        except Exception:
            max_memory = None

    try:
        from_pretrained_kwargs = dict(
            attn_implementation="eager",
            use_cache=False,           # required for gradient checkpointing
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,    # stream weights to devices to reduce peak memory
            trust_remote_code=True,    # gpt-oss requires special modeling code
        )
        if fsdp_mode:
            # Force BF16 on CPU load so we don't expand to FP32 (~80 GB).
            from_pretrained_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, **from_pretrained_kwargs)
    except Exception as exc:
        print("[train] ERROR loading model:", exc)
        raise

    # For merged models, tokenizer may be in the same dir; fall back to MODEL_ID
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_training_args(
    smoke_test: bool = False,
    resume_from: str | None = None,
    use_cpu: bool = False,
    num_epochs: int | None = None,
    max_length: int = 4096,
    per_device_batch_size: int | None = None,
    gradient_accumulation_steps_override: int | None = None,
    max_steps: int | None = None,
    learning_rate: float | None = None,
    output_dir: Path | None = None,
) -> SFTConfig:
    """Return an SFTConfig tuned for smoke or full runs.

    - `use_cpu=True` will disable `bf16` and set TrainingArguments.use_cpu so
      the Transformers/Accelerate runtime knows we're executing on the CPU.
    - `per_device_batch_size` and `gradient_accumulation_steps_override` can override
      defaults for tight-memory situations.
    """
    out_dir = output_dir or _CHECKPOINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # When running on CPU we must not request bf16 (invalid on CPU).
    bf16_enabled = False if use_cpu else True

    # Adaptive batch size: allow overrides for tight-memory situations
    batch_size = per_device_batch_size if per_device_batch_size is not None else (1 if smoke_test else 1)
    accum_steps = gradient_accumulation_steps_override if gradient_accumulation_steps_override is not None else (2 if smoke_test else 8)

    return SFTConfig(
        output_dir=str(out_dir),
        # --- Batch / accumulation ------------------------------------------
        # With longer sequences (4096), use batch=1 with higher accum to stay
        # within VRAM. Effective batch = 1*8 = 8 (good for 57 training examples)
        # Can be overridden for tight-memory conditions.
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        # --- Optimizer / schedule ------------------------------------------
        learning_rate=learning_rate or 1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5 if smoke_test else 5,
        # --- Precision & memory --------------------------------------------
        bf16=bf16_enabled,
        gradient_checkpointing=True,
        # H2 fix (paper appendix A 2026-04-07): we keep reentrant gradient
        # checkpointing (lower activation memory) and instead set
        # lora_dropout=0.0 in lora_config.yaml. With no dropout, there is no
        # RNG-dependent mask whose recompute could differ from the forward
        # pass and cascade into MoE-routing shape mismatches on cuda:1.
        # Tried use_reentrant=False first; it OOMed at step 6 on 2x RTX 8000.
        # --- Sequence length (new in TRL 0.28) ----------------------------
        # TLA+ specs are long (avg 1924 tokens, p90=3236). Lower max_length saves VRAM.
        max_length=512 if smoke_test else max_length,
        # --- Logging & checkpointing ----------------------------------------
        # Scale epochs based on dataset size to avoid overfitting.
        # When called from self_improve.py, num_epochs is set dynamically.
        # Default reduced from 10 to 5: with small datasets (200-300 examples),
        # 10+ epochs causes severe memorization and SANY regression.
        num_train_epochs=1 if smoke_test else (num_epochs or 5),
        max_steps=5 if smoke_test else (max_steps if max_steps is not None else -1),
        # Eval is DISABLED for full runs — the model is split across 2 GPUs
        # via pipeline parallelism and eval mode (no gradient checkpointing)
        # keeps all layer activations in VRAM, causing OOM on the 200K-vocab
        # cross_entropy (10.5 GiB allocation).  Training forward pass is fine
        # because gradient_checkpointing drastically reduces activation memory.
        # We evaluate the final checkpoint manually after training instead.
        eval_strategy="steps" if smoke_test else "no",
        eval_steps=5 if smoke_test else None,
        save_strategy="steps",
        save_steps=5 if smoke_test else 20,
        save_total_limit=5,
        logging_steps=1 if (smoke_test or max_steps) else 5,
        # --- MLflow --------------------------------------------------------
        report_to="mlflow",
        run_name="chattla-gpt-oss-20b",
        # --- CPU flag (Transformers) --------------------------------------
        use_cpu=use_cpu,
        # --- Resume --------------------------------------------------------
        resume_from_checkpoint=resume_from,
    )


def main(
    smoke_test: bool = False,
    resume_from: str | None = None,
    device_map: str = "auto",
    max_gpu_memory_mb: int | None = None,
    lora_layers_override: list[int] | None = None,
    lora_top_k: int | None = None,
    num_epochs: int | None = None,
    max_length: int = 4096,
    run_dpo_after: bool = False,
    per_device_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    max_steps: int | None = None,
    learning_rate: float | None = None,
    base_model: str | None = None,
    prover: bool = False,
) -> None:
    if prover:
        mlflow.set_experiment("ChatTLA-Prover-gpt-oss-20b")
    else:
        mlflow.set_experiment("ChatTLA-gpt-oss-20b")

    # --- Data ---------------------------------------------------------------
    train_path = _PROVER_TRAIN_JSONL if prover else _TRAIN_JSONL
    eval_path = _PROVER_EVAL_JSONL if prover else _EVAL_JSONL
    output_dir = _PROVER_CHECKPOINT_DIR if prover else _CHECKPOINT_DIR
    if not train_path.exists():
        print(f"[train] ERROR: {train_path} not found.")
        sys.exit(1)
    if not eval_path.exists():
        print(f"[train] ERROR: {eval_path} not found.")
        sys.exit(1)
    print(f"[train] mode={'PROVER' if prover else 'spec-gen'}  train={train_path.name}  eval={eval_path.name}")

    # Load via from_list to avoid PyArrow arrow.json extension type error
    # (ArrowNotImplementedError: MakeBuilder for type extension<arrow.json>)
    def _load_jsonl(path: Path) -> Dataset:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return Dataset.from_list(records)

    try:
        train_dataset = _load_jsonl(train_path)
        eval_dataset = _load_jsonl(eval_path)
    except Exception as e:
        print(f"[train] from_list load failed ({e}), falling back to load_dataset...")
        train_dataset = load_dataset("json", data_files=str(train_path), split="train")
        eval_dataset = load_dataset("json", data_files=str(eval_path), split="train")

    if smoke_test:
        train_dataset = train_dataset.select(range(min(10, len(train_dataset))))
        eval_dataset  = eval_dataset.select(range(min(5,  len(eval_dataset))))
        print(f"[train] Smoke test mode: {len(train_dataset)} train, {len(eval_dataset)} eval examples.")

    # --- Model + LoRA -------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(
        device_map=device_map, max_gpu_memory_mb=max_gpu_memory_mb, base_model=base_model,
    )

    # Determine which transformer layers are resident on the GPU and restrict
    # LoRA to those layers. This avoids creating a split adapter state where
    # some adapters live on `cuda` while others are meta/cpu (which caused the
    # MmBackward0 device-mismatch error).
    gpu_layer_indices = []
    try:
        # most HF causal LM wrappers expose the transformer layers as
        # `model.model.layers` — fall back gracefully if structure differs
        layers_list = getattr(model, "model", None)
        if layers_list is not None and hasattr(model.model, "layers"):
            for i, layer in enumerate(model.model.layers):
                # if any parameter in this layer is on a CUDA device, mark it
                if any(getattr(p, 'device', None) is not None and getattr(p, 'device').type == 'cuda' for p in layer.parameters()):
                    gpu_layer_indices.append(i)
    except Exception:
        gpu_layer_indices = []

    lora_config = load_lora_config()

    # Allow explicit overrides from the CLI.  Precedence (highest→lowest):
    # 1) --lora-layers  2) --lora-top-k  3) GPU-resident layers detected earlier
    if lora_layers_override:
        lora_config.layers_to_transform = lora_layers_override
        print(f"[train] Overriding LoRA layers → {lora_layers_override}")
    elif lora_top_k is not None:
        lora_config.layers_to_transform = list(range(lora_top_k))
        print(f"[train] Applying LoRA to first {lora_top_k} layers → {lora_config.layers_to_transform}")
    elif gpu_layer_indices:
        # restrict LoRA to the GPU-resident layers to keep all trainable
        # params on the same device
        lora_config.layers_to_transform = gpu_layer_indices
        print(f"[train] Restricting LoRA to GPU-resident layers → {gpu_layer_indices}")
    else:
        # With 2x 48GB GPUs we can afford to train all layers.
        # Do NOT restrict — let PEFT apply LoRA to every layer.
        print(f"[train] Applying LoRA to ALL layers (no layers_to_transform restriction)")

    # DEBUG: show the LoRA/PEFT config being passed to PEFT so we can verify
    # that `target_parameters` is not present and `target_modules` is correct.
    print("[train] LoRA/PEFT config:", lora_config)

    # Preserve the base model's hf_device_map before PEFT wrapping — PeftModel
    # doesn't always propagate it, which causes the Trainer to think the model
    # is NOT distributed and wrap it in DataParallel (crash: meta device params).
    base_device_map = getattr(model, "hf_device_map", None)

    model = get_peft_model(model, lora_config)

    # Restore hf_device_map on the PeftModel so the HF Trainer recognises it
    # as a model-parallel model and skips DataParallel wrapping.
    if base_device_map and not getattr(model, "hf_device_map", None):
        model.hf_device_map = base_device_map
        print(f"[train] Restored hf_device_map on PeftModel ({len(base_device_map)} entries)")

    # Ensure PEFT/LoRA params and base-model params are placed consistently.
    # When `from_pretrained(..., device_map="auto")` is used the model can be
    # split across `cpu`/`cuda` with `hf_device_map` present — that can leave
    # some placeholders on the `meta` device and cause autograd device-mismatch
    # errors ("expected device meta but got cuda:0").  Clear any existing HF
    # device-map and move the PEFT-wrapped model to CPU so the Trainer /
    # accelerate runtime will perform a correct, consistent device placement.
    # Make sure LoRA adapter parameters live on the same device as their
    # corresponding base-layer parameters.  PEFT's LoraLayer already exposes
    # `_move_adapter_to_device_of_base_layer`, but call it explicitly here to
    # avoid any transient mismatches when the base model is dispatched across
    # devices (meta / cpu / cuda).
    try:
        from peft.tuners.lora import LoraLayer, ParamWrapper

        for module in model.modules():
            if isinstance(module, (LoraLayer, ParamWrapper)):
                # move every adapter present on this LoraLayer to the base-layer
                # device (no-op when already aligned)
                adapter_names = getattr(module, "r", None) or getattr(module, "lora_A", {}).keys()
                for adapter_name in list(adapter_names):
                    try:
                        module._move_adapter_to_device_of_base_layer(adapter_name)
                    except Exception:
                        # don't fail training on a best-effort device alignment;
                        # we'll surface any remaining mismatch below
                        pass
    except Exception:
        # if PEFT internals change or import fails, continue — we'll still
        # validate devices on the trainable params below
        pass

    # If the base model is intentionally loaded on CPU but CUDA is available,
    # place only the trainable LoRA/PEFT parameters on the GPU and *prevent*
    # the Trainer from moving the entire model to the training device. This
    # allows CPU-hosted base weights + GPU-resident adapters (small VRAM
    # footprint) so we can coexist with other GPU processes.
    if device_map == "cpu" and torch.cuda.is_available():
        cuda_dev = torch.device("cuda:0")
        moved = 0
        for name, p in model.named_parameters():
            if p.requires_grad and getattr(p, "device", None) is not None and p.device.type != "cuda":
                p.data = p.data.to(cuda_dev)
                if p.grad is not None:
                    p.grad = p.grad.to(cuda_dev)
                moved += 1
        # Signal to Trainer/Accelerate that the model is already dispatched
        # across devices (one entry isn't enough for `accelerate.verify_device_map`).
        # Use a small two-entry map so `prepare_model` will *not* call `.to(device)`
        # on the whole module (which would OOM). The exact keys don't matter for
        # our use-case — verify_device_map only checks `len(hf_device_map) > 1`.
        model.hf_device_map = {"base_cpu": "cpu", "lora_cuda": "cuda:0"}
        print(f"[train] Moved {moved} trainable LoRA params to {cuda_dev}; base model kept on CPU")

    model.print_trainable_parameters()

    # DEBUG: list all parameter names that have `requires_grad=True` so we can
    # confirm only the adapter/LoRA parameters (and not experts/FFN) are trainable.
    print("[train] Parameters with requires_grad=True (name -> device):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name} -> {getattr(param, 'device', 'unknown')}")

    # Sanity check — only fail when a *single layer* contains a mix of
    # trainable (LoRA) params on CUDA and base params on the `meta` device.
    # Having *other* layers on `meta` is acceptable so long as they are not
    # being trained / touched on the GPU.
    import re

    layer_index_pattern = re.compile(r"\.layers\.(\d+)\.")
    layer_info: dict[int, dict[str, bool]] = {}
    for name, p in model.named_parameters():
        m = layer_index_pattern.search(name)
        if not m:
            continue
        idx = int(m.group(1))
        info = layer_info.setdefault(idx, {"has_meta": False, "has_train_on_cuda": False})
        dev = getattr(p, 'device', None)
        if dev is not None and dev.type == 'meta':
            info['has_meta'] = True
        if p.requires_grad and dev is not None and dev.type == 'cuda':
            info['has_train_on_cuda'] = True

    mixed_layers = [i for i, info in layer_info.items() if info['has_meta'] and info['has_train_on_cuda']]
    if mixed_layers:
        raise RuntimeError(
            f"Device-mismatch detected in transformer layers: {mixed_layers}. "
            "This means LoRA adapters for these layers live on CUDA while the "
            "corresponding base parameters are not materialized (meta/cpu). "
            "Either load the base model with `device_map='cpu'` or restrict "
            "LoRA to the layers that are resident on the same device."
        )

    # --- Eval callback (TLC for spec-gen, TLAPS for prover) -----------------
    if prover:
        from src.training.tlaps_eval_callback import TLAPSEvalCallback
        eval_callback = TLAPSEvalCallback(
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            n_samples=2 if smoke_test else 6,
        )
    else:
        eval_callback = TLCEvalCallback(
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            n_samples=5 if smoke_test else 50,
        )

    # --- Trainer ------------------------------------------------------------
    # Pass `use_cpu=True` when the user explicitly requested `device_map='cpu'`
    # or when CUDA is not available so SFTConfig validates correctly.
    use_cpu_flag = (device_map == "cpu") or (not torch.cuda.is_available())
    training_args = build_training_args(
        smoke_test=smoke_test,
        resume_from=resume_from,
        use_cpu=use_cpu_flag,
        num_epochs=num_epochs,
        max_length=max_length,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps_override=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        output_dir=output_dir,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[eval_callback, ClearCacheCallback()],
    )

    print("[train] Starting training...")
    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except RuntimeError as err:
        # Detect common CUDA OOM and give actionable suggestions
        msg = str(err)
        if "out of memory" in msg.lower():
            print("[train] FATAL: CUDA out-of-memory during training startup.")
            print("[train] Suggestions:")
            print("  - free other GPU processes or set CUDA_VISIBLE_DEVICES to a free GPU")
            print("  - set --device-map=cpu to keep base model on CPU and reduce VRAM usage")
            print("  - reduce LoRA target layers (edit src/training/lora_config.yaml or use smaller device_map)")
            print("  - set PYTORCH_ALLOC_CONF=expandable_segments:True to reduce fragmentation")
            print("  - use --max-length 2048 (or 1536) to reduce activation memory")
        raise

    print(f"[train] Training complete. Checkpoints saved to {_CHECKPOINT_DIR}")

    if run_dpo_after:
        try:
            from src.training.train_dpo import run_after_sft as _run_dpo

            _run_dpo(
                model,
                tokenizer,
                device_map=device_map,
                max_gpu_memory_mb=max_gpu_memory_mb,
                max_length=max_length,
                smoke_test=smoke_test,
            )
        except Exception as e:
            print(f"[train] DPO phase skipped due to error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune gpt-oss-20b for TLA+ generation")
    parser.add_argument("--smoke-test", action="store_true", help="Run 10 steps with 10 examples (validates setup)")
    parser.add_argument("--resume",     default=None,        help="Resume from checkpoint path")
    parser.add_argument("--device-map", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device map to use when loading the base model (default: auto)")
    parser.add_argument("--max-gpu-memory-mb", type=int, default=None,
                        help="Cap (MB) for GPU memory that the loader may assign (used when device-map=auto)")
    parser.add_argument("--lora-layers", default=None,
                        help="Comma-separated list of transformer layer indices to apply LoRA to (e.g. '0,1,2')")
    parser.add_argument("--lora-top-k", type=int, default=None,
                        help="Apply LoRA only to the first K transformer layers (indices 0..K-1)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs (default: auto-scale by dataset size)")
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Max sequence length (default: 4096; lower values reduce VRAM on shared machines)")
    parser.add_argument("--per-device-batch-size", type=int, default=None,
                        help="Override per_device_train_batch_size (default: 1; use 1 for tight memory)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                        help="Override gradient_accumulation_steps (default: 2 smoke / 8 full; lower for tight memory)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max training steps (default: -1 = train full epochs)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (default: 1e-4)")
    parser.add_argument("--dpo-after", action="store_true",
                        help="After SFT, run DPO on gold pairs in data/processed/rl/dpo_pairs.jsonl (if >=2 rows)")
    parser.add_argument("--base-model", default=None,
                        help="Path or HF ID to load instead of openai/gpt-oss-20b "
                             "(e.g. outputs/merged_model/ for incremental DPO on a previously trained model)")
    parser.add_argument("--prover", action="store_true",
                        help="Train the prover model: read prover_train.jsonl/prover_eval.jsonl, "
                             "use TLAPSEvalCallback, save to outputs/checkpoints_prover/")
    args = parser.parse_args()

    # parse lora_layers override
    lora_layers_override = None
    if args.lora_layers:
        lora_layers_override = [int(x) for x in args.lora_layers.split(",") if x.strip()]

    main(
        smoke_test=args.smoke_test,
        resume_from=args.resume,
        device_map=args.device_map,
        max_gpu_memory_mb=args.max_gpu_memory_mb,
        lora_layers_override=lora_layers_override,
        lora_top_k=args.lora_top_k,
        num_epochs=args.epochs,
        max_length=args.max_length,
        run_dpo_after=args.dpo_after,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        base_model=args.base_model,
        prover=args.prover,
    )