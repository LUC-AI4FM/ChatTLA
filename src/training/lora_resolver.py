"""Architecture-aware LoRA resolution + trainable-parameter floor.

Fixes the silent failure that invalidated the 2026-07-25 Qwen3.6-27B arm
(results/analysis/w4_qwen36_arm_invalid.md in prove-TLA): a gpt-oss-hardcoded
``target_parameters=['mlp.experts.gate_up_proj','mlp.experts.down_proj']``
matched nothing on a dense model, PEFT no-op'd without warning, and the run
trained 0.0195% of the model while reporting success.

Drop-in usage from train.py::

    from lora_resolver import resolve_lora_config, assert_trainable_floor
    lora_cfg = resolve_lora_config(model.config, yaml_cfg)
    model = get_peft_model(model, lora_cfg)
    assert_trainable_floor(model)   # aborts (exit 3) below the floor
"""
from __future__ import annotations

# Trainable fraction floor, in percent. gpt-oss-20b with correct MoE coverage
# trains 0.4401%; the broken dense run trained 0.0195%. 0.1% cleanly separates.
TRAINABLE_FLOOR_PCT = 0.1

MOE_EXPERT_TARGET_PARAMETERS = [
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
]


def is_moe(model_config) -> bool:
    """A model is MoE iff its config declares a positive expert count."""
    for attr in ("num_local_experts", "num_experts", "n_routed_experts"):
        n = getattr(model_config, attr, None)
        if isinstance(n, int) and n > 1:
            return True
    return False


def resolve_lora_config(model_config, yaml_cfg: dict):
    """Build a LoraConfig whose FFN coverage matches the architecture.

    MoE (gpt-oss family): all-linear + target_parameters for the packed
    expert tensors that all-linear cannot see.
    Dense (Qwen, Llama, ...): all-linear alone covers attention AND FFN;
    target_parameters is forced to None so a stale yaml cannot reintroduce
    the no-op selector.
    """
    from peft import LoraConfig

    moe = is_moe(model_config)
    target_parameters = list(MOE_EXPERT_TARGET_PARAMETERS) if moe else None
    print(f"[lora_resolver] architecture={'MoE' if moe else 'dense'} "
          f"target_modules=all-linear target_parameters={target_parameters}")
    return LoraConfig(
        r=yaml_cfg["r"],
        lora_alpha=yaml_cfg["lora_alpha"],
        lora_dropout=yaml_cfg["lora_dropout"],
        bias=yaml_cfg["bias"],
        target_modules="all-linear",
        target_parameters=target_parameters,
        task_type="CAUSAL_LM",
    )


def trainable_pct(model) -> float:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return 100.0 * trainable / max(total, 1)


def assert_trainable_floor(model, floor_pct: float = TRAINABLE_FLOOR_PCT) -> float:
    """Abort the process (exit 3) if the adapter barely attached.

    A 0.0195%-trainable run must never exit 0 — that silent success is what
    made the Qwen arm expensive.
    """
    pct = trainable_pct(model)
    print(f"[lora_resolver] trainable = {pct:.4f}% (floor {floor_pct}%)")
    if pct < floor_pct:
        raise SystemExit(
            f"FATAL: trainable parameters {pct:.4f}% < floor {floor_pct}%. "
            f"LoRA selector likely matched nothing on this architecture. "
            f"Refusing to train a frozen model."
        )
    return pct
