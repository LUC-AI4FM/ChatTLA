"""Architecture-aware LoRA resolution + trainable-parameter floor.

Fixes the silent failure that invalidated the 2026-07-25 Qwen3.6-27B arm
(results/analysis/w4_qwen36_arm_invalid.md in prove-TLA): a gpt-oss-hardcoded
``target_parameters=['mlp.experts.gate_up_proj','mlp.experts.down_proj']``
matched nothing on a dense model, PEFT no-op'd without warning, and the run
trained 0.0195% of the model while reporting success.

Usage from train.py, which reads r/alpha/dropout from the yaml and hands the
yaml's coverage fields here to be corrected for the architecture in hand::

    lora_cfg = load_lora_config()                       # yaml
    apply_target_coverage(lora_cfg, model.config)        # architecture
    model = get_peft_model(model, lora_cfg)
    assert_trainable_floor(model)   # aborts below the floor
"""
from __future__ import annotations

# Trainable fraction floor, in percent. gpt-oss-20b with correct MoE coverage
# trains 0.4401%; the broken dense run trained 0.0195%. 0.1% cleanly separates.
TRAINABLE_FLOOR_PCT = 0.1

MOE_EXPERT_TARGET_PARAMETERS = [
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
]

# MoE attention coverage. The 2026-07-09 grad-norm probe (job 7243501) confirmed
# this list plus MOE_EXPERT_TARGET_PARAMETERS is what reaches gpt-oss experts.
MOE_ATTENTION_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Dense coverage: one string PEFT expands to every nn.Linear, attention and FFN.
DENSE_TARGET_MODULES = "all-linear"

# Substrings that mark a module name as feed-forward rather than attention.
# Covers gpt-oss/Llama/Qwen (gate/up/down_proj), GPT-2 (c_fc), BERT-ish (fc1/fc2),
# and Mixtral's w1/w2/w3.
_FFN_NAME_HINTS = (
    "mlp", "ffn", "feed_forward",
    "gate_proj", "up_proj", "down_proj", "gate_up_proj",
    "fc1", "fc2", "c_fc", "w1", "w2", "w3",
)


def is_moe(model_config) -> bool:
    """A model is MoE iff its config declares a positive expert count."""
    for attr in ("num_local_experts", "num_experts", "n_routed_experts"):
        n = getattr(model_config, attr, None)
        if isinstance(n, int) and n > 1:
            return True
    return False


def is_attention_only(target_modules) -> bool:
    """True when an explicit target_modules list names no feed-forward module.

    A string ("all-linear", or a regex) is never attention-only: PEFT expands
    it over every nn.Linear, FFN included.
    """
    if not isinstance(target_modules, (list, tuple, set)):
        return False
    return not any(hint in m for m in target_modules for hint in _FFN_NAME_HINTS)


def resolve_target_coverage(model_config, target_modules=None, target_parameters=None):
    """Return (target_modules, target_parameters) whose FFN coverage matches
    the architecture in hand.

    MoE (gpt-oss family): the experts are 3D packed params, invisible to both
    "all-linear" and any module list, so ``target_parameters`` is mandatory and
    is forced on even if the yaml forgot it.

    Dense (Qwen, Llama, ...): there are no expert params to name, so
    ``target_parameters`` matches nothing and PEFT no-op's it silently. Worse,
    the MoE yaml's attention-only module list would then freeze the entire FFN
    — that pairing is exactly what trained 0.0195% of Qwen3.6-27B. Fall back to
    "all-linear", which covers attention and FFN both.
    """
    if is_moe(model_config):
        modules = target_modules or list(MOE_ATTENTION_TARGET_MODULES)
        params = list(target_parameters or MOE_EXPERT_TARGET_PARAMETERS)
        if not target_parameters:
            print("[lora_resolver] MoE: forcing expert target_parameters", params)
        print(f"[lora_resolver] architecture=MoE target_modules={modules} "
              f"target_parameters={params}")
        return modules, params

    modules = target_modules
    if modules is None or is_attention_only(modules):
        print(f"[lora_resolver] dense model: target_modules={modules} would freeze "
              f"the FFN; using {DENSE_TARGET_MODULES!r} instead")
        modules = DENSE_TARGET_MODULES
    if target_parameters:
        print("[lora_resolver] dense model: dropping MoE target_parameters",
              list(target_parameters))
    print(f"[lora_resolver] architecture=dense target_modules={modules} "
          f"target_parameters=None")
    return modules, None


def apply_target_coverage(lora_config, model_config):
    """Correct an already-built LoraConfig in place. Returns it for chaining."""
    lora_config.target_modules, lora_config.target_parameters = resolve_target_coverage(
        model_config,
        getattr(lora_config, "target_modules", None),
        getattr(lora_config, "target_parameters", None),
    )
    return lora_config


def resolve_lora_config(model_config, yaml_cfg: dict):
    """Build a LoraConfig from yaml hyperparameters + resolved coverage."""
    from peft import LoraConfig

    modules, params = resolve_target_coverage(
        model_config,
        yaml_cfg.get("target_modules"),
        yaml_cfg.get("target_parameters"),
    )
    return LoraConfig(
        r=yaml_cfg["r"],
        lora_alpha=yaml_cfg["lora_alpha"],
        lora_dropout=yaml_cfg["lora_dropout"],
        bias=yaml_cfg["bias"],
        target_modules=modules,
        target_parameters=params,
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
