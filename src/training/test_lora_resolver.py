"""CPU-only tests for lora_resolver — runnable on a login node, no GPUs.

    python -m pytest test_lora_resolver.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

# Importable both from this directory (login-node usage above) and from the
# repo root, where pytest collects it as src.training.test_lora_resolver.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lora_resolver import (  # noqa: E402
    MOE_ATTENTION_TARGET_MODULES,
    apply_target_coverage,
    assert_trainable_floor,
    is_attention_only,
    is_moe,
    resolve_lora_config,
    resolve_target_coverage,
    trainable_pct,
)

# The yaml as shipped for gpt-oss: attention-only modules + expert parameters.
MOE_YAML = {
    "r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
}
DENSE_CONFIG = SimpleNamespace(hidden_size=4096)
MOE_CONFIG = SimpleNamespace(num_local_experts=32)


def test_is_moe_gpt_oss():
    assert is_moe(SimpleNamespace(num_local_experts=32))


def test_is_moe_dense_qwen():
    # dense config: attribute absent entirely
    assert not is_moe(SimpleNamespace(hidden_size=4096))


def test_is_moe_rejects_degenerate_one_expert():
    assert not is_moe(SimpleNamespace(num_experts=1))


class NestedConfig:
    """Qwen3.5/3.6 shape: expert count lives only on the nested text config."""

    def __init__(self, text_config):
        self.hidden_size = 4096  # outer object answers nothing about experts
        self._text_config = text_config

    def get_text_config(self):
        return self._text_config


def test_is_moe_sees_nested_text_config():
    # Qwen3.5-35B-A3B: 256 routed experts, invisible at the top level
    assert is_moe(NestedConfig(SimpleNamespace(num_experts=256, vocab_size=248320)))


def test_is_moe_nested_dense_stays_dense():
    # Qwen3.6-27B: nested, but genuinely dense
    assert not is_moe(NestedConfig(SimpleNamespace(vocab_size=248320)))


def test_is_moe_tolerates_get_text_config_raising():
    class Broken:
        def get_text_config(self):
            raise RuntimeError("no text config")

    assert not is_moe(Broken())


def test_is_moe_reads_plain_text_config_attribute():
    cfg = SimpleNamespace(text_config=SimpleNamespace(num_local_experts=128))
    assert is_moe(cfg)


def test_nested_moe_gets_expert_target_parameters():
    """The regression this closes: a nested MoE resolved as dense would get
    all-linear, which reaches mlp.shared_expert but not the packed routed
    experts -- and still clears the trainable floor, so nothing would catch it.
    """
    modules, params = resolve_target_coverage(
        NestedConfig(SimpleNamespace(num_experts=256)),
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        None,
    )
    assert params == ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    assert modules != "all-linear"


def test_is_attention_only_flags_the_qwen_arm_list():
    assert is_attention_only(["q_proj", "k_proj", "v_proj", "o_proj"])


def test_is_attention_only_false_when_ffn_named():
    assert not is_attention_only(["q_proj", "gate_proj", "down_proj"])


def test_is_attention_only_false_for_all_linear():
    # a string is expanded by PEFT over every nn.Linear, FFN included
    assert not is_attention_only("all-linear")


def test_resolver_moe_gets_target_parameters():
    cfg = resolve_lora_config(MOE_CONFIG, MOE_YAML)
    assert cfg.target_parameters == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]
    # MoE keeps the confirmed attention list; the experts arrive via parameters.
    # (LoraConfig normalizes a list of modules to a set.)
    assert set(cfg.target_modules) == set(MOE_ATTENTION_TARGET_MODULES)


def test_resolver_moe_forces_target_parameters_when_yaml_omits_them():
    yaml_cfg = dict(MOE_YAML)
    yaml_cfg.pop("target_parameters")
    cfg = resolve_lora_config(MOE_CONFIG, yaml_cfg)
    assert cfg.target_parameters == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]


def test_resolver_dense_has_no_target_parameters():
    cfg = resolve_lora_config(DENSE_CONFIG, MOE_YAML)
    assert not cfg.target_parameters


def test_resolver_dense_never_trains_attention_only():
    """The exact regression: the gpt-oss yaml applied to a dense model.

    Dropping target_parameters is not enough — leaving the attention-only
    module list is what froze Qwen's whole FFN at 0.0195% trainable.
    """
    cfg = resolve_lora_config(DENSE_CONFIG, MOE_YAML)
    assert cfg.target_modules == "all-linear"


def test_resolver_dense_respects_a_deliberate_ffn_list():
    yaml_cfg = dict(MOE_YAML, target_modules=["q_proj", "gate_proj", "down_proj"])
    cfg = resolve_lora_config(DENSE_CONFIG, yaml_cfg)
    assert set(cfg.target_modules) == {"q_proj", "gate_proj", "down_proj"}


def test_resolve_target_coverage_defaults_dense_with_no_yaml_modules():
    modules, params = resolve_target_coverage(DENSE_CONFIG)
    assert (modules, params) == ("all-linear", None)


def test_apply_target_coverage_mutates_in_place():
    """train.py's live path: yaml-built config, corrected for the architecture."""
    cfg = SimpleNamespace(
        target_modules=list(MOE_ATTENTION_TARGET_MODULES),
        target_parameters=["mlp.experts.gate_up_proj"],
    )
    assert apply_target_coverage(cfg, DENSE_CONFIG) is cfg
    assert cfg.target_modules == "all-linear"
    assert cfg.target_parameters is None


class TinyModel(nn.Module):
    def __init__(self, trainable_frac_target: float):
        super().__init__()
        self.frozen = nn.Linear(1000, 1000)  # ~1M params
        self.frozen.requires_grad_(False)
        n = max(int(1_001_000 * trainable_frac_target / 100), 2)
        self.adapter = nn.Parameter(torch.zeros(n))


def test_floor_aborts_on_frozen_model():
    m = TinyModel(0.0195)  # the broken Qwen run's fraction
    with pytest.raises(SystemExit):
        assert_trainable_floor(m)


def test_floor_passes_on_healthy_model():
    m = TinyModel(0.44)  # gpt-oss-20b's healthy fraction
    assert assert_trainable_floor(m) > 0.1


def test_trainable_pct_zero_safe():
    m = TinyModel(0.44)
    for p in m.parameters():
        p.requires_grad_(False)
    assert trainable_pct(m) == 0.0
