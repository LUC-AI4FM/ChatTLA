"""CPU-only tests for lora_resolver — runnable on a login node, no GPUs.

    python -m pytest test_lora_resolver.py -q
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lora_resolver import (
    assert_trainable_floor,
    is_moe,
    resolve_lora_config,
    trainable_pct,
)


def test_is_moe_gpt_oss():
    assert is_moe(SimpleNamespace(num_local_experts=32))


def test_is_moe_dense_qwen():
    # dense config: attribute absent entirely
    assert not is_moe(SimpleNamespace(hidden_size=4096))


def test_is_moe_rejects_degenerate_one_expert():
    assert not is_moe(SimpleNamespace(num_experts=1))


def test_resolver_moe_gets_target_parameters():
    cfg = resolve_lora_config(
        SimpleNamespace(num_local_experts=32),
        {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "bias": "none"},
    )
    assert cfg.target_parameters == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]
    assert cfg.target_modules == "all-linear"


def test_resolver_dense_has_no_target_parameters():
    cfg = resolve_lora_config(
        SimpleNamespace(hidden_size=4096),
        {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "bias": "none"},
    )
    assert not cfg.target_parameters
    assert cfg.target_modules == "all-linear"


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
