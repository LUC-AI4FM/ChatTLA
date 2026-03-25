"""Tests for TLC MC* model-check shim detection."""

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relpath,expected_shim",
    [
        (
            "data/external/tlaplus-examples/specifications/PaxosHowToWinATuringAward/MCConsensus.tla",
            True,
        ),
        (
            "data/external/tlaplus-examples/specifications/PaxosHowToWinATuringAward/Consensus.tla",
            False,
        ),
        ("data/FormaLLM/data/KeyValueStore/tla/MCKVS.tla", True),
        ("data/FormaLLM/data/FiniteMonotonic/tla/MCCRDT.tla", True),
    ],
)
def test_model_check_shim_examples(relpath: str, expected_shim: bool) -> None:
    from src.training.module_family import is_model_check_shim, parse_module_name

    p = _REPO / relpath
    if not p.is_file():
        pytest.skip(f"missing: {relpath}")
    t = p.read_text(encoding="utf-8", errors="replace")
    mn = parse_module_name(t)
    assert is_model_check_shim(mn, t) is expected_shim
