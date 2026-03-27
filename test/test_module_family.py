"""Tests for TLC MC* model-check shim detection and spec-context gaps."""

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


def test_missing_context_extends_skips_stdlib() -> None:
    from src.training.module_family import format_spec_context_gap_notice, missing_context_module_names

    tla = (
        "---- MODULE MCConsensus ----\n"
        "EXTENDS Consensus, Naturals, TLC\n"
        "VARIABLE x\n"
        "===="
    )
    assert missing_context_module_names(tla) == ("Consensus",)
    notice = format_spec_context_gap_notice(tla)
    assert notice is not None
    assert "Consensus" in notice
    assert "Naturals" not in notice


def test_missing_context_instance() -> None:
    from src.training.module_family import missing_context_module_names

    tla = "---- MODULE Foo ----\nEXTENDS Naturals\nBar == INSTANCE Voting\n===="
    assert "Voting" in missing_context_module_names(tla)


def test_defined_modules_suppresses_gap() -> None:
    from src.training.module_family import missing_context_module_names

    tla = "---- MODULE Foo ----\nEXTENDS Bar\n===="
    assert missing_context_module_names(tla, defined_modules=frozenset({"Foo", "Bar"})) == ()


def test_no_gap_when_only_stdlib_extends() -> None:
    from src.training.module_family import missing_context_module_names

    tla = "---- MODULE X ----\nEXTENDS Naturals, Integers\n===="
    assert missing_context_module_names(tla) == ()
