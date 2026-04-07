"""
tlaps_validator.py — Wrapper around the TLAPS proof checker (tlapm).

Goal
----
A minimal, fast wrapper that runs `tlapm` on a .tla file containing
THEOREM/PROOF blocks and reports whether the proofs are machine-checkable.

Tiers
-----
proved        — all proof obligations discharged by backend solvers
partial       — at least one obligation proved, at least one failed
unproved      — file contains theorems but no obligation was discharged
parse_error   — tlapm could not parse the file
no_theorems   — file is well-formed TLA+ but contains no THEOREM
                (treated as a degenerate "success" by tlapm; we flag separately
                so reward functions can avoid handing free credit to specs that
                contain zero proof content)

Notes on this tlapm install (1.5.0)
-----------------------------------
- Multi-threaded mode crashes with `schedule.ml` assertion failure on this
  system. We always pass `--threads 1`.
- tlapm exits 0 even on failure, so we parse stdout/stderr.
- Bundled solvers live under src/shared/tlaps/lib/tlaps/bin (z3, zenon, etc.).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Resolve tlapm relative to repo root so this works in any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TLAPM = _REPO_ROOT / "src" / "shared" / "tlaps" / "bin" / "tlapm"

# Default per-spec timeout. Proofs are heavier than TLC runs but most small
# benchmarks finish in well under a minute.
_TLAPM_TIMEOUT_S = 120


@dataclass
class TLAPSResult:
    """Result of a tlapm run on a .tla file."""
    tier: str                              # proved | partial | unproved | parse_error | no_theorems
    obligations_total: int = 0
    obligations_proved: int = 0
    obligations_failed: int = 0
    errors: list[str] = field(default_factory=list)
    raw_output: str = ""
    runtime_seconds: float = 0.0
    tla_file: str = ""
    timed_out: bool = False

    @property
    def is_proved(self) -> bool:
        return self.tier == "proved" and self.obligations_total > 0


def validate_file(
    tla_path: Path,
    tlapm: Path = _DEFAULT_TLAPM,
    timeout: int = _TLAPM_TIMEOUT_S,
) -> TLAPSResult:
    """Run tlapm on a .tla file and parse its output.

    The file must EXTEND TLAPS to use proof constructs. The caller is
    responsible for arranging EXTENDS paths if the spec depends on
    non-standard modules.
    """
    if not tlapm.exists():
        raise FileNotFoundError(f"tlapm not found at {tlapm}")

    tla_path = tla_path.resolve()
    tla_text = tla_path.read_text(encoding="utf-8", errors="replace")
    if not _has_theorems(tla_text):
        return TLAPSResult(
            tier="no_theorems",
            tla_file=str(tla_path),
            raw_output="(skipped: no THEOREM declarations in file)",
        )

    cmd = [
        str(tlapm),
        "--threads", "1",       # 1.5.0 multi-thread bug
        "--stretch", "1",
        str(tla_path),
    ]
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tla_path.parent,
        )
        elapsed = time.monotonic() - t0
        output = result.stdout + result.stderr
        return _parse_result(output, tla_path, elapsed)
    except subprocess.TimeoutExpired:
        return TLAPSResult(
            tier="unproved",
            timed_out=True,
            runtime_seconds=float(timeout),
            tla_file=str(tla_path),
            raw_output=f"tlapm timed out after {timeout}s",
        )


def validate_string(
    tla_content: str,
    module_name: str = "Temp",
    tlapm: Path = _DEFAULT_TLAPM,
    timeout: int = _TLAPM_TIMEOUT_S,
) -> TLAPSResult:
    """Run tlapm on an in-memory spec string via a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tla_path = Path(tmpdir) / f"{module_name}.tla"
        tla_path.write_text(tla_content, encoding="utf-8")
        return validate_file(tla_path, tlapm=tlapm, timeout=timeout)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_THEOREM_RE = re.compile(r"^\s*THEOREM\b", re.MULTILINE)
_PARSE_ERR_RE = re.compile(r"Could not parse|Parser\.parse_file|syntax error", re.IGNORECASE)
# Examples we tested against tlapm 1.5.0:
#   "[INFO]: All 1 obligation proved."
#   "[INFO]: All 7 obligations proved."
#   "[ERROR]: 1/1 obligation failed."
#   "[ERROR]: 3/7 obligations failed."
_ALL_PROVED_RE = re.compile(r"All\s+(\d+)\s+obligations?\s+proved", re.IGNORECASE)
_FAILED_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+obligations?\s+failed", re.IGNORECASE)


def _has_theorems(tla_content: str) -> bool:
    return bool(_THEOREM_RE.search(tla_content))


def _parse_result(output: str, tla_path: Path, elapsed: float) -> TLAPSResult:
    if _PARSE_ERR_RE.search(output):
        return TLAPSResult(
            tier="parse_error",
            errors=[line for line in output.splitlines() if "error" in line.lower()][:5],
            raw_output=output,
            runtime_seconds=elapsed,
            tla_file=str(tla_path),
        )

    # tlapm prints one "[INFO]: All N obligation(s) proved." per processed module
    # (including any EXTENDed modules like the TLAPS stdlib stub, which reports 0).
    # Failures are reported as "[ERROR]: X/Y obligation(s) failed." Aggregate both
    # so a file with 1 success in user code + 0 in stdlib lands at total=1, not 0.
    proved_sum = sum(int(m) for m in _ALL_PROVED_RE.findall(output))

    failed_sum = 0
    total_from_failed = 0
    for failed_str, total_str in _FAILED_RE.findall(output):
        failed_sum += int(failed_str)
        total_from_failed += int(total_str)

    total = proved_sum + total_from_failed
    proved = proved_sum + max(total_from_failed - failed_sum, 0)

    if failed_sum > 0:
        tier = "partial" if proved > 0 else "unproved"
    elif proved > 0:
        tier = "proved"
    else:
        # No recognizable summary line at all, or only zero-obligation modules.
        tier = "unproved"

    return TLAPSResult(
        tier=tier,
        obligations_total=total,
        obligations_proved=proved,
        obligations_failed=failed_sum,
        errors=[line.strip() for line in output.splitlines()
                if line.strip().startswith("[ERROR]")][:10],
        raw_output=output,
        runtime_seconds=elapsed,
        tla_file=str(tla_path),
    )
