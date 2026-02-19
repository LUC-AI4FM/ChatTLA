"""
sany_validator.py — Wrapper around the TLA+ SANY syntax checker.

SANY (Syntax ANalysis sYstem) is the TLA+ parser shipped inside tla2tools.jar.
Running it against a .tla file is the minimum bar a spec must clear before it
can enter training data.  A SANY-clean spec is "silver tier".

Research note
-------------
We invoke SANY via `java -jar tla2tools.jar -tool SANY <file>` rather than
calling the Java API directly.  This keeps the Python layer clean, avoids JNI
complexity, and mirrors exactly what practitioners do at their desks.

The SANY output format is line-oriented:
  - "Parsing file ..." lines are informational
  - Lines containing "errors detected" with a non-zero count are failures
  - "No error" in the summary = success
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Path to bundled TLC jar relative to this file's location.
_TLA_TOOLS_JAR = Path(__file__).resolve().parents[1] / "shared" / "tlc" / "tla2tools.jar"


@dataclass
class SANYResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    raw_output: str = ""
    tla_file: str = ""


def validate_file(tla_path: Path, jar: Path = _TLA_TOOLS_JAR) -> SANYResult:
    """
    Run SANY on an existing .tla file on disk.

    Parameters
    ----------
    tla_path : Path   Path to the .tla file.
    jar      : Path   Path to tla2tools.jar (default: bundled copy).

    Returns
    -------
    SANYResult with `valid=True` if SANY reports zero errors.
    """
    if not jar.exists():
        raise FileNotFoundError(
            f"tla2tools.jar not found at {jar}. "
            "Run: wget -O src/shared/tlc/tla2tools.jar "
            "https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar"
        )

    cmd = ["java", "-jar", str(jar), "-tool", "SANY", str(tla_path)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        stdout = result.stdout + result.stderr
        errors = _parse_errors(stdout)
        valid = len(errors) == 0 and _detect_success(stdout)
        return SANYResult(
            valid=valid,
            errors=errors,
            raw_output=stdout,
            tla_file=str(tla_path),
        )
    except subprocess.TimeoutExpired:
        return SANYResult(
            valid=False,
            errors=["SANY timed out after 30s"],
            raw_output="",
            tla_file=str(tla_path),
        )


def validate_string(tla_content: str, module_name: str = "Temp", jar: Path = _TLA_TOOLS_JAR) -> SANYResult:
    """
    Run SANY on an in-memory TLA+ string by writing it to a temp file first.

    Parameters
    ----------
    tla_content : str   Full .tla spec text.
    module_name : str   Used as the temp file stem (SANY infers module name
                        from filename, so they must match).
    jar         : Path  Path to tla2tools.jar.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tla_path = Path(tmpdir) / f"{module_name}.tla"
        tla_path.write_text(tla_content, encoding="utf-8")
        return validate_file(tla_path, jar=jar)


def _detect_success(output: str) -> bool:
    """SANY prints 'No errors' on success (case-insensitive match)."""
    return bool(re.search(r"no\s+error", output, re.IGNORECASE))


def _parse_errors(output: str) -> list[str]:
    """
    Extract error lines from SANY output.
    SANY error lines typically start with "***" or contain "Error:".
    """
    errors: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("***") or re.match(r"^\s*Error", stripped, re.IGNORECASE):
            errors.append(stripped)
    return errors
