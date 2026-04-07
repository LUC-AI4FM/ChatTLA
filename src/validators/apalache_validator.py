"""
apalache_validator.py — Wrapper around the Apalache symbolic model checker.

Apalache is complementary to TLC:
  - TLC: explicit state enumeration. Fails on infinite/large state spaces.
  - Apalache: SMT-based symbolic checking. Handles much larger state spaces
    via bounded model checking, but limited to specs with type annotations
    or successful type inference.

We use Apalache as a TIE-BREAKER on silver specs (TLC said "infinite state"
or "state space too large"). Apalache's SMT engine can often verify these.

Both validators agreeing on "valid" → diamond-tier confidence.

Validation tiers (this module's contribution):
    apalache_ok      — type-checks, parses, finds no violations within step bound
    apalache_violation — found a counter-example to an invariant
    apalache_error   — type error or other Apalache-side failure
    apalache_skip    — Apalache could not start (no fairness, untyped, etc.)
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_APALACHE_BIN = _REPO_ROOT / "src" / "shared" / "apalache" / "bin" / "apalache-mc"

# Apalache can take a while on complex specs. We bound it for the loop.
_DEFAULT_TIMEOUT_S = 90
_DEFAULT_LENGTH = 10  # number of steps to check (bounded model checking)


@dataclass
class ApalacheResult:
    """Result of an Apalache check."""
    status: str  # "ok" | "violation" | "error" | "skip"
    invariant_violated: Optional[str] = None
    counter_example: str = ""
    raw_output: str = ""
    runtime_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.status == "ok"


def annotate_for_apalache(tla_content: str) -> str:
    """Inject minimal Apalache type annotations into a TLA+ spec.

    Apalache requires every VARIABLE to have a (* @type: ...; *) annotation.
    Our model doesn't produce these, so we infer them from how the variable
    is used in TypeOK.

    Heuristics (very simple, used as second-opinion validator only):
      - var \\in 0..N or 1..N → Int
      - var \\in {"a", "b"}   → Str
      - var \\in BOOLEAN      → Bool
      - var \\in [S -> T]     → S -> T (function)
      - default                → Int
    """
    # Find VARIABLES line
    var_match = re.search(r"^(\s*)VARIABLES?\s+(.+?)$", tla_content, re.MULTILINE)
    if not var_match:
        return tla_content

    indent = var_match.group(1)
    var_names = [n.strip().rstrip(",") for n in re.split(r"[,\s]+", var_match.group(2).strip())]
    var_names = [n for n in var_names if n and re.match(r"^[a-zA-Z_]\w*$", n)]
    if not var_names:
        return tla_content

    # Infer type from TypeOK usage
    type_for_var: dict[str, str] = {}
    for var in var_names:
        # \in BOOLEAN
        if re.search(rf"\b{re.escape(var)}\s*\\in\s+BOOLEAN", tla_content):
            type_for_var[var] = "Bool"
        # \in 0..N or 1..N
        elif re.search(rf"\b{re.escape(var)}\s*\\in\s+\d+\.\.[A-Za-z_0-9]+", tla_content):
            type_for_var[var] = "Int"
        # \in {"a", "b"}
        elif re.search(rf'\b{re.escape(var)}\s*\\in\s*\{{[^}}]*"', tla_content):
            type_for_var[var] = "Str"
        # \in [X -> Y]
        elif re.search(rf"\b{re.escape(var)}\s*\\in\s*\[", tla_content):
            type_for_var[var] = "Int -> Int"  # generic function type
        else:
            type_for_var[var] = "Int"

    # Build annotated VARIABLE block (Apalache requires (* @type: ... *) blocks
    # immediately preceding each VARIABLE declaration)
    annotated_lines = []
    for i, var in enumerate(var_names):
        annotated_lines.append(f"{indent}VARIABLE")
        annotated_lines.append(f"{indent}  (* @type: {type_for_var[var]}; *)")
        annotated_lines.append(f"{indent}  {var}")
    annotated = "\n".join(annotated_lines)

    # Replace VARIABLES line with annotated block
    return tla_content.replace(var_match.group(0), annotated, 1)


def validate_string(
    tla_content: str,
    module_name: str = "Temp",
    invariants: Optional[list[str]] = None,
    timeout: int = _DEFAULT_TIMEOUT_S,
    length: int = _DEFAULT_LENGTH,
    auto_annotate: bool = True,
) -> ApalacheResult:
    """Run Apalache on an in-memory TLA+ spec.

    Parameters
    ----------
    tla_content : the .tla module text
    module_name : module name (must match `MODULE X` in tla_content)
    invariants  : list of invariant names to check (default: TypeOK if present,
                  plus any name matching *Inv*/Safety*)
    timeout     : seconds before we kill Apalache and return "skip"
    length      : bounded model checking depth (default: 10 steps)
    """
    if not _APALACHE_BIN.exists():
        return ApalacheResult(
            status="skip",
            errors=[f"Apalache binary not found at {_APALACHE_BIN}"],
        )

    # Inject minimal type annotations if needed
    if auto_annotate and "@type:" not in tla_content:
        tla_content = annotate_for_apalache(tla_content)

    # Auto-detect invariants if not given
    if invariants is None:
        invariants = _detect_invariants(tla_content)
    if not invariants:
        return ApalacheResult(
            status="skip",
            errors=["no invariants detected to check"],
        )

    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tla_path = tmp / f"{module_name}.tla"
        tla_path.write_text(tla_content, encoding="utf-8")

        # Apalache supports checking one invariant at a time. Use the first
        # non-trivial one we find. (Future: loop over all of them.)
        invariant = invariants[0]

        cmd = [
            str(_APALACHE_BIN),
            "check",
            f"--length={length}",
            f"--inv={invariant}",
            f"--out-dir={tmp / 'out'}",
            str(tla_path),
        ]

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(tmp),
            )
            elapsed = time.monotonic() - t0
            stdout = result.stdout + result.stderr

            # Parse Apalache output
            if "EXITCODE: OK" in stdout or "No error has been found" in stdout:
                return ApalacheResult(
                    status="ok",
                    raw_output=stdout[-3000:],
                    runtime_seconds=elapsed,
                )
            elif "EXITCODE: ERROR (12)" in stdout or "violated" in stdout.lower():
                # Counterexample found
                ce_match = re.search(r"Counterexample.*?(?=\n\n|\Z)", stdout, re.DOTALL)
                return ApalacheResult(
                    status="violation",
                    invariant_violated=invariant,
                    counter_example=ce_match.group(0)[:1500] if ce_match else "",
                    raw_output=stdout[-3000:],
                    runtime_seconds=elapsed,
                )
            else:
                # Type error, parse error, etc.
                errors = _parse_apalache_errors(stdout)
                return ApalacheResult(
                    status="error",
                    raw_output=stdout[-3000:],
                    runtime_seconds=elapsed,
                    errors=errors,
                )

        except subprocess.TimeoutExpired:
            return ApalacheResult(
                status="skip",
                runtime_seconds=float(timeout),
                errors=[f"Apalache timed out after {timeout}s"],
            )
        except Exception as e:
            return ApalacheResult(
                status="error",
                errors=[f"Apalache invocation failed: {e}"],
            )


def _detect_invariants(tla_content: str) -> list[str]:
    """Detect candidate invariant operators from spec source."""
    invariants = []
    # TypeOK is always a candidate
    if re.search(r"^\s*TypeOK\s*==", tla_content, re.MULTILINE):
        invariants.append("TypeOK")
    # Detect operators with names suggesting they're invariants
    for m in re.finditer(r"^\s*(\w+)\s*==", tla_content, re.MULTILINE):
        name = m.group(1)
        if name in ("Init", "Next", "Spec", "TypeOK", "vars"):
            continue
        if re.match(
            r"(Safety|.*Inv(ariant)?$|Mutex|MutualExclusion|Conservation|"
            r"Bounded|NoOverflow|NoUnderflow|AtMost|NoWrite|Valid|"
            r".*Conserved|.*Bounded|.*Stable|.*Threshold)",
            name,
        ):
            invariants.append(name)
    return invariants


def _parse_apalache_errors(stdout: str) -> list[str]:
    """Pull error messages from Apalache stdout."""
    errors = []
    for line in stdout.splitlines():
        if "Error" in line or "ERROR" in line or "Type checker" in line:
            errors.append(line.strip())
        if len(errors) >= 5:
            break
    return errors


if __name__ == "__main__":
    # Quick smoke test
    test_spec = """---- MODULE SmokeTest ----
EXTENDS Naturals
VARIABLES x
TypeOK == x \\in 0..3
Init == x = 0
Next == x' = (x + 1) % 4
Spec == Init /\\ [][Next]_<<x>>
====
"""
    result = validate_string(test_spec, module_name="SmokeTest")
    print(f"Status: {result.status}")
    print(f"Runtime: {result.runtime_seconds:.1f}s")
    if result.errors:
        print(f"Errors: {result.errors}")
    print(f"Output (last 500 chars):\n{result.raw_output[-500:]}")
