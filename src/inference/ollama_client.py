"""
ollama_client.py — Local Ollama inference client for ChatTLA.

Wraps the Ollama Python SDK to provide a clean API for TLA+ spec generation
using the fine-tuned ChatTLA model (or the base gpt-oss:20b for comparison).

The client always applies the gpt-oss harmony format via openai-harmony.
Without this the model produces degraded output.

Models
------
  chattla:20b      — Fine-tuned ChatTLA model (after convert_to_gguf.py)
  gpt-oss:20b      — Base model baseline

Reasoning levels
----------------
  low    — fast responses, suitable for interactive use
  medium — balanced; default for spec generation
  high   — deep analysis; use for complex distributed systems specs

Usage
-----
    from src.inference.ollama_client import ChatTLAClient

    client = ChatTLAClient()
    spec = client.generate_spec("A distributed read-write lock with N readers and 1 writer.")
    print(spec)

    # Async:
    import asyncio
    spec = asyncio.run(client.agenerate_spec("Two-phase commit protocol."))
"""

from __future__ import annotations

import os
import re
from typing import Optional

_OLLAMA_HOST   = os.getenv("OLLAMA_HOST",   "http://localhost:11434")
_DEFAULT_MODEL = os.getenv("CHATTLA_MODEL", "chattla:20b")

_DEVELOPER_PROMPT = """\
You are ChatTLA, an expert at writing verified TLA+ formal specifications.
Respond only with the TLA+ module, no commentary or explanation.
1. Start the module with ---- MODULE <ModuleName> ----
2. End with ====
3. Include EXTENDS, VARIABLES, Init, Next, and Spec operators
4. After the TLA+ module, append a TLC configuration block:
   SPECIFICATION Spec
   INVARIANT TypeOK   (if TypeOK is defined)
\
"""


def _build_harmony_prompt(developer_content: str, user_content: str) -> str:
    """Build a raw harmony-format prompt that forces TLA+ code output.

    gpt-oss-20b uses the harmony prompt format with channels (analysis, final).
    By jumping straight to ``<|channel|>final<|message|>`` AND seeding the
    output with ``---- MODULE``, we prevent the model from entering a
    degenerate analysis loop and force it to produce TLA+ immediately.
    """
    return (
        f"<|start|>system<|message|>You are ChatTLA, an expert at writing verified TLA+ formal specifications.<|end|>\n"
        f"<|start|>developer<|message|>{developer_content}<|end|>\n"
        f"<|start|>user<|message|>{user_content}<|end|>\n"
        f"<|start|>assistant<|channel|>final<|message|>---- MODULE"
    )


class ChatTLAClient:
    """
    Ollama client for TLA+ spec generation via the locally hosted ChatTLA model.

    Parameters
    ----------
    model         : str   Ollama model tag (default: chattla:20b)
    host          : str   Ollama server host (default: http://localhost:11434)
    reasoning     : str   Harmony reasoning level: "low", "medium", or "high"
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host:  str = _OLLAMA_HOST,
        reasoning: str = "medium",
    ):
        import ollama  # lazy import

        self.model     = model
        self.reasoning = reasoning
        self._client   = ollama.Client(host=host)

    def generate_spec(
        self,
        nl_description: str,
        module_name: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """
        Generate a TLA+ specification from a natural-language description.

        Parameters
        ----------
        nl_description : str   Plain-English description of the system to model.
        module_name    : str   Desired module name (hinted to the model if provided).
        temperature    : float Sampling temperature; lower = more deterministic.

        Returns
        -------
        str   The extracted TLA+ module text (---- MODULE ... ====).
              Returns the raw output if delimiters cannot be found.
        """
        user_content = nl_description.strip()
        if module_name:
            user_content += f"\n\nUse module name: {module_name}"

        developer_content = f"{_DEVELOPER_PROMPT}\nReasoning: {self.reasoning}"
        prompt = _build_harmony_prompt(developer_content, user_content)

        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            raw=True,
            options={
                "temperature": temperature,
                "repeat_penalty": 1.3,
                "num_predict": 2048,
                "top_k": 40,
                "top_p": 0.9,
                "stop": ["<|end|>", "<|start|>", "\n===="],
            },
        )
        # Reconstruct: prompt seeded "---- MODULE", model continues from there
        raw = "---- MODULE" + response["response"]
        # Ensure the module has a closing delimiter
        if "====" not in raw:
            raw += "\n===="
        return _extract_tla(raw)

    def validate_and_generate(
        self,
        nl_description: str,
        max_retries: int = 3,
    ) -> tuple[str, str]:
        """
        Generate a spec and run TLC validation.  If TLC reports errors,
        feed the error back to the model for self-correction (up to max_retries).

        The correction loop now distinguishes SANY failures (syntax) from TLC
        failures (semantic), providing targeted error feedback at each stage.

        Returns
        -------
        (spec: str, tier: str)   Final spec text and validation tier ("gold"|"silver"|"bronze").
        """
        from src.validators.sany_validator import validate_string as sany_validate
        from src.validators.tlc_validator import validate_string

        spec = self.generate_spec(nl_description)

        # Pre-process: strip common generation artefacts before validation
        spec = _sanitize_spec(spec)

        # Detect PlusCal — if present, don't try to fix it, just regenerate
        # with an explicit "no PlusCal" hint. PlusCal can't be mechanically
        # converted to valid pure TLA+ by stripping alone.
        had_pluscal = bool(re.search(
            r"--(?:fair\s+)?algorithm\b|BEGIN TRANSLATION|end\s+algorithm",
            spec, re.IGNORECASE
        ))

        # Apply deterministic Python fixer before any SANY/TLC validation.
        # This catches the ~20 most common syntax patterns the model gets wrong
        # (e.g. \notin, double-prime, missing commas, alignment issues) and
        # avoids wasting self-correction retries on mechanically-fixable errors.
        from src.training.self_improve import fix_tla_syntax, validate_with_sany
        if not had_pluscal:
            fix_result = fix_tla_syntax(spec)
            if fix_result.fixes_applied:
                # Check if the Python-fixed version passes SANY
                is_valid, _ = validate_with_sany(fix_result.fixed_spec)
                if is_valid:
                    spec = fix_result.fixed_spec

        for attempt in range(max_retries):
            m = re.search(r"----\s*MODULE\s+(\w+)", spec)
            module_name = m.group(1) if m else "Generated"

            # If the spec has PlusCal, skip validation and force a regeneration
            # with an explicit "no PlusCal" instruction.
            if re.search(r"--(?:fair\s+)?algorithm\b|end\s+algorithm", spec, re.IGNORECASE):
                spec = self._self_correct_sany(
                    spec,
                    "CRITICAL: Your spec uses PlusCal syntax (--algorithm, begin, "
                    "end algorithm, :=, labels). PlusCal is NOT pure TLA+ and cannot "
                    "be parsed by SANY. You MUST rewrite using only pure TLA+ operators: "
                    "Init ==, Next ==, /\\, \\/, UNCHANGED, primed variables (x'), etc. "
                    "Do NOT use --algorithm, begin, end algorithm, while, if/then, or :=.",
                    attempt,
                )
                spec = _sanitize_spec(spec)
                continue

            # Step 1: SANY check first (fast, catches syntax issues)
            sany_result = sany_validate(spec, module_name=module_name)
            if not sany_result.valid:
                # Try Python fixer before burning a self-correction attempt
                fix_result = fix_tla_syntax(spec, "\n".join(sany_result.errors[:5]))
                if fix_result.fixes_applied:
                    fixed_sany = sany_validate(fix_result.fixed_spec, module_name=module_name)
                    if fixed_sany.valid:
                        spec = fix_result.fixed_spec
                        continue  # Re-enter loop for TLC check

                error_detail = "\n".join(sany_result.errors[:5])
                if not error_detail:
                    error_detail = sany_result.raw_output[-500:]
                spec = self._self_correct_sany(spec, error_detail, attempt)
                spec = _sanitize_spec(spec)
                continue

            # Step 2: Full TLC check
            result = validate_string(spec, module_name=module_name)

            if result.tier == "gold":
                return spec, "gold"
            if result.tier == "silver":
                return spec, "silver"

            # Bronze with TLC errors: feed TLC violations back
            error_summary = "\n".join(result.tlc_violations[:5])
            spec = self._self_correct(spec, error_summary)
            spec = _sanitize_spec(spec)

        # Final validation after all retries
        m = re.search(r"----\s*MODULE\s+(\w+)", spec)
        module_name = m.group(1) if m else "Generated"
        result = validate_string(spec, module_name=module_name)
        return spec, result.tier

    def _self_correct_sany(self, buggy_spec: str, sany_errors: str, attempt: int) -> str:
        """Ask the model to fix SANY parse errors with targeted guidance."""
        developer_content = f"{_DEVELOPER_PROMPT}\nReasoning: high"

        # Build targeted hints based on common SANY failure patterns
        hints = _diagnose_sany_errors(buggy_spec, sany_errors)

        user_content = (
            f"This TLA+ spec has SANY parse errors (attempt {attempt + 1}):\n\n"
            f"SANY errors:\n{sany_errors}\n\n"
        )
        if hints:
            user_content += f"Known issues to fix:\n{hints}\n\n"
        user_content += (
            f"Buggy spec:\n{buggy_spec}\n\n"
            f"Fix ALL syntax errors. Output only pure TLA+ (no PlusCal, no markdown)."
        )
        prompt = _build_harmony_prompt(developer_content, user_content)

        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            raw=True,
            options={
                "temperature": 0.1,
                "repeat_penalty": 1.3,
                "num_predict": 2048,
                "stop": ["<|end|>", "<|start|>", "\n===="],
            },
        )
        raw = "---- MODULE" + response["response"]
        if "====" not in raw:
            raw += "\n===="
        return _extract_tla(raw)

    def _self_correct(self, buggy_spec: str, error_msg: str) -> str:
        """Ask the model to fix a spec given a TLC error message."""
        developer_content = f"{_DEVELOPER_PROMPT}\nReasoning: {self.reasoning}"
        user_content = (
            f"This TLA+ spec has errors:\n{error_msg}\n\n"
            f"Buggy spec:\n{buggy_spec}\n\nFix the spec and output only the corrected TLA+ module."
        )
        prompt = _build_harmony_prompt(developer_content, user_content)

        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            raw=True,
            options={
                "temperature": 0.1,
                "repeat_penalty": 1.3,
                "num_predict": 2048,
                "stop": ["<|end|>", "<|start|>", "\n===="],
            },
        )
        raw = "---- MODULE" + response["response"]
        if "====" not in raw:
            raw += "\n===="
        return _extract_tla(raw)


def _extract_tla(text: str) -> str:
    """Extract ---- MODULE ... ==== block from model output."""
    m = re.search(r"(----\s*MODULE\b.*?====)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _sanitize_spec(spec: str) -> str:
    """
    Apply rule-based fixes for common generation artefacts that cause SANY failures.

    These are patterns the model consistently produces that are never valid TLA+:
    - PlusCal algorithm blocks (--algorithm / --fair algorithm)
    - Markdown code fences
    - Repeated comment blocks (degenerate repetition)
    - Trailing garbage after ====
    """
    # Remove PlusCal blocks: (* --algorithm ... end algorithm; *)
    spec = re.sub(
        r"\(\*\s*--(?:fair\s+)?algorithm\b.*?end\s+algorithm\s*;?\s*\*\)",
        "",
        spec,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove standalone PlusCal keywords that leak into TLA+
    spec = re.sub(r"^\s*(begin|end\s+algorithm|macro|procedure)\b.*$", "", spec, flags=re.MULTILINE | re.IGNORECASE)

    # Remove markdown code fences
    spec = re.sub(r"^```\w*\s*$", "", spec, flags=re.MULTILINE)

    # Truncate degenerate repetition (same block repeated 3+ times)
    lines = spec.splitlines()
    if len(lines) > 40:
        spec = _dedup_repeated_blocks(lines)

    # Ensure module ends at ==== (strip any trailing noise)
    m = re.search(r"(----\s*MODULE\b.*?====)", spec, re.DOTALL)
    if m:
        spec = m.group(1)

    return spec.strip()


def _dedup_repeated_blocks(lines: list[str], window: int = 5) -> str:
    """
    Detect and collapse degenerate repetition where the model outputs
    the same N-line block over and over (common with comments/invariants).
    """
    if len(lines) <= window * 3:
        return "\n".join(lines)

    # Check if the last `window` lines are a repeating block
    tail = "\n".join(lines[-window:])
    count = 0
    i = len(lines) - window
    while i >= window:
        block = "\n".join(lines[i - window:i])
        if block.strip() == tail.strip():
            count += 1
            i -= window
        else:
            break

    if count >= 2:
        # Keep only up to the first repetition
        cut = len(lines) - (count * window)
        return "\n".join(lines[:cut])

    return "\n".join(lines)


def _diagnose_sany_errors(spec: str, sany_errors: str) -> str:
    """
    Analyse the spec and SANY errors to produce targeted fix instructions.

    Returns a string of hints the model can use to fix the spec, or empty string.
    """
    hints: list[str] = []

    # PlusCal mixed in
    if re.search(r"(--algorithm|--fair algorithm|begin|end algorithm)", spec, re.IGNORECASE):
        hints.append("- Remove ALL PlusCal syntax (--algorithm, begin, end algorithm, macro, procedure). Use pure TLA+.")

    # CONSTANT declared with a value (TLA+ has CONSTANT, not CONSTANT = value)
    if re.search(r"^\s*CONSTANTS?\s+\w+\s*=", spec, re.MULTILINE):
        hints.append("- CONSTANT/CONSTANTS declarations must not have '=' values. Use 'CONSTANT N' then define 'N == 5' separately or override in .cfg.")

    # vars == {...} using set braces instead of tuple <<>>
    if re.search(r"vars\s*==\s*\{", spec):
        hints.append("- 'vars' should be a tuple <<v1, v2, ...>>, not a set {v1, v2, ...}.")

    # Double prime (x'' instead of x')
    if re.search(r"\w''", spec):
        hints.append("- Use single prime (x') for next-state variables, not double prime (x'').")

    # Missing ==== closing delimiter
    if "====" not in spec:
        hints.append("- Add '====' as the last line to close the module.")

    # UNCHANGED used with wrong syntax
    if re.search(r"UNCHANGED\s+[a-zA-Z]", spec) and not re.search(r"UNCHANGED\s*<<", spec):
        hints.append("- UNCHANGED with multiple variables must use tuple syntax: UNCHANGED <<v1, v2>>.")

    # Conflicting UNCHANGED (priming a variable AND listing it in UNCHANGED)
    for m in re.finditer(r"UNCHANGED\s*<<([^>]+)>>", spec):
        unchanged_vars = {v.strip() for v in m.group(1).split(",")}
        # Check nearby lines for primed versions of same vars
        context_start = max(0, spec.rfind("\n", 0, m.start()) - 200)
        context = spec[context_start:m.start()]
        for var in unchanged_vars:
            if re.search(rf"\b{re.escape(var)}'\s*=", context):
                hints.append(f"- Variable '{var}' is both primed and listed in UNCHANGED in the same action. Remove it from UNCHANGED.")
                break

    # Repeated blocks
    lines = spec.splitlines()
    if len(lines) > 60:
        # Check for degenerate repetition
        block = "\n".join(lines[-5:])
        count = spec.count(block)
        if count > 2:
            hints.append(f"- The spec contains degenerate repetition (a block appears {count}+ times). Remove all duplicates.")

    return "\n".join(hints)


# ---------------------------------------------------------------------------
# Convenience CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a TLA+ spec via ChatTLA")
    parser.add_argument("description", help="Natural-language system description")
    parser.add_argument("--model",     default=_DEFAULT_MODEL)
    parser.add_argument("--reasoning", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--validate",  action="store_true", help="Run TLC validation after generation")
    args = parser.parse_args()

    client = ChatTLAClient(model=args.model, reasoning=args.reasoning)
    if args.validate:
        spec, tier = client.validate_and_generate(args.description)
        print(spec)
        print(f"\n[Validation tier: {tier}]")
    else:
        print(client.generate_spec(args.description))
