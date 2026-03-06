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

Critical TLA+ syntax rules:
- EXTENDS Integers for Int, +, -, *, \\div; EXTENDS Sequences for Seq, Append, Len, Head, Tail; EXTENDS FiniteSets for Cardinality, IsFiniteSet
- Declare ALL state variables in a VARIABLES line (every primed variable x' must appear in VARIABLES)
- Use = (not ==) inside Init and Next action conjuncts: /\\ x = value
- Function construction: [x \\in S |-> expr] (NOT [x \\in S : expr])
- Use \\in SUBSET S for set quantification (NOT \\E x \\subseteq S)
- Do NOT use PlusCal syntax (:=, --algorithm, labels, while, goto)
- TypeOK must be defined if referenced as INVARIANT
- Spec == Init /\\ [][Next]_vars where vars == <<v1, v2, ...>>
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
        self._temp_override: Optional[float] = None  # set by benchmark for multi-attempt

    def generate_spec(
        self,
        nl_description: str,
        module_name: Optional[str] = None,
        temperature: float = 0.05,
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

        temp = self._temp_override if self._temp_override is not None else temperature
        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            raw=True,
            options={
                "temperature": temp,
                "repeat_penalty": 1.3,
                "num_predict": 4096,
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
        spec = _extract_tla(raw)
        return _sanitize_spec(spec)

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

        # Track the best spec/tier seen across all attempts to prevent regressions.
        # Tier ordering: gold > silver > bronze
        _TIER_RANK = {"gold": 3, "silver": 2, "bronze": 1}
        best_spec = spec
        best_tier = "bronze"

        def _update_best(candidate_spec: str, candidate_tier: str):
            nonlocal best_spec, best_tier
            if _TIER_RANK.get(candidate_tier, 0) > _TIER_RANK.get(best_tier, 0):
                best_spec = candidate_spec
                best_tier = candidate_tier

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
            _update_best(spec, result.tier)

            if result.tier == "gold":
                return spec, "gold"
            if result.tier == "silver":
                # Silver = SANY passed. Don't risk regressing with more retries
                # unless we still have attempts left. Keep it as the best and
                # try one more self-correct for gold, but never return worse.
                return spec, "silver"

            # Bronze with TLC errors: feed TLC violations back
            error_summary = "\n".join(result.tlc_violations[:5])
            spec = self._self_correct(spec, error_summary)
            spec = _sanitize_spec(spec)

        # Final validation after all retries
        m = re.search(r"----\s*MODULE\s+(\w+)", spec)
        module_name = m.group(1) if m else "Generated"
        result = validate_string(spec, module_name=module_name)
        _update_best(spec, result.tier)

        # Return the best spec seen across all attempts
        return best_spec, best_tier

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
                "temperature": 0.05,
                "repeat_penalty": 1.3,
                "num_predict": 4096,
                "stop": ["<|end|>", "<|start|>", "\n===="],
            },
        )
        raw = "---- MODULE" + response["response"]
        if "====" not in raw:
            raw += "\n===="
        return _sanitize_spec(_extract_tla(raw))

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
                "temperature": 0.05,
                "repeat_penalty": 1.3,
                "num_predict": 4096,
                "stop": ["<|end|>", "<|start|>", "\n===="],
            },
        )
        raw = "---- MODULE" + response["response"]
        if "====" not in raw:
            raw += "\n===="
        return _sanitize_spec(_extract_tla(raw))


def _extract_tla(text: str) -> str:
    """Extract ---- MODULE ... ==== block from model output."""
    m = re.search(r"(----\s*MODULE\b.*?====)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _sanitize_spec(spec: str) -> str:
    """
    Apply rule-based fixes for common generation artefacts that cause SANY failures.
    """
    # ------------------------------------------------------------------
    # 1.  PlusCal removal (must happen first — PlusCal can confuse later regexes)
    # ------------------------------------------------------------------

    # Remove full PlusCal blocks: (* --algorithm ... end algorithm; *)
    spec = re.sub(
        r"\(\*\s*--(?:fair\s+)?algorithm\b.*?end\s+algorithm\s*;?\s*\*\)",
        "",
        spec,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove full PlusCal --spec blocks: (* --spec Name ... end spec *)
    spec = re.sub(
        r"\(\*\s*--spec\b.*?end\s+spec\s*;?\s*\*\)",
        "",
        spec,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove single-line PlusCal comment: (* --algorithm Name *)
    spec = re.sub(
        r"\(\*\s*--(?:fair\s+)?algorithm\b[^)]*\*\)",
        "",
        spec,
        flags=re.IGNORECASE,
    )

    # Remove (* --spec Name *) single-line comment
    spec = re.sub(
        r"\(\*\s*--(?:spec|operator)\b[^)]*\*\)",
        "",
        spec,
        flags=re.IGNORECASE,
    )

    # Remove duplicate VARIABLES declarations (keep the first one)
    def _dedup_variables(spec_text: str) -> str:
        lines = spec_text.split('\n')
        seen_vars = False
        result = []
        for line in lines:
            if re.match(r'^VARIABLES?\s+\w', line):
                if seen_vars:
                    continue  # skip duplicate
                seen_vars = True
            result.append(line)
        return '\n'.join(result)
    spec = _dedup_variables(spec)

    # Remove PlusCal 'define ... end define' blocks
    spec = re.sub(
        r"^define\b.*?^end\s+define\b[^\n]*",
        "",
        spec,
        flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )

    # Remove standalone PlusCal keywords
    spec = re.sub(
        r"^\s*(begin|end\s+algorithm|end\s+define|macro|procedure|process|do\b|end\s+process|end\s+while|end\s+if|end\s+do|await|goto|call|return|with\b[^=]*\bdo)\b.*$",
        "",
        spec,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Remove PlusCal-style lowercase 'variables' blocks (NOT uppercase VARIABLES)
    spec = re.sub(
        r"^variables\s*\n(?:\s+\w+[^\n]*\n)+",
        "",
        spec,
        flags=re.MULTILINE,
    )

    # Remove PlusCal assignment statements (x := y  or  x := y;)
    spec = re.sub(r"^\s+\w+\[?[^\n]*\]?\s*:=\s*[^\n]*$", "", spec, flags=re.MULTILINE)
    # Remove WHILE...DO, END; blocks (case-insensitive)
    spec = re.sub(r"^\s*while\b[^\n]*\bdo\s*$", "", spec, flags=re.MULTILINE | re.IGNORECASE)
    spec = re.sub(r"^\s*end\s*;\s*$", "", spec, flags=re.MULTILINE | re.IGNORECASE)
    # Remove PlusCal 'skip', 'print', 'assert' standalone lines
    spec = re.sub(r"^\s*skip\s*;?\s*$", "", spec, flags=re.MULTILINE)
    spec = re.sub(r"^\s*print\b[^\n]*;\s*$", "", spec, flags=re.MULTILINE)
    # Remove if/then/else/elsif/end if PlusCal blocks
    spec = re.sub(r"^\s*if\b[^\n]*\bthen\s*$", "", spec, flags=re.MULTILINE | re.IGNORECASE)
    spec = re.sub(r"^\s*(else|elsif)\b[^\n]*$", "", spec, flags=re.MULTILINE | re.IGNORECASE)
    spec = re.sub(r"^\s*end\s+(if|do|while|process|algorithm|define|macro|procedure)\s*;?\s*$", "", spec, flags=re.MULTILINE | re.IGNORECASE)

    # ------------------------------------------------------------------
    # 2.  Fix CONSTANT/CONSTANTS declarations
    # ------------------------------------------------------------------

    # Auto-add Integers to EXTENDS if Int, \div, or unary minus is used
    # Unary minus patterns: {-1}, |-> -1, = -1, .. -1, etc.
    needs_integers = bool(
        re.search(r'\bInt\b|\\div\b', spec)
        or re.search(r'[{,|>\s=]\s*-\d', spec)  # unary minus before a digit
        or re.search(r'\.\.\s*-\d', spec)          # range with negative end
    )
    if needs_integers and not re.search(r'EXTENDS\s+[^\n]*Integers', spec):
        spec = re.sub(r'(EXTENDS\s+[^\n]*)', r'\1, Integers', spec, count=1)

    # Fix standalone / used as integer division: replace with \div
    # Only match " / " (space-slash-space) not part of /\ or \/ or comments
    spec = re.sub(r'(?<=[)\w]) / (?=[(\w])', r' \\div ', spec)

    # Fix RANGE usage: define Range(f) if RANGE is used as an operator
    if re.search(r'\bRANGE\b', spec) and not re.search(r'Range\s*\(', spec):
        # Replace standalone RANGE with inline definition
        def _replace_range(m):
            var = m.group(1)
            return '{(' + var + ')[x] : x \\in DOMAIN ' + var + '}'
        spec = re.sub(r'\bRANGE\s+(\w+)', _replace_range, spec)

    # Fix: CONSTANT N \in Nat  ->  CONSTANT N
    spec = re.sub(
        r"^(CONSTANTS?\s+\w+)\s*\\in\b[^\n]*$",
        r"\1",
        spec,
        flags=re.MULTILINE,
    )

    # Fix multi-line CONSTANTS that accidentally swallowed VARIABLES.
    # Pattern: "CONSTANTS N, Participants, Coordinator, VARIABLES, state, ..."
    # Split at the word VARIABLES: put everything before as CONSTANTS, after as VARIABLES
    def _split_constants_variables(m: re.Match) -> str:
        full = m.group(0)
        # Find VARIABLES keyword inside the list
        parts = re.split(r",\s*VARIABLES?\s*,?", full, maxsplit=1)
        if len(parts) == 2:
            const_part = parts[0].strip().rstrip(",")
            var_part = parts[1].strip().lstrip(",").strip()
            result = const_part + "\n"
            if var_part:
                # Extract variable names
                var_names = [n.strip() for n in var_part.split(",") if n.strip() and re.match(r"^\w+$", n.strip())]
                if var_names:
                    result += f"VARIABLES {', '.join(var_names)}\n"
            return result
        return full

    spec = re.sub(
        r"^CONSTANTS?\s+[^\n]*VARIABLES?\b[^\n]*$",
        _split_constants_variables,
        spec,
        flags=re.MULTILINE,
    )

    # Fix multi-line CONSTANTS block (indented names on separate lines)
    # Stop before lines that have VARIABLES, a definition (==), or non-indented content
    def _fix_constants_block(m: re.Match) -> str:
        keyword = m.group(1)
        body = m.group(2)
        names = []
        for line in body.splitlines():
            stripped = line.strip().rstrip(",")
            if not stripped or stripped.startswith("\\*") or stripped.startswith("(*"):
                continue
            # Stop if we hit VARIABLES or a definition
            if re.match(r"VARIABLES?|VARIABLE\b|\w+\s*==", stripped):
                break
            name_match = re.match(r"(\w+)", stripped)
            if name_match:
                name = name_match.group(1)
                # Don't include TLA+ keywords as constant names
                if name not in ("VARIABLES", "VARIABLE", "EXTENDS", "ASSUME", "THEOREM", "LOCAL"):
                    names.append(name)
        if names:
            return f"{keyword} {', '.join(names)}\n"
        return m.group(0)

    spec = re.sub(
        r"^(CONSTANTS?)\s*\n((?:\s+\w+[^\n]*\n)+)",
        _fix_constants_block,
        spec,
        flags=re.MULTILINE,
    )

    # ------------------------------------------------------------------
    # 3.  Fix VARIABLES declarations
    # ------------------------------------------------------------------

    # Auto-detect and add missing VARIABLES declaration
    # Any identifier used with prime (x') MUST be a state variable
    def _ensure_variables_declared(spec_text: str) -> str:
        # Find all primed variable names
        primed_vars = set(re.findall(r"\b(\w+)'", spec_text))
        # Also check Init == body for x = ... patterns (state variable initialization)
        # The Init body ends at the next operator definition (with or without params)
        init_match = re.search(r"^Init\s*==\s*(.*?)(?=^\w+(?:\([^)]*\))?\s*==|\Z)", spec_text, re.MULTILINE | re.DOTALL)
        if init_match:
            init_body = init_match.group(1)
            # Match both /\ x = value and /\ x \in Set patterns
            init_vars = set(re.findall(r"/\\\s*(\w+)\s*(?:=|\\in)\s*", init_body))
            primed_vars |= init_vars
        
        # Remove known non-variable names
        non_vars = {'TRUE', 'FALSE', 'BOOLEAN', 'Nat', 'Int', 'STRING', 'SUBSET',
                     'UNION', 'DOMAIN', 'ENABLED', 'UNCHANGED', 'EXCEPT', 'IF',
                     'THEN', 'ELSE', 'LET', 'IN', 'CASE', 'OTHER', 'CHOOSE', 'WITH'}
        primed_vars -= non_vars
        
        if not primed_vars:
            return spec_text
        
        # Find existing VARIABLES declaration
        var_match = re.search(r"^VARIABLES?\s+(.+?)$", spec_text, re.MULTILINE)
        if var_match:
            declared = {v.strip() for v in var_match.group(1).split(',')}
            missing = primed_vars - declared
            if missing:
                # Add missing variables to existing declaration
                new_decl = var_match.group(0).rstrip() + ', ' + ', '.join(sorted(missing))
                spec_text = spec_text[:var_match.start()] + new_decl + spec_text[var_match.end():]
        else:
            # No VARIABLES declaration at all — add one after EXTENDS or CONSTANTS
            insert_after = None
            # Try after CONSTANTS
            const_match = re.search(r"^CONSTANTS?\s+.*$", spec_text, re.MULTILINE)
            if const_match:
                insert_after = const_match.end()
            else:
                # Try after EXTENDS
                ext_match = re.search(r"^EXTENDS\s+.*$", spec_text, re.MULTILINE)
                if ext_match:
                    insert_after = ext_match.end()
                else:
                    # Try after MODULE header
                    mod_match = re.search(r"^----\s*MODULE.*----\s*$", spec_text, re.MULTILINE)
                    if mod_match:
                        insert_after = mod_match.end()
            
            if insert_after is not None:
                var_line = f"\nVARIABLES {', '.join(sorted(primed_vars))}\n"
                spec_text = spec_text[:insert_after] + var_line + spec_text[insert_after:]
        
        return spec_text
    
    spec = _ensure_variables_declared(spec)

    # Fix multi-line VARIABLES with \in type annotations and/or trailing comments
    def _fix_variables_block(m: re.Match) -> str:
        keyword = m.group(1)  # VARIABLES or VARIABLE
        body = m.group(2)
        var_names = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Stop if we hit a non-variable line (definition, comment-only, empty)
            if re.match(r"(Init|Next|Spec|TypeOK|\w+)\s*==", stripped):
                break
            if stripped.startswith("(*") or stripped.startswith("\\*"):
                continue
            # Remove trailing comment: "subscribers,  \* Mapping..." -> "subscribers"
            stripped = re.sub(r"\s*\\?\*.*$", "", stripped)
            stripped = stripped.strip().rstrip(",").strip()
            if not stripped:
                continue
            # "state \in [1..N -> ...]" -> "state"
            var_match = re.match(r"(\w+)\s*\\in\b", stripped)
            if var_match:
                var_names.append(var_match.group(1))
            elif re.match(r"^\w+$", stripped):
                var_names.append(stripped)
        if var_names:
            return f"{keyword} {', '.join(var_names)}\n"
        return m.group(0)

    spec = re.sub(
        r"^(VARIABLES?)\s*\n((?:\s+[^\n]+\n)+)",
        _fix_variables_block,
        spec,
        flags=re.MULTILINE,
    )

    # Fix single-line: VARIABLES state \in [1..N -> ...]
    spec = re.sub(
        r"^(VARIABLES?\s+)(\w+)\s*\\in\s*\[[^\]]*\]\s*$",
        r"\1\2",
        spec,
        flags=re.MULTILINE,
    )

    # ------------------------------------------------------------------
    # 4.  Operator fixes
    # ------------------------------------------------------------------

    # \neq -> # (SANY doesn't know \neq)
    spec = re.sub(r"\\neq\b", "#", spec)

    # Fix double-escaped LaTeX operators
    spec = spec.replace("\\\\notin", "\\notin")
    spec = spec.replace("\\\\in", "\\in")
    spec = spec.replace("\\\\cup", "\\cup")
    spec = spec.replace("\\\\cap", "\\cap")
    spec = spec.replace("\\\\subseteq", "\\subseteq")

    # Fix RandomOffset() -> RandomOffset (TLA+ 0-arg operators have no parens)
    spec = re.sub(r"(\b[A-Z]\w+)\(\)", r"\1", spec)

    # Fix 'bmod' -> '%' (not a TLA+ operator)
    spec = re.sub(r"\bbmod\b", "%", spec)

    # Fix 'Mod' -> '%' (unknown operator in TLA+, use modular arithmetic)
    spec = re.sub(r"\bMod\b", "%", spec)

    # Fix escaped percent '\%' -> '%' (\% crashes Java's String.format in SANY)
    spec = spec.replace("\\%", "%")

    # Fix primed indexed variables: v[i]' -> v'[i]
    # In TLA+, v[i]' is ambiguous; standard form is v'[i] (prime the variable, then index)
    spec = re.sub(r"(\b\w+)(\[[^\]]+\])'", r"\1'\2", spec)

    # Fix existential/universal with \subseteq: \E x \subseteq S : -> \E x \in SUBSET S :
    spec = re.sub(r"(\\[EA])\s+(\w+)\s*\\subseteq\s+", r"\1 \2 \\in SUBSET ", spec)

    # Expand comma-separated quantifier bindings to nested quantifiers:
    # \E x \in S, y \in T : P  -> \E x \in S : \E y \in T : P
    # This is needed when the second binding depends on the first (e.g. y \in f[x])
    def _expand_quantifier_bindings(m: re.Match) -> str:
        quant = m.group(1)  # \E or \A
        bindings_str = m.group(2)  # "x \in S, y \in T"
        body_colon = ":"
        # Split on comma followed by a word and \in
        parts = re.split(r',\s*(?=\w+\s*\\in\b)', bindings_str)
        if len(parts) <= 1:
            return m.group(0)  # No expansion needed
        result = ""
        for i, part in enumerate(parts):
            part = part.strip()
            if i < len(parts) - 1:
                result += f"{quant} {part} : "
            else:
                result += f"{quant} {part} :"
        return result
    spec = re.sub(r"(\\[EA])\s+((?:\w+\s*\\in\s*[^:,]+,\s*)+\w+\s*\\in\s*[^:]+?)\s*:", _expand_quantifier_bindings, spec)

    # Fix function construction [x \in S : expr] -> [x \in S |-> expr]
    # Only when \in appears before : inside brackets (distinguishes from record set syntax)
    spec = re.sub(r"\[(\w+)\s*\\in\s*([^\]]*?)\s*:\s*(?!:)", r"[\1 \\in \2 |-> ", spec)

    # Fix == used as = inside conjunction lines (/\ x == value -> /\ x = value)
    # In TLA+, == starts a definition, = is equality. Inside conjunctions, == is almost always wrong.
    def _fix_eq_in_conj(m: re.Match) -> str:
        line = m.group(0)
        # Replace the first == on this line with = (skip the leading /\)
        return re.sub(r'==', '=', line, count=1)
    spec = re.sub(r"^\s*[/\\][/\\]\s+\w+[^\n]*==.*$", _fix_eq_in_conj, spec, flags=re.MULTILINE)

    # Fix Seq(X, Y) -> Seq(X) (Seq takes one argument in TLA+)
    spec = re.sub(r"\bSeq\(([^,)]+),[^)]+\)", r"Seq(\1)", spec)

    # Fix Seq[X] -> Seq(X) (bracket notation not valid in TLA+)
    spec = re.sub(r"\bSeq\[([^\]]+)\]", r"Seq(\1)", spec)

    # Fix [a, b] used as range -> a..b (common with clock offsets etc)
    spec = re.sub(r"\[(-?\w+),\s*(-?\w+)\]", r"\1..\2", spec)

    # Fix '\subset' -> '\subseteq' (\subset not in TLA+)
    spec = re.sub(r"\\subset\b(?!eq)", r"\\subseteq", spec)

    # Fix '\sum' and '\max' and '\min' (not standard TLA+, replace with comments)
    spec = re.sub(r"\\sum\b", "Sum", spec)
    spec = re.sub(r"\\max\b", "Max", spec)
    spec = re.sub(r"\\min\b", "Min", spec)

    # Fix VARIABLES with inline initialization: VARIABLES flag = [...], turn = 1
    def _fix_var_init(m: re.Match) -> str:
        keyword = m.group(1)
        body = m.group(2)
        # Extract variable names before = signs
        names = []
        for part in body.split(','):
            part = part.strip()
            eq_idx = part.find('=')
            if eq_idx > 0 and part[eq_idx-1:eq_idx+1] != '==':
                name = part[:eq_idx].strip()
            else:
                name = part.strip()
            name_m = re.match(r'(\w+)', name)
            if name_m:
                names.append(name_m.group(1))
        if names:
            return f"{keyword} {', '.join(names)}"
        return m.group(0)

    spec = re.sub(
        r"^(VARIABLES?)\s+(.+?=.+)$",
        _fix_var_init,
        spec,
        flags=re.MULTILINE,
    )

    # Fix 'Any' type (not valid TLA+) -> STRING or remove
    spec = re.sub(r"\bAny\b", "STRING", spec)

    # Strip ENDIF — not valid TLA+. IF/THEN/ELSE doesn't use ENDIF.
    spec = re.sub(r"^\s*ENDIF\s*$", "", spec, flags=re.MULTILINE)

    # Fix LET x = expr IN -> LET x == expr IN  (TLA+ LET uses == not =)
    spec = re.sub(r"\bLET\s+(\w+)\s*=\s*(?!=)", r"LET \1 == ", spec)

    # Fix incomplete quantifiers: \E x \in S (no : body) -> \E x \in S :
    # When the quantifier ends at end-of-line without a colon
    spec = re.sub(r"(\\[EA]\s+\w+\s*\\in\s*\w+)\s*$", r"\1 :", spec, flags=re.MULTILINE)

    # Collapse double conjunction:  /\ (* comment *) /\  ->  /\
    # This happens when a TLA+ comment sits between two /\ operators.
    spec = re.sub(r"/\\\s*\(\*[^)]*\*\)\s*/\\", r"/\\", spec)
    # Also collapse plain  /\  /\  (whitespace only between)
    spec = re.sub(r"/\\\s+/\\", r"/\\", spec)

    # Fix lambda-like |-> used as predicate argument in Filter/Select calls.
    # Filter(seq, t |-> expr) -> SelectSeq(seq, LAMBDA t : expr)
    spec = re.sub(r"\bFilter\(", "SelectSeq(", spec)
    # Only replace Select( when it's NOT SelectSeq( already
    spec = re.sub(r"\bSelect\((?!Seq)", "SelectSeq(", spec)
    # Convert t |-> expr to LAMBDA t : expr ONLY inside SelectSeq calls.
    # Also handle t' |-> expr (primed lambda variable) by stripping the prime.
    def _fix_selectseq_lambda(m: re.Match) -> str:
        before = m.group(1)  # "SelectSeq(seq, "
        var = m.group(2)     # might include trailing '
        body = m.group(3)
        # Strip prime from lambda variable name
        var = var.rstrip("'")
        # Strip primes from variable references in body that match the lambda var
        body = body.replace(f"{var}'.", f"{var}.")
        body = body.replace(f"{var}'[", f"{var}[")
        return f"{before}LAMBDA {var} : {body}"
    spec = re.sub(
        r"(SelectSeq\([^,]+,\s*)(\w+'?)\s*\|->\s*([^)]+)",
        _fix_selectseq_lambda,
        spec,
    )

    # Auto-define Max(S) and Min(S) when used but not defined
    spec = _auto_define_max_min(spec)

    # Auto-detect undeclared set-like identifiers and add as CONSTANTS.
    # If a capitalized name is used in \in Foo, [x \in Foo |->], or Foo ->
    # but is not declared as VARIABLE, CONSTANT, or defined with ==,
    # it should be a CONSTANT.
    spec = _auto_add_constants(spec)

    # ------------------------------------------------------------------
    # 5.  Structural cleanup
    # ------------------------------------------------------------------

    # Remove markdown code fences
    spec = re.sub(r"^```\w*\s*$", "", spec, flags=re.MULTILINE)

    # Truncate degenerate repetition
    lines = spec.splitlines()
    if len(lines) > 30:
        spec = _dedup_repeated_blocks(lines)

    # Close unclosed comments
    open_comments = len(re.findall(r"\(\*", spec)) - len(re.findall(r"\*\)", spec))
    if open_comments > 0:
        for _ in range(open_comments):
            last_open = spec.rfind("(*")
            if last_open >= 0:
                close_pos = spec.find("*)", last_open)
                if close_pos < 0:
                    eol = spec.find("\n", last_open)
                    if eol >= 0:
                        spec = spec[:eol] + " *)" + spec[eol:]
                    else:
                        spec += " *)"

    # Truncate incomplete trailing lines (no ==== at end)
    if "====" not in spec:
        lines = spec.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if not line:
                continue
            if re.match(r"^(Init|Next|Spec|TypeOK|\w+)\s*==", line) or line.endswith(">>") or line == "":
                break
        spec = "\n".join(lines[:i+1]) + "\n\n===="

    # Strip trailing noise after ====
    m = re.search(r"(----\s*MODULE\b.*?====)", spec, re.DOTALL)
    if m:
        spec = m.group(1)

    # Remove blank lines that contain only whitespace
    spec = re.sub(r"\n[ \t]+\n", "\n\n", spec)

    return spec.strip()


def _auto_define_max_min(spec: str) -> str:
    """
    If Max or Min is used as an operator (Max(...) or Min(...)) but not defined
    with ==, inject a definition after the EXTENDS line.

    Max(S) == CHOOSE x \in S : \A y \in S : x >= y
    Then convert Max([i \in D |-> expr]) to Max({expr : i \in D}).
    """
    need_max = bool(re.search(r'\bMax\s*\(', spec)) and not re.search(r'\bMax\s*(?:\([^)]*\))?\s*==', spec)
    need_min = bool(re.search(r'\bMin\s*\(', spec)) and not re.search(r'\bMin\s*(?:\([^)]*\))?\s*==', spec)

    if not need_max and not need_min:
        return spec

    defs = []
    if need_max:
        defs.append("Max(S) == CHOOSE x \\in S : \\A y \\in S : x >= y")
    if need_min:
        defs.append("Min(S) == CHOOSE x \\in S : \\A y \\in S : x =< y")

    # Insert after EXTENDS line (or after CONSTANTS, or after MODULE header)
    insert_block = "\n" + "\n".join(defs) + "\n"

    ext_match = re.search(r'^EXTENDS\s+.*$', spec, re.MULTILINE)
    const_match = re.search(r'^CONSTANTS?\s+.*$', spec, re.MULTILINE)
    var_match = re.search(r'^VARIABLES?\s+', spec, re.MULTILINE)

    if ext_match:
        pos = ext_match.end()
    elif const_match:
        pos = const_match.end()
    elif var_match:
        pos = var_match.start()
        insert_block = "\n".join(defs) + "\n\n"
    else:
        mod_match = re.search(r'^----\s*MODULE.*----\s*$', spec, re.MULTILINE)
        pos = mod_match.end() if mod_match else 0

    spec = spec[:pos] + insert_block + spec[pos:]

    # Convert Max([i \in D |-> expr]) to Max({expr : i \in D})
    # Uses bracket-counting to handle nested brackets in expr.
    def _convert_max_func_to_set(spec_text: str) -> str:
        for op in ('Max', 'Min'):
            pattern = re.compile(rf'\b{op}\(\s*\[')
            offset = 0
            while True:
                m = pattern.search(spec_text, offset)
                if not m:
                    break
                # Find the start of [ inside Max(
                bracket_start = spec_text.index('[', m.start())
                # Parse: var \in domain |-> expr
                inner_start = bracket_start + 1
                # Find |-> while counting brackets
                depth = 1
                i = inner_start
                arrow_pos = -1
                while i < len(spec_text) and depth > 0:
                    if spec_text[i] == '[':
                        depth += 1
                    elif spec_text[i] == ']':
                        depth -= 1
                        if depth == 0:
                            break
                    elif spec_text[i:i+3] == '|->' and depth == 1 and arrow_pos == -1:
                        arrow_pos = i
                    i += 1
                if depth != 0 or arrow_pos == -1:
                    offset = m.end()
                    continue
                closing_bracket = i
                # Extract var \in domain  and  expr
                binding = spec_text[inner_start:arrow_pos].strip()
                expr = spec_text[arrow_pos+3:closing_bracket].strip()
                # Expect closing ) after ]
                rest = spec_text[closing_bracket+1:].lstrip()
                if not rest.startswith(')'):
                    offset = m.end()
                    continue
                paren_pos = spec_text.index(')', closing_bracket+1)
                # Build replacement: Max({expr : var \in domain})
                replacement = f"{op}({{{expr} : {binding}}})"
                spec_text = spec_text[:m.start()] + replacement + spec_text[paren_pos+1:]
                offset = m.start() + len(replacement)
        return spec_text

    spec = _convert_max_func_to_set(spec)

    return spec


def _auto_add_constants(spec: str) -> str:
    """
    Detect capitalized identifiers used as sets (in \\in X, [x \\in X |->], etc.)
    that are not declared anywhere, and add them as CONSTANTS.
    """
    # Collect already-declared names
    declared: set[str] = set()
    # CONSTANTS
    for m in re.finditer(r'^CONSTANTS?\s+(.+)$', spec, re.MULTILINE):
        declared |= {n.strip() for n in m.group(1).split(',')}
    # VARIABLES
    for m in re.finditer(r'^VARIABLES?\s+(.+)$', spec, re.MULTILINE):
        declared |= {n.strip() for n in m.group(1).split(',')}
    # Operator definitions (name ==) and their parameters
    for m in re.finditer(r'^(\w+)\s*(?:\(([^)]*)\))?\s*==', spec, re.MULTILINE):
        declared.add(m.group(1))
        if m.group(2):
            # Add parameter names too (e.g., Max(S) -> S is a parameter, not a constant)
            for param in m.group(2).split(','):
                declared.add(param.strip())
    # Standard library names and built-in sets
    builtins = {
        'Nat', 'Int', 'Real', 'STRING', 'BOOLEAN', 'TRUE', 'FALSE',
        'SUBSET', 'UNION', 'DOMAIN', 'ENABLED', 'UNCHANGED', 'EXCEPT',
        'IF', 'THEN', 'ELSE', 'LET', 'IN', 'CASE', 'OTHER', 'CHOOSE',
        'WITH', 'Seq', 'Append', 'Len', 'Head', 'Tail', 'SubSeq',
        'SelectSeq', 'Cardinality', 'IsFiniteSet', 'EXTENDS',
        'VARIABLES', 'VARIABLE', 'CONSTANTS', 'CONSTANT', 'ASSUME',
        'THEOREM', 'LOCAL', 'INSTANCE', 'MODULE', 'Sequences',
        'FiniteSets', 'Integers', 'Naturals', 'Reals', 'TLC', 'Bags',
    }
    declared |= builtins

    # Find capitalized identifiers used as sets
    # Patterns: \in Foo, [x \in Foo |->], Foo ->, SUBSET Foo, |-> Foo]
    candidates: set[str] = set()
    for m in re.finditer(r'\\in\s+([A-Z]\w*)', spec):
        candidates.add(m.group(1))
    for m in re.finditer(r'([A-Z]\w*)\s*->', spec):
        candidates.add(m.group(1))
    for m in re.finditer(r'SUBSET\s+([A-Z]\w*)', spec):
        candidates.add(m.group(1))
    # Also: \E x \in Foo, \A x \in Foo
    for m in re.finditer(r'\\[EA]\s+\w+\s*\\in\s+([A-Z]\w*)', spec):
        candidates.add(m.group(1))
    # x \cup Foo, x \cap Foo
    for m in re.finditer(r'\\(?:cup|cap)\s+([A-Z]\w*)', spec):
        candidates.add(m.group(1))
    # |-> Foo] (function type range)
    for m in re.finditer(r'\|->\s+([A-Z]\w*)\s*\]', spec):
        candidates.add(m.group(1))

    # Filter: only keep names not already declared
    missing = sorted(candidates - declared)
    if not missing:
        return spec

    # Insert CONSTANTS declaration (or extend existing one)
    const_match = re.search(r'^(CONSTANTS?)\s+(.+)$', spec, re.MULTILINE)
    if const_match:
        # Extend existing CONSTANTS line
        existing = const_match.group(2).rstrip()
        new_line = f"{const_match.group(1)} {existing}, {', '.join(missing)}"
        spec = spec[:const_match.start()] + new_line + spec[const_match.end():]
    else:
        # Insert new CONSTANTS after EXTENDS (or before VARIABLES)
        ext_match = re.search(r'^EXTENDS\s+.*$', spec, re.MULTILINE)
        var_match = re.search(r'^VARIABLES?\s+', spec, re.MULTILINE)
        if ext_match:
            insert_at = ext_match.end()
            spec = spec[:insert_at] + f"\nCONSTANTS {', '.join(missing)}" + spec[insert_at:]
        elif var_match:
            insert_at = var_match.start()
            spec = spec[:insert_at] + f"CONSTANTS {', '.join(missing)}\n" + spec[insert_at:]

    return spec


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
