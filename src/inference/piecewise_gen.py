"""
piecewise_gen.py — Piece-wise verified TLA+ generation.

Inspired by LMGPA (Zhou & Tripakis 2024): instead of generating an entire
.tla file in one shot, decompose into 5 sequential pieces, validate each
piece against TLC/SANY, retry failures locally, and assemble the final spec.

Pipeline:
    1. VARIABLES   — what state variables exist
    2. TypeOK      — finite-set type invariant
    3. Init        — concrete initial assignment, must satisfy TypeOK
    4. Next        — disjunction of actions, each with full UNCHANGED
    5. Invariants  — additional safety properties (optional)

Each piece is validated incrementally by SANY (and TLC where applicable).
Failed pieces are retried with structured error feedback (up to 3 rounds).

Usage
-----
    from src.inference.piecewise_gen import generate_piecewise

    spec = generate_piecewise(
        problem_id="BM001",
        nl_description="Mutual exclusion algorithm for N processes...",
        module_name="MutualExclusion",
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Piece-wise prompts — each piece gets a focused, narrow task
# ─────────────────────────────────────────────────────────────────────────────

_VARIABLES_PROMPT = """\
You are designing a TLA+ specification for the following system:

{nl_description}

Task: List the state variables this system needs.

Output ONLY a single line in this exact format:
VARIABLES var1, var2, var3

Rules:
- Use lowercase variable names.
- Pick the minimum set of variables needed to describe the system's state.
- Do NOT define what the variables mean — just list them.
- Do NOT output anything except the VARIABLES line.
"""

_TYPEOK_PROMPT = """\
You are designing a TLA+ specification for the following system:

{nl_description}

The state variables are: {variables}

Task: Define a TypeOK invariant that constrains every variable to a finite set.

Output ONLY the TypeOK definition in this exact format:
TypeOK ==
  /\\ var1 \\in <finite_set>
  /\\ var2 \\in <finite_set>
  ...

Rules:
- Every variable from VARIABLES must appear in TypeOK.
- Use bounded ranges like 0..N, finite explicit sets like {{"a", "b", "c"}}, or [1..N -> {{"x", "y"}}] for functions.
- NEVER use Nat or Int unbounded — TLC cannot enumerate them.
- If you need a CONSTANT, write "CONSTANT N" on a separate line BEFORE TypeOK.
- Do NOT output anything except the optional CONSTANT line(s) and the TypeOK definition.
"""

_INIT_PROMPT = """\
You are designing a TLA+ specification for the following system:

{nl_description}

The state variables are: {variables}
The TypeOK invariant is:
{typeok}

Task: Define Init — the initial state predicate.

Output ONLY the Init definition in this exact format:
Init ==
  /\\ var1 = <concrete_value>
  /\\ var2 = <concrete_value>
  ...

Rules:
- Every variable must get a concrete starting value.
- The values must satisfy TypeOK.
- Use = (not ==) for variable assignments inside Init.
- For function-typed variables, use [x \\in S |-> default_value].
- Do NOT output anything except the Init definition.
"""

_NEXT_PROMPT = """\
You are designing a TLA+ specification for the following system:

{nl_description}

The state variables are: {variables}
The TypeOK invariant is:
{typeok}
The Init predicate is:
{init}

Task: Define Next — the transition relation as a disjunction of actions.

Output ONLY the Next definition and any helper action definitions:

Action1 ==
  /\\ <preconditions on current state>
  /\\ var1' = <new value>
  /\\ var2' = <new value>
  /\\ UNCHANGED <<var3>>

Action2 ==
  /\\ ...

Next == Action1 \\/ Action2 \\/ ...

Rules:
- EVERY action must specify ALL variables: either prime them (var') or list them in UNCHANGED <<...>>.
- Never prime a variable AND list it in UNCHANGED in the same action — that's a contradiction.
- If the system can terminate, add: Terminating == /\\ <termination_condition> /\\ UNCHANGED <<all, vars>>
- Use \\/ (disjunction) to combine actions in Next.
- Do NOT redefine variables. Use the ones from VARIABLES.
- Do NOT output anything except action definitions and the Next line.
"""

_INVARIANTS_PROMPT = """\
You are designing a TLA+ specification for the following system:

{nl_description}

The state variables are: {variables}
The TypeOK invariant is:
{typeok}
The Init predicate is:
{init}
The Next relation is:
{next_def}

Task: Define one safety invariant that captures a non-trivial property of this system.
The invariant must constrain behavior — it should be FALSE if you weakened the spec
(e.g., removed a precondition).

Output ONLY the invariant definition in this format:
SafetyInvariant ==
  <your invariant predicate here>

Rules:
- The invariant must be a meaningful property, not just TypeOK restated.
- Examples: mutual exclusion (at most one process in CS), conservation (sum is constant),
  bounded queue (Len(q) <= K), etc.
- Use \\A, \\E, \\in, =, #, /\\, \\/, => for the predicate body.
- Do NOT output anything except the SafetyInvariant definition.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PieceResult:
    """Result of generating one piece of the spec."""
    name: str           # "variables", "typeok", "init", "next", "invariant"
    text: str           # the generated TLA+ snippet
    valid: bool         # passed validation
    attempts: int       # number of attempts taken
    errors: list[str] = field(default_factory=list)  # validator errors if invalid


@dataclass
class PiecewiseResult:
    """Final result of piece-wise generation."""
    problem_id: str
    spec: str                          # full assembled .tla module text
    pieces: list[PieceResult]          # per-piece results
    final_tier: str                    # "gold" | "silver" | "bronze"
    total_attempts: int                # sum of all piece attempts
    constants: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Piece extractors — pull just the relevant piece from model output
# ─────────────────────────────────────────────────────────────────────────────

def _extract_variables_line(text: str) -> Optional[tuple[str, list[str]]]:
    """Extract a VARIABLES line and the variable names from model output.

    Dedupes repeated names (model sometimes degenerates into 'a, a, a, a...')
    and caps at 10 variables (any spec with more is suspect).
    """
    # Strip code fences if model added them
    text = re.sub(r"```\w*\n?", "", text)
    text = re.sub(r"```", "", text)

    m = re.search(r"VARIABLES?\s+(.+?)(?:\n|$)", text)
    if not m:
        return None
    raw = m.group(1).strip()
    # Extract the variable names (split on whitespace/comma, drop empties)
    raw_names = [n.strip().rstrip(",") for n in re.split(r"[,\s]+", raw)]
    raw_names = [n for n in raw_names if n and re.match(r"^[a-zA-Z_]\w*$", n)]

    # Dedupe (preserve order). Model degeneracy: 'a, a, a, a, ...' → just 'a'
    seen = set()
    names = []
    for n in raw_names:
        if n not in seen:
            seen.add(n)
            names.append(n)

    # Sanity cap: more than 10 distinct variables is suspicious for a benchmark spec
    if len(names) > 10:
        names = names[:10]

    if not names:
        return None
    line = f"VARIABLES " + ", ".join(names)
    return line, names


def _extract_definition(text: str, name: str) -> Optional[str]:
    """Extract a definition like 'name == ...' from model output, including
    multi-line conjunctions/disjunctions."""
    text = re.sub(r"```\w*\n?", "", text)
    text = re.sub(r"```", "", text)

    lines = text.splitlines()
    # Find the line that starts the definition
    start_idx = None
    for i, line in enumerate(lines):
        if re.match(rf"^\s*{re.escape(name)}\s*==", line):
            start_idx = i
            break
    if start_idx is None:
        return None

    # The first line ALWAYS belongs to this definition.
    out_lines = [lines[start_idx]]

    for line in lines[start_idx + 1:]:
        stripped = line.strip()
        # Stop at the next top-level definition (some other Word == ...)
        # Definitions in TLA+ start with Word at column 0.
        if re.match(r"^[A-Za-z_]\w*\s*==", line):
            break
        if re.match(r"^={4,}", line):
            break
        if stripped.startswith("MODULE") or stripped.startswith("EXTENDS"):
            break
        out_lines.append(line)

    # Trim trailing blank lines
    while out_lines and not out_lines[-1].strip():
        out_lines.pop()

    return "\n".join(out_lines) if out_lines else None


def _extract_constants(text: str) -> list[str]:
    """Extract any CONSTANT declarations from model output."""
    text = re.sub(r"```\w*\n?", "", text)
    text = re.sub(r"```", "", text)
    consts = []
    for m in re.finditer(r"^\s*CONSTANTS?\s+(.+?)(?:\n|$)", text, re.MULTILINE):
        rest = m.group(1).strip()
        for part in rest.split(","):
            ident = part.strip().split(":")[0].split("(")[0].strip().rstrip(",")
            if ident and re.match(r"^[A-Za-z_]\w*$", ident):
                consts.append(ident)
    return consts


# ─────────────────────────────────────────────────────────────────────────────
# Spec assembly + validation
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_spec(
    module_name: str,
    constants: list[str],
    variables_line: str,
    typeok: str,
    init: str,
    next_def: str,
    invariant: Optional[str] = None,
) -> str:
    """Assemble verified pieces into a complete TLA+ module."""
    parts = [
        f"---- MODULE {module_name} ----",
        "EXTENDS Naturals, Integers, Sequences, FiniteSets, TLC",
        "",
    ]

    if constants:
        parts.append("CONSTANTS " + ", ".join(constants))
        parts.append("")

    parts.append(variables_line)
    parts.append(f"vars == <<{', '.join(_extract_var_names(variables_line))}>>")
    parts.append("")
    parts.append(typeok)
    parts.append("")
    parts.append(init)
    parts.append("")
    parts.append(next_def)
    parts.append("")

    if invariant:
        parts.append(invariant)
        parts.append("")

    parts.append("Spec == Init /\\ [][Next]_vars")
    parts.append("")
    parts.append("====")

    return "\n".join(parts)


def _extract_var_names(variables_line: str) -> list[str]:
    """Extract variable names from a VARIABLES line."""
    m = re.match(r"VARIABLES?\s+(.+)", variables_line.strip())
    if not m:
        return []
    return [n.strip().rstrip(",") for n in re.split(r"[,\s]+", m.group(1)) if n.strip()]


def _validate_piece_in_context(
    module_name: str,
    constants: list[str],
    variables_line: str,
    typeok: Optional[str] = None,
    init: Optional[str] = None,
    next_def: Optional[str] = None,
    invariant: Optional[str] = None,
) -> tuple[bool, list[str]]:
    """Validate a partial spec by assembling it into a stub module and running SANY.

    Builds the smallest valid module that contains the pieces we have so far
    and runs SANY on it. If only VARIABLES exist, we add a trivial Init/Next
    so SANY accepts the module.
    """
    from src.validators.sany_validator import validate_string

    # Build a complete-enough module to satisfy SANY
    var_names = _extract_var_names(variables_line)

    parts = [f"---- MODULE {module_name} ----"]
    parts.append("EXTENDS Naturals, Integers, Sequences, FiniteSets, TLC")
    parts.append("")
    if constants:
        parts.append("CONSTANTS " + ", ".join(constants))
        parts.append("")
    parts.append(variables_line)
    if var_names:
        parts.append(f"vars == <<{', '.join(var_names)}>>")
    parts.append("")

    if typeok:
        parts.append(typeok)
        parts.append("")
    elif var_names:
        # Stub TypeOK so the module parses
        parts.append("TypeOK == TRUE")
        parts.append("")

    if init:
        parts.append(init)
        parts.append("")
    elif var_names:
        # Stub Init that's always TRUE for whatever variables exist
        stub_init = "Init ==\n" + "\n".join(f"  /\\ {v} = 0" for v in var_names)
        parts.append(stub_init)
        parts.append("")

    if next_def:
        parts.append(next_def)
        parts.append("")
    elif var_names:
        # Stub Next that just preserves all variables
        parts.append(f"Next == UNCHANGED vars")
        parts.append("")

    if invariant:
        parts.append(invariant)
        parts.append("")

    parts.append("Spec == Init /\\ [][Next]_vars")
    parts.append("")
    parts.append("====")

    stub_spec = "\n".join(parts)
    result = validate_string(stub_spec, module_name=module_name)
    return result.valid, result.errors[:5]


# ─────────────────────────────────────────────────────────────────────────────
# Per-piece generation with retry
# ─────────────────────────────────────────────────────────────────────────────

def _call_model(client, prompt: str, temperature: float = 0.2, max_tokens: int = 800) -> str:
    """Single model call returning raw response text.

    Bypasses the standard generate_spec() because we don't want to seed
    '---- MODULE'. We want the model to produce structured pieces.
    """
    import ollama
    # Build a clean harmony-format prompt without the MODULE seed
    developer_content = (
        "You are an expert TLA+ formal methods engineer. "
        "Follow the user's instructions precisely. Output only what is asked, "
        "no markdown fences, no extra explanation."
    )
    harmony_prompt = (
        f"<|start|>system<|message|>You are ChatTLA, an expert at writing TLA+ specifications.<|end|>\n"
        f"<|start|>developer<|message|>{developer_content}<|end|>\n"
        f"<|start|>user<|message|>{prompt}<|end|>\n"
        f"<|start|>assistant<|channel|>final<|message|>"
    )
    response = client._client.generate(
        model=client.model,
        prompt=harmony_prompt,
        raw=True,
        options={
            "temperature": temperature,
            "repeat_penalty": 1.4,    # higher penalty to prevent token-level repetition collapse
            "repeat_last_n": 256,
            "num_predict": max_tokens,
            "top_k": 40,
            "top_p": 0.9,
            "stop": ["<|return|>", "<|end|>", "<|start|>", "\n\n\n"],
        },
    )
    return response["response"].strip()


def _gen_variables(client, nl_description: str, module_name: str, max_attempts: int = 3) -> PieceResult:
    """Generate the VARIABLES declaration. Validates by assembling a stub module."""
    prompt = _VARIABLES_PROMPT.format(nl_description=nl_description)
    last_text = ""
    last_errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            text = _call_model(client, prompt, temperature=0.1 + attempt * 0.1, max_tokens=120)
            last_text = text
            extracted = _extract_variables_line(text)
            if extracted:
                line, names = extracted
                if names:
                    # Validate that the VARIABLES line parses cleanly through SANY
                    valid, errs = _validate_piece_in_context(
                        module_name, [], line,
                    )
                    if valid:
                        return PieceResult(name="variables", text=line, valid=True, attempts=attempt)
                    last_errors = errs
        except Exception as e:
            last_text = f"(error: {e})"
            last_errors = [str(e)]
    return PieceResult(
        name="variables", text=last_text, valid=False, attempts=max_attempts,
        errors=last_errors or ["could not extract VARIABLES line with valid identifiers"],
    )


def _gen_typeok(
    client, nl_description: str, variables_line: str, module_name: str,
    max_attempts: int = 3,
) -> tuple[PieceResult, list[str]]:
    """Generate TypeOK invariant. Returns (piece, constants_extracted)."""
    prompt = _TYPEOK_PROMPT.format(nl_description=nl_description, variables=variables_line)
    last_text = ""
    last_errors = []
    constants: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            text = _call_model(client, prompt, temperature=0.1 + attempt * 0.1)
            last_text = text
            consts_here = _extract_constants(text)
            typeok = _extract_definition(text, "TypeOK")
            if typeok:
                # Validate by assembling into a stub module
                valid, errs = _validate_piece_in_context(
                    module_name, consts_here, variables_line, typeok=typeok,
                )
                if valid:
                    return PieceResult(name="typeok", text=typeok, valid=True, attempts=attempt), consts_here
                last_errors = errs
                # Add error context to next attempt's prompt
                prompt = (
                    _TYPEOK_PROMPT.format(nl_description=nl_description, variables=variables_line)
                    + f"\n\nPrevious attempt had SANY errors:\n{chr(10).join(errs[:3])}\n\nFix the errors and try again."
                )
        except Exception as e:
            last_text = f"(error: {e})"
            last_errors = [str(e)]
    return (
        PieceResult(name="typeok", text=last_text, valid=False, attempts=max_attempts, errors=last_errors),
        constants,
    )


def _gen_init(
    client, nl_description: str, variables_line: str, typeok: str,
    constants: list[str], module_name: str, max_attempts: int = 3,
) -> PieceResult:
    """Generate Init predicate."""
    prompt = _INIT_PROMPT.format(nl_description=nl_description, variables=variables_line, typeok=typeok)
    last_text = ""
    last_errors = []
    for attempt in range(1, max_attempts + 1):
        try:
            text = _call_model(client, prompt, temperature=0.1 + attempt * 0.1)
            last_text = text
            init = _extract_definition(text, "Init")
            if init:
                valid, errs = _validate_piece_in_context(
                    module_name, constants, variables_line, typeok=typeok, init=init,
                )
                if valid:
                    return PieceResult(name="init", text=init, valid=True, attempts=attempt)
                last_errors = errs
                prompt = (
                    _INIT_PROMPT.format(nl_description=nl_description, variables=variables_line, typeok=typeok)
                    + f"\n\nPrevious attempt had SANY errors:\n{chr(10).join(errs[:3])}\n\nFix the errors and try again."
                )
        except Exception as e:
            last_text = f"(error: {e})"
            last_errors = [str(e)]
    return PieceResult(name="init", text=last_text, valid=False, attempts=max_attempts, errors=last_errors)


_SINGLE_ACTION_PROMPT = """\
You are designing a TLA+ specification for the following system:

{nl_description}

The state variables are: {variables}
TypeOK is:
{typeok}
Init is:
{init}

Already-defined actions ({action_count} so far):
{existing_actions}

Task: Define ONE more action for this system. The action should be a valid
state transition that hasn't been covered yet.

Output ONLY the action definition in this exact format:
{action_name} ==
  /\\ <preconditions on current state>
  /\\ var1' = <new value>
  /\\ var2' = <new value>
  /\\ UNCHANGED <<remaining_vars>>

Rules:
- Specify EVERY variable: either prime them OR put them in UNCHANGED.
- Never prime a variable AND list it in UNCHANGED in the same action.
- Use = (not ==) for comparisons inside conjuncts.
- Output ONLY the action definition, nothing else.
"""


def _gen_next(
    client, nl_description: str, variables_line: str, typeok: str, init: str,
    constants: list[str], module_name: str, max_attempts: int = 3,
) -> PieceResult:
    """Generate Next transition relation.

    First tries the all-at-once approach (faster). If 3 attempts fail,
    falls back to per-action generation: ask the model for one action at
    a time, validate each, then stitch them into Next == A1 \\/ A2 \\/ ...
    """
    prompt = _NEXT_PROMPT.format(
        nl_description=nl_description, variables=variables_line, typeok=typeok, init=init
    )
    last_text = ""
    last_errors = []
    for attempt in range(1, max_attempts + 1):
        try:
            text = _call_model(client, prompt, temperature=0.2 + attempt * 0.1, max_tokens=1500)
            last_text = text
            next_def = _extract_next_block(text)
            if next_def:
                valid, errs = _validate_piece_in_context(
                    module_name, constants, variables_line,
                    typeok=typeok, init=init, next_def=next_def,
                )
                if valid:
                    return PieceResult(name="next", text=next_def, valid=True, attempts=attempt)
                last_errors = errs
                prompt = (
                    _NEXT_PROMPT.format(
                        nl_description=nl_description, variables=variables_line,
                        typeok=typeok, init=init,
                    )
                    + f"\n\nPrevious attempt had SANY errors:\n{chr(10).join(errs[:3])}\n\nFix the errors and try again."
                )
        except Exception as e:
            last_text = f"(error: {e})"
            last_errors = [str(e)]

    # ── Fallback: per-action generation ──
    # All-at-once failed; ask the model for one action at a time and stitch.
    fallback_attempts = 0
    valid_actions: list[str] = []
    for action_idx in range(1, 5):  # try up to 4 actions
        action_name = f"Action{action_idx}"
        existing = "\n\n".join(valid_actions) if valid_actions else "(none yet)"
        single_prompt = _SINGLE_ACTION_PROMPT.format(
            nl_description=nl_description,
            variables=variables_line,
            typeok=typeok,
            init=init,
            action_count=len(valid_actions),
            existing_actions=existing,
            action_name=action_name,
        )
        try:
            text = _call_model(client, single_prompt, temperature=0.3, max_tokens=600)
            fallback_attempts += 1
            action_def = _extract_definition(text, action_name)
            if not action_def:
                continue
            # Validate this single action by adding it to a stub Next
            test_next = "\n\n".join(valid_actions + [action_def]) + "\n\n"
            test_next += f"Next == " + " \\/ ".join(f"Action{i+1}" for i in range(len(valid_actions) + 1))
            valid, errs = _validate_piece_in_context(
                module_name, constants, variables_line,
                typeok=typeok, init=init, next_def=test_next,
            )
            if valid:
                valid_actions.append(action_def)
        except Exception:
            continue

    if valid_actions:
        # Build final Next definition
        final_next = "\n\n".join(valid_actions) + "\n\n"
        final_next += f"Next == " + " \\/ ".join(f"Action{i+1}" for i in range(len(valid_actions)))
        return PieceResult(
            name="next", text=final_next, valid=True,
            attempts=max_attempts + fallback_attempts,
        )

    return PieceResult(
        name="next", text=last_text, valid=False,
        attempts=max_attempts + fallback_attempts, errors=last_errors,
    )


def _extract_next_block(text: str) -> Optional[str]:
    """Extract the Next definition AND any helper action definitions it references."""
    text = re.sub(r"```\w*\n?", "", text)
    text = re.sub(r"```", "", text)

    # Find Next == ...
    next_def = _extract_definition(text, "Next")
    if not next_def:
        return None

    # Collect all action names referenced in Next (capitalized identifiers
    # that aren't TLA+ keywords)
    action_names = re.findall(r"\b([A-Z][a-zA-Z_0-9]*)\b", next_def)
    keywords = {"UNCHANGED", "TRUE", "FALSE", "CHOOSE", "LET", "IN", "IF", "THEN", "ELSE",
                "EXCEPT", "DOMAIN", "UNION", "SUBSET", "SF", "WF"}
    action_names = [n for n in action_names if n not in keywords and n != "Next"]

    # Find each action definition in the text
    helper_defs = []
    for action in dict.fromkeys(action_names):  # dedupe, preserve order
        defn = _extract_definition(text, action)
        if defn and defn != next_def:
            helper_defs.append(defn)

    if helper_defs:
        return "\n\n".join(helper_defs) + "\n\n" + next_def
    return next_def


def _gen_invariant(
    client, nl_description: str, variables_line: str, typeok: str, init: str, next_def: str,
    constants: list[str], module_name: str, max_attempts: int = 2,
) -> PieceResult:
    """Generate optional safety invariant. Failures are non-fatal."""
    prompt = _INVARIANTS_PROMPT.format(
        nl_description=nl_description, variables=variables_line,
        typeok=typeok, init=init, next_def=next_def,
    )
    last_text = ""
    last_errors = []
    for attempt in range(1, max_attempts + 1):
        try:
            text = _call_model(client, prompt, temperature=0.2 + attempt * 0.1)
            last_text = text
            inv = _extract_definition(text, "SafetyInvariant")
            if inv:
                valid, errs = _validate_piece_in_context(
                    module_name, constants, variables_line,
                    typeok=typeok, init=init, next_def=next_def, invariant=inv,
                )
                if valid:
                    return PieceResult(name="invariant", text=inv, valid=True, attempts=attempt)
                last_errors = errs
        except Exception as e:
            last_text = f"(error: {e})"
            last_errors = [str(e)]
    return PieceResult(name="invariant", text=last_text, valid=False, attempts=max_attempts, errors=last_errors)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────

def generate_piecewise(
    problem_id: str,
    nl_description: str,
    module_name: str = "Spec",
    model_tag: str = "chattla:20b",
) -> PiecewiseResult:
    """Generate a TLA+ spec piece-by-piece with per-piece SANY validation."""
    from src.inference.ollama_client import ChatTLAClient
    from src.validators.tlc_validator import validate_string

    client = ChatTLAClient(model=model_tag, reasoning="low")
    pieces: list[PieceResult] = []
    total_attempts = 0

    # ── Step 1: VARIABLES ──
    var_piece = _gen_variables(client, nl_description, module_name)
    pieces.append(var_piece)
    total_attempts += var_piece.attempts
    if not var_piece.valid:
        return PiecewiseResult(
            problem_id=problem_id, spec="", pieces=pieces,
            final_tier="bronze", total_attempts=total_attempts,
        )
    variables_line = var_piece.text

    # ── Step 2: TypeOK ──
    typeok_piece, constants = _gen_typeok(client, nl_description, variables_line, module_name)
    pieces.append(typeok_piece)
    total_attempts += typeok_piece.attempts
    if not typeok_piece.valid:
        return PiecewiseResult(
            problem_id=problem_id, spec="", pieces=pieces,
            final_tier="bronze", total_attempts=total_attempts, constants=constants,
        )
    typeok = typeok_piece.text

    # ── Step 3: Init ──
    init_piece = _gen_init(client, nl_description, variables_line, typeok, constants, module_name)
    pieces.append(init_piece)
    total_attempts += init_piece.attempts
    if not init_piece.valid:
        return PiecewiseResult(
            problem_id=problem_id, spec="", pieces=pieces,
            final_tier="bronze", total_attempts=total_attempts, constants=constants,
        )
    init = init_piece.text

    # ── Step 4: Next ──
    next_piece = _gen_next(client, nl_description, variables_line, typeok, init, constants, module_name)
    pieces.append(next_piece)
    total_attempts += next_piece.attempts
    if not next_piece.valid:
        return PiecewiseResult(
            problem_id=problem_id, spec="", pieces=pieces,
            final_tier="bronze", total_attempts=total_attempts, constants=constants,
        )
    next_def = next_piece.text

    # ── Step 5: Invariant (optional) ──
    inv_piece = _gen_invariant(
        client, nl_description, variables_line, typeok, init, next_def, constants, module_name
    )
    pieces.append(inv_piece)
    total_attempts += inv_piece.attempts
    invariant = inv_piece.text if inv_piece.valid else None

    # ── Final assembly + full TLC validation ──
    final_spec = _assemble_spec(
        module_name=module_name,
        constants=constants,
        variables_line=variables_line,
        typeok=typeok,
        init=init,
        next_def=next_def,
        invariant=invariant,
    )

    tlc_result = validate_string(final_spec, module_name=module_name)
    final_tier = tlc_result.tier

    return PiecewiseResult(
        problem_id=problem_id,
        spec=final_spec,
        pieces=pieces,
        final_tier=final_tier,
        total_attempts=total_attempts,
        constants=constants,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI for benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def _benchmark(model_tag: str, output_csv: str, limit: Optional[int] = None) -> None:
    """Run the piece-wise generator against the 20-problem benchmark suite."""
    import csv
    import json
    import time
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    bench_path = repo_root / "data" / "benchmarks" / "benchmark_suite.json"
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with bench_path.open() as f:
        problems = json.load(f)
    if limit:
        problems = problems[:limit]

    fields = [
        "model", "benchmark_id", "name", "domain", "difficulty",
        "sany_pass", "tlc_pass", "tlc_tier",
        "variables_ok", "typeok_ok", "init_ok", "next_ok", "invariant_ok",
        "total_attempts", "runtime_s",
    ]

    rows = []
    sany_passes = 0
    tlc_passes = 0

    print(f"\n[piecewise] Model: {model_tag}")
    print(f"[piecewise] Benchmark: {len(problems)} problems\n")

    for p in problems:
        pid = p["id"]
        # Build module name from benchmark name
        mod_name = re.sub(r"[^A-Za-z0-9]", "", p["name"]) or pid

        description = p["description"]
        if p.get("hints"):
            description += f"\n\nHints: {p['hints']}"

        t0 = time.monotonic()
        try:
            result = generate_piecewise(
                problem_id=pid,
                nl_description=description,
                module_name=mod_name,
                model_tag=model_tag,
            )
            elapsed = time.monotonic() - t0
            piece_status = {p.name: p.valid for p in result.pieces}
            sany_pass = int(result.final_tier in ("silver", "gold"))
            tlc_pass = int(result.final_tier == "gold")
            sany_passes += sany_pass
            tlc_passes += tlc_pass

            print(
                f"  [{pid}] {p['name']:35s} tier={result.final_tier:6s} "
                f"pieces={sum(piece_status.values())}/{len(piece_status)} "
                f"attempts={result.total_attempts} t={elapsed:.0f}s"
            )

            rows.append({
                "model": model_tag,
                "benchmark_id": pid,
                "name": p["name"],
                "domain": p.get("domain", ""),
                "difficulty": p.get("difficulty", 0),
                "sany_pass": sany_pass,
                "tlc_pass": tlc_pass,
                "tlc_tier": result.final_tier,
                "variables_ok": int(piece_status.get("variables", False)),
                "typeok_ok": int(piece_status.get("typeok", False)),
                "init_ok": int(piece_status.get("init", False)),
                "next_ok": int(piece_status.get("next", False)),
                "invariant_ok": int(piece_status.get("invariant", False)),
                "total_attempts": result.total_attempts,
                "runtime_s": round(elapsed, 1),
            })
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"  [{pid}] {p['name']:35s} ERROR: {e}")
            rows.append({
                "model": model_tag,
                "benchmark_id": pid,
                "name": p["name"],
                "domain": p.get("domain", ""),
                "difficulty": p.get("difficulty", 0),
                "sany_pass": 0,
                "tlc_pass": 0,
                "tlc_tier": "error",
                "variables_ok": 0, "typeok_ok": 0, "init_ok": 0,
                "next_ok": 0, "invariant_ok": 0,
                "total_attempts": 0,
                "runtime_s": round(elapsed, 1),
            })

    n = len(problems)
    print(
        f"\n[piecewise] {model_tag}: SANY={sany_passes}/{n} ({sany_passes/n:.0%})  "
        f"TLC={tlc_passes}/{n} ({tlc_passes/n:.0%})"
    )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[piecewise] Results: {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Piece-wise verified TLA+ generation")
    p.add_argument("--model", default="chattla:20b", help="Ollama model tag")
    p.add_argument("--output", default="outputs/benchmark_results/piecewise_20260406.csv")
    p.add_argument("--limit", type=int, default=None, help="Limit benchmark size for testing")
    args = p.parse_args()
    _benchmark(args.model, args.output, args.limit)
