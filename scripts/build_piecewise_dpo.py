#!/usr/bin/env python3
"""build_piecewise_dpo.py — Generate piecewise + full-spec DPO pairs.

DeepSeek-Prover-V2-inspired: decompose TLA+ specs into 5 pieces, generate
multiple candidates for each piece, validate incrementally, and form
preference pairs from (passing, failing) candidates.

Pieces:
  1. VARIABLES  — state variable declaration
  2. TypeOK     — finite-set type invariant
  3. Init       — initial state predicate
  4. Next       — action disjunction
  5. Invariants — safety properties

For each piece, we generate N candidates via Ollama, validate with SANY
(and TLC depth-1 where applicable), and pair (best, worst) by validation
tier.

Also generates full-spec DPO pairs from the 200-topic curriculum.

Usage:
    python -m scripts.build_piecewise_dpo --smoke          # 5 specs, 2 candidates
    python -m scripts.build_piecewise_dpo --max-specs 50   # first 50 specs
    python -m scripts.build_piecewise_dpo                  # all specs (~4-8h)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.validators.sany_validator import validate_string as sany_validate
from src.validators.component_validator import reward_from_spec

# ── Section extraction (reused from build_multitask_sft.py) ──────────────

def _extract_section(spec: str, name: str) -> Optional[str]:
    lines = spec.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if re.match(rf"^\s*{re.escape(name)}\s*==", line):
            start_idx = i
            break
    if start_idx is None:
        return None
    out_lines = [lines[start_idx]]
    for line in lines[start_idx + 1:]:
        if re.match(r"^[A-Za-z_]\w*\s*==", line):
            break
        if re.match(r"^={4,}", line):
            break
        out_lines.append(line)
    while out_lines and not out_lines[-1].strip():
        out_lines.pop()
    return "\n".join(out_lines) if out_lines else None


def _extract_variables_decl(spec: str) -> Optional[str]:
    m = re.search(r"^\s*VARIABLES?\s+.+$", spec, re.MULTILINE)
    return m.group(0).strip() if m else None


def _extract_constants_decl(spec: str) -> Optional[str]:
    matches = re.findall(r"^\s*CONSTANTS?\s+.+$", spec, re.MULTILINE)
    return "\n".join(m.strip() for m in matches) if matches else None


def _extract_extends(spec: str) -> Optional[str]:
    m = re.search(r"^\s*EXTENDS\s+.+$", spec, re.MULTILINE)
    return m.group(0).strip() if m else None


def _extract_module_header(spec: str) -> str:
    m = re.search(r"^-+\s*MODULE\s+\w+\s*-+", spec, re.MULTILINE)
    return m.group(0) if m else "---- MODULE Temp ----"


def _extract_module_name(spec: str) -> str:
    m = re.search(r"MODULE\s+(\w+)", spec)
    return m.group(1) if m else "Temp"


def _extract_nl_description(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            content = re.sub(
                r"^Write a TLA\+\s*(formal\s+)?specification\s*(for\s*(the following)?:?\s*)?",
                "", content, flags=re.IGNORECASE,
            ).strip()
            return content
    return ""


def _extract_spec(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if "MODULE" in content:
                return content
    return ""

# ── Piece-wise prompts (from piecewise_gen.py) ───────────────────────────

PIECE_PROMPTS = {
    "VARIABLES": """\
You are designing a TLA+ specification for the following system:

{nl_description}

Task: List the state variables this system needs.

Output ONLY a single line in this exact format:
VARIABLES var1, var2, var3

Rules:
- Use lowercase variable names.
- Pick the minimum set of variables needed to describe the system's state.
- Do NOT define what the variables mean — just list them.
- Do NOT output anything except the VARIABLES line.""",

    "TypeOK": """\
You are designing a TLA+ specification for the following system:

{nl_description}

The state variables are: {variables}

Task: Define a TypeOK invariant that constrains every variable to a finite set.

Output ONLY the TypeOK definition in this exact format:
TypeOK ==
  /\\ var1 \\in <finite_set>
  /\\ var2 \\in <finite_set>

Rules:
- Every variable from VARIABLES must appear in TypeOK.
- Use bounded ranges like 0..N, finite explicit sets, or [1..N -> {{"x", "y"}}] for functions.
- NEVER use Nat or Int unbounded.
- If you need a CONSTANT, write "CONSTANT N" on a separate line BEFORE TypeOK.
- Do NOT output anything except the optional CONSTANT line(s) and the TypeOK definition.""",

    "Init": """\
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

Rules:
- Every variable must get a concrete starting value.
- The values must satisfy TypeOK.
- Use = (not ==) for variable assignments inside Init.
- Do NOT output anything except the Init definition.""",

    "Next": """\
You are designing a TLA+ specification for the following system:

{nl_description}

The state variables are: {variables}
The TypeOK invariant is:
{typeok}
The initial state is:
{init}

Task: Define the Next action — the transition relation.

Output ONLY the Next action definition and any helper sub-actions.

Rules:
- Every disjunct in Next must specify ALL variables: either prime them (x' = ...) or use UNCHANGED <<x>>.
- Use ASCII operators only (/\\, \\/, \\in, ->, |->).
- Define each sub-action separately, then combine: Next == Action1 \\/ Action2 \\/ ...
- If the system can terminate, add a Terminating == UNCHANGED vars disjunct.""",

    "Invariants": """\
You are designing a TLA+ specification for the following system:

{nl_description}

Here is the partial spec so far:
{partial_spec}

Task: Define additional safety invariants beyond TypeOK.

Output ONLY the invariant definitions. For example:
SafetyInv ==
  /\\ <meaningful_property>

Rules:
- The invariant must be meaningful — not just a restatement of TypeOK.
- It should capture a real safety property of the system.
- If no meaningful invariant exists, output: NoExtraInvariants == TRUE""",
}


# ── Ollama generation ────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str, temperature: float) -> str:
    """Generate a single completion via Ollama REST API."""
    import urllib.request
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 1024},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read()).get("response", "")
    except Exception as e:
        print(f"  [ollama] error: {e}")
        return ""


# ── Validation helpers ───────────────────────────────────────────────────

def _assemble_spec(
    module_name: str,
    extends: str,
    constants: str,
    variables: str,
    typeok: str,
    init: str,
    next_def: str,
    invariants: str,
    spec_def: str = "",
) -> str:
    """Assemble a full TLA+ spec from pieces."""
    parts = [f"---- MODULE {module_name} ----"]
    if extends:
        parts.append(extends)
    if constants:
        parts.append(constants)
    parts.append(variables)
    if typeok:
        parts.append(f"\n{typeok}")
    if init:
        parts.append(f"\n{init}")
    if next_def:
        parts.append(f"\n{next_def}")
    # vars tuple
    var_names = re.findall(r"VARIABLES?\s+(.*)", variables)
    if var_names:
        vlist = [v.strip() for v in var_names[0].split(",")]
        parts.append(f"\nvars == << {', '.join(vlist)} >>")
    if not spec_def:
        spec_def = "Spec == Init /\\ [][Next]_vars"
    parts.append(f"\n{spec_def}")
    if invariants:
        parts.append(f"\n{invariants}")
    parts.append("====")
    return "\n".join(parts)


def _validate_piece_sany(assembled_spec: str, module_name: str) -> bool:
    """Quick SANY check on an assembled spec."""
    result = sany_validate(assembled_spec, module_name=module_name)
    return result.valid


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


# ── Main DPO pair generation ────────────────────────────────────────────

def generate_piecewise_pairs(
    specs: list[dict],
    model: str = "chattla:20b",
    n_candidates: int = 8,
    output_path: Path = _REPO_ROOT / "data" / "processed" / "piecewise_dpo_pairs.jsonl",
    smoke: bool = False,
) -> int:
    """Generate piecewise DPO pairs from gold specs.

    For each spec and each piece, generate candidates, validate, and form
    (passing, failing) preference pairs.
    """
    if smoke:
        specs = specs[:5]
        n_candidates = 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    seen_hashes: set[str] = set()

    with open(output_path, "w", encoding="utf-8", buffering=1) as out_f:  # line-buffered
        for si, row in enumerate(specs):
            messages = row.get("messages", [])
            spec = _extract_spec(messages)
            nl = _extract_nl_description(messages)
            if not spec or not nl:
                continue

            module_name = _extract_module_name(spec)
            extends = _extract_extends(spec) or "EXTENDS Naturals, FiniteSets"
            constants = _extract_constants_decl(spec) or ""
            gold_variables = _extract_variables_decl(spec) or ""
            gold_typeok = _extract_section(spec, "TypeOK") or ""
            gold_init = _extract_section(spec, "Init") or ""
            gold_next = _extract_section(spec, "Next") or ""
            gold_invs = ""
            for inv_name in ("SafetyInv", "MutexSafe", "Safety", "InOrderPrefix",
                             "ChannelSafe", "Consistency", "NoDeadlock"):
                section = _extract_section(spec, inv_name)
                if section:
                    gold_invs += section + "\n"

            print(f"[{si+1}/{len(specs)}] {module_name}: nl={len(nl)}ch, "
                  f"vars={bool(gold_variables)}, typeok={bool(gold_typeok)}, "
                  f"init={bool(gold_init)}, next={bool(gold_next)}")

            # Generate candidates for each piece
            pieces = {
                "VARIABLES": {"gold": gold_variables, "context": {}},
                "TypeOK": {"gold": gold_typeok, "context": {"variables": gold_variables}},
                "Init": {"gold": gold_init, "context": {"variables": gold_variables, "typeok": gold_typeok}},
                "Next": {"gold": gold_next, "context": {
                    "variables": gold_variables, "typeok": gold_typeok, "init": gold_init}},
                "Invariants": {"gold": gold_invs, "context": {"partial_spec": spec[:2000]}},
            }

            for piece_name, piece_info in pieces.items():
                if not piece_info["gold"]:
                    continue

                template = PIECE_PROMPTS[piece_name]
                ctx = {"nl_description": nl[:800], **piece_info["context"]}
                prompt = template.format_map({k: ctx.get(k, "") for k in
                    re.findall(r"\{(\w+)\}", template)})

                candidates: list[dict] = []
                for ci in range(n_candidates):
                    temp = 0.7 + 0.3 * (ci / max(n_candidates - 1, 1))
                    text = _call_ollama(prompt, model, temperature=temp)
                    if not text.strip():
                        continue

                    # Assemble full spec with this candidate piece
                    assembled_parts = {
                        "VARIABLES": gold_variables,
                        "TypeOK": gold_typeok,
                        "Init": gold_init,
                        "Next": gold_next,
                        "Invariants": gold_invs,
                    }
                    assembled_parts[piece_name] = text.strip()

                    assembled = _assemble_spec(
                        module_name, extends, constants,
                        assembled_parts["VARIABLES"],
                        assembled_parts["TypeOK"],
                        assembled_parts["Init"],
                        assembled_parts["Next"],
                        assembled_parts["Invariants"],
                    )

                    sany_ok = _validate_piece_sany(assembled, module_name)
                    # For Init/Next/Invariants, also get component reward
                    reward = 0.0
                    if sany_ok and piece_name in ("Init", "Next", "Invariants"):
                        try:
                            reward = reward_from_spec(
                                assembled, module_name=module_name,
                                run_depth1=True, run_full_tlc=False,
                            )
                        except Exception:
                            pass
                    elif sany_ok:
                        reward = 0.3  # SANY pass is worth something for VARIABLES/TypeOK

                    candidates.append({
                        "text": text.strip(),
                        "sany_ok": sany_ok,
                        "reward": reward,
                        "hash": _content_hash(text),
                    })

                # Form pairs: best vs worst by reward
                if len(candidates) < 2:
                    continue

                candidates.sort(key=lambda c: c["reward"], reverse=True)
                best = candidates[0]
                # Find worst that's meaningfully different
                pairs_made = 0
                for worst in reversed(candidates):
                    if worst["hash"] == best["hash"]:
                        continue
                    if abs(best["reward"] - worst["reward"]) < 0.01:
                        continue
                    pair_hash = f"{module_name}_{piece_name}_{best['hash']}_{worst['hash']}"
                    if pair_hash in seen_hashes:
                        continue
                    seen_hashes.add(pair_hash)

                    pair = {
                        "prompt": prompt,
                        "chosen": best["text"],
                        "rejected": worst["text"],
                        "chosen_reward": best["reward"],
                        "rejected_reward": worst["reward"],
                        "piece_name": piece_name,
                        "module_name": module_name,
                        "chosen_sany_ok": best["sany_ok"],
                        "rejected_sany_ok": worst["sany_ok"],
                    }
                    out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    total_pairs += 1
                    pairs_made += 1
                    if pairs_made >= 3:  # max 3 pairs per (spec, piece)
                        break

                print(f"  {piece_name}: {len(candidates)} candidates, {pairs_made} pairs "
                      f"(best={best['reward']:.2f}, sany={sum(c['sany_ok'] for c in candidates)}/{len(candidates)})")

    return total_pairs


def generate_fullspec_pairs(
    topics: list[dict],
    model: str = "chattla:20b",
    n_candidates: int = 4,
    output_path: Path = _REPO_ROOT / "data" / "processed" / "piecewise_dpo_pairs.jsonl",
    smoke: bool = False,
) -> int:
    """Generate full-spec DPO pairs from the 200-topic curriculum."""
    if smoke:
        topics = topics[:3]
        n_candidates = 2

    total_pairs = 0
    developer_prompt = (
        "You are ChatTLA, an expert at writing verified TLA+ formal specifications.\n"
        "Write a complete, valid TLA+ spec that passes SANY and TLC.\n"
        "Output only the spec — no markdown fences, no explanation.\n"
        "Reasoning: medium"
    )

    with output_path.open("a", encoding="utf-8") as out_f:
        for ti, topic in enumerate(topics):
            module = topic.get("module", "Spec")
            desc = topic.get("desc", "")
            if not desc:
                continue

            prompt = f"{developer_prompt}\n\nWrite a TLA+ specification for:\n{desc}"
            print(f"[fullspec {ti+1}/{len(topics)}] {module}")

            candidates: list[dict] = []
            for ci in range(n_candidates):
                temp = 0.5 + 0.4 * (ci / max(n_candidates - 1, 1))
                text = _call_ollama(prompt, model, temperature=temp)
                if not text.strip():
                    continue
                try:
                    reward = reward_from_spec(text, run_depth1=True, run_full_tlc=True)
                except Exception:
                    reward = 0.0
                candidates.append({
                    "text": text.strip(),
                    "reward": reward,
                    "hash": _content_hash(text),
                })

            if len(candidates) < 2:
                continue

            candidates.sort(key=lambda c: c["reward"], reverse=True)
            best = candidates[0]
            worst = candidates[-1]
            if best["hash"] == worst["hash"] or abs(best["reward"] - worst["reward"]) < 0.01:
                continue

            pair = {
                "prompt": prompt,
                "chosen": best["text"],
                "rejected": worst["text"],
                "chosen_reward": best["reward"],
                "rejected_reward": worst["reward"],
                "piece_name": "full_spec",
                "module_name": module,
                "chosen_sany_ok": True,
                "rejected_sany_ok": False,
            }
            out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            total_pairs += 1
            print(f"  full_spec: best={best['reward']:.2f}, worst={worst['reward']:.2f}")

    return total_pairs


def _load_diamond_specs() -> list[dict]:
    """Load diamond SFT specs for piecewise pair generation."""
    rows = []
    for name in ("diamond_curated.jsonl", "diamond_sft.jsonl"):
        path = _REPO_ROOT / "data" / "processed" / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # Deduplicate by module name
    seen = set()
    deduped = []
    for row in rows:
        module = row.get("module", "") or _extract_module_name(
            _extract_spec(row.get("messages", [])))
        if module in seen:
            continue
        seen.add(module)
        deduped.append(row)
    return deduped


def _load_topics() -> list[dict]:
    """Load 200 topics from diamond_gen_topics.json."""
    path = _REPO_ROOT / "data" / "diamond_gen_topics.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    topics = []
    for batch in data.get("batches", []):
        topics.extend(batch.get("topics", []))
    return topics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="chattla:20b",
                        help="Ollama model for candidate generation")
    parser.add_argument("--n-candidates", type=int, default=8,
                        help="Candidates per piece (default 8)")
    parser.add_argument("--max-specs", type=int, default=None,
                        help="Limit number of specs to process")
    parser.add_argument("--output", default="data/processed/piecewise_dpo_pairs.jsonl")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: 5 specs, 2 candidates")
    parser.add_argument("--skip-piecewise", action="store_true",
                        help="Only generate full-spec pairs")
    parser.add_argument("--skip-fullspec", action="store_true",
                        help="Only generate piecewise pairs")
    args = parser.parse_args()

    output_path = _REPO_ROOT / args.output

    total = 0

    if not args.skip_piecewise:
        specs = _load_diamond_specs()
        if args.max_specs:
            specs = specs[:args.max_specs]
        print(f"Loaded {len(specs)} diamond specs for piecewise DPO")
        total += generate_piecewise_pairs(
            specs, model=args.model, n_candidates=args.n_candidates,
            output_path=output_path, smoke=args.smoke,
        )

    if not args.skip_fullspec:
        topics = _load_topics()
        print(f"\nLoaded {len(topics)} topics for full-spec DPO")
        total += generate_fullspec_pairs(
            topics, model=args.model, n_candidates=4,
            output_path=output_path, smoke=args.smoke,
        )

    print(f"\nTotal DPO pairs generated: {total}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
