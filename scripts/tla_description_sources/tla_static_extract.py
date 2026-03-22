"""
Deterministic extraction of structured description fields from TLA+ source text.

No LLM: regex + line scans. Works well on typical Examples specs; edge cases may
need manual review. Complements README/manifest/PDF harvest prose.
"""

from __future__ import annotations

import re
from typing import Any

# Top-level definition: Name or Name(args) followed by ==
_NAME_EQ = re.compile(r"^\s*([A-Za-z]\w*)\s*(?:\([^)]*\))?\s*==")


def strip_block_comments(text: str) -> str:
    """Remove (* ... *) with nesting."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if i + 1 < n and text[i : i + 2] == "(*":
            depth = 1
            i += 2
            while i < n and depth:
                if i + 1 < n and text[i : i + 2] == "(*":
                    depth += 1
                    i += 2
                elif i + 1 < n and text[i : i + 2] == "*)":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def strip_line_comments(text: str) -> str:
    """Remove \\* to end of line."""
    lines = []
    for line in text.splitlines():
        j = 0
        buf = []
        while j < len(line):
            if j + 1 < len(line) and line[j] == "\\" and line[j + 1] == "*":
                break
            buf.append(line[j])
            j += 1
        lines.append("".join(buf))
    return "\n".join(lines)


def preprocess_tla(text: str) -> str:
    return strip_line_comments(strip_block_comments(text))


def split_decl_list(blob: str) -> list[str]:
    """Split 'a, b, c' respecting no nesting (good enough for CONSTANTS/V VARIABLES)."""
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in blob:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            p = "".join(cur).strip()
            if p:
                parts.append(p)
            cur = []
        else:
            cur.append(ch)
    p = "".join(cur).strip()
    if p:
        parts.append(p)
    return parts


def extract_constants_block(text: str) -> str:
    m = re.search(
        r"(?m)^\s*CONSTANTS?\s+(.+?)(?=^\s*(?:VARIABLES?|ASSUME|EXTENDS)\b)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return ""
    return m.group(1).strip().replace("\n", " ")[:2000]


def extract_variables_block(text: str) -> list[str]:
    m = re.search(
        r"(?m)^\s*VARIABLES?\s+(.+?)(?=^\s*(?:CONSTANTS?|ASSUME|EXTENDS|[A-Za-z]\w*\s*(?:\([^)]*\))?\s*==))",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return []
    blob = m.group(1).strip()
    return split_decl_list(blob)


def extract_op_body(text: str, op_name: str, max_chars: int = 4000) -> str:
    pat = rf"^\s*{re.escape(op_name)}\s*(?:\([^)]*\))?\s*=="
    m = re.search(pat, text, re.MULTILINE)
    if not m:
        return ""
    rest = text[m.end() :]
    lines_out: list[str] = []
    for line in rest.splitlines():
        me = _NAME_EQ.match(line)
        if me and me.group(1) != op_name:
            break
        lines_out.append(line)
    body = "\n".join(lines_out).strip()
    if len(body) > max_chars:
        body = body[: max_chars - 20] + "\n… [truncated]"
    return body


def list_definition_names(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        me = _NAME_EQ.match(line)
        if me:
            names.append(me.group(1))
    return names


def pick_primary_init(text: str) -> str:
    for cand in ("Init", "init"):
        b = extract_op_body(text, cand, max_chars=3500)
        if b:
            return b
    for name in list_definition_names(text):
        if name.startswith("Init") and name != "Init":
            b = extract_op_body(text, name, max_chars=2500)
            if b:
                return b
    return ""


def pick_next_body(text: str) -> str:
    for cand in ("Next", "next"):
        b = extract_op_body(text, cand, max_chars=5000)
        if b:
            return b
    return ""


def pick_spec_line(text: str) -> str:
    for cand in ("Spec", "FairSpec", "LiveSpec"):
        b = extract_op_body(text, cand, max_chars=2000)
        if b:
            return b
    return ""


def names_referenced_in_next(next_body: str, known: set[str]) -> list[str]:
    """Heuristic: names from `known` that appear as callees in Next."""
    skip = {
        "Next",
        "Init",
        "Spec",
        "TRUE",
        "FALSE",
        "IF",
        "THEN",
        "ELSE",
        "LET",
        "IN",
        "DOMAIN",
        "UNION",
        "SUBSET",
    }
    found: list[str] = []
    for name in sorted(known, key=lambda x: -len(x)):
        if len(name) < 2 or name in skip:
            continue
        if re.search(rf"\b{re.escape(name)}\s*\(", next_body):
            found.append(name)
    return found


def guess_invariants(names: list[str], text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    inv_pat = re.compile(r"(Type|Inv|Invariant|Safety|Live|Spec|Inductive)", re.I)
    for n in names:
        if inv_pat.search(n) or n.startswith("Type") or "Inv" in n:
            body = extract_op_body(text, n, max_chars=1200)
            if body:
                purpose = "Type/safety/live constraint" if "Type" in n else "Named property or invariant"
                out.append({"name": n, "assertion": body, "purpose": purpose})
    return out[:12]


def infer_variable_type(name: str) -> str:
    # Very rough — TLA+ is untyped; we hint from naming
    lower = name.lower()
    if lower.startswith("pc") or lower == "self":
        return "[control / process state]"
    if "chan" in lower or "msg" in lower or "buf" in lower:
        return "[message/channel component]"
    if lower.endswith("s") and lower not in ("vars",):
        return "[set or sequence component]"
    return "[state function / data value]"


def nondeterminism_note(full_text: str, next_body: str) -> str:
    blob = next_body + full_text[:12000]
    bits: list[str] = []
    if r"\E" in blob or " \\E " in blob:
        bits.append("existential quantification (\\E) chooses a witness (e.g. process or schedule step).")
    if "CHOOSE" in blob:
        bits.append("CHOOSE picks a deterministic value satisfying a predicate.")
    if r"\/" in next_body and "Next" in full_text:
        bits.append("disjunction (\\/) combines alternative transitions.")
    if not bits:
        return "No explicit \\E/CHOOSE detected in scanned Next/spec fragment; may be deterministic or use library ops."
    return " ".join(bits)


def fairness_note(full_text: str) -> str:
    spec_blob = pick_spec_line(full_text) + full_text[-4000:]
    parts: list[str] = []
    if "WF_" in spec_blob:
        parts.append("Weak fairness (WF_) on action(s).")
    if "SF_" in spec_blob:
        parts.append("Strong fairness (SF_) on action(s).")
    if "Fairness" in spec_blob or "fair" in spec_blob.lower():
        parts.append("Fairness constraints appear in Spec or related definitions.")
    if not parts:
        return "No WF_/SF_ tokens found in Spec fragment; liveness may be absent or expressed differently."
    return " ".join(parts)


def build_programmatic_narrative(
    *,
    readme_title: str,
    module_name: str,
    header_excerpt: str,
    var_names: list[str],
    next_body: str,
    harvest_prose_tail: str,
    full_clean_text: str,
) -> str:
    """3–5 sentences, no TLA+ syntax in narrative (plain English summaries)."""
    title = readme_title.strip() or f"the {module_name} specification"
    s1 = f"This work formalizes {title} as a TLA+ module."
    s2 = (
        f"The model state is organized around variables such as {', '.join(var_names[:8])}"
        + (" among others." if len(var_names) > 8 else ".")
        if var_names
        else "State variables are declared in the module (see technical.variables)."
    )
    nd = nondeterminism_note(full_clean_text, next_body)
    s3 = f"Regarding branching: {nd}"
    s4 = fairness_note(full_clean_text)
    if header_excerpt:
        hint = header_excerpt.replace("\n", " ").strip()[:400]
        s5 = f"Module commentary from authors: {hint}"
    else:
        s5 = ""
    # Optional: PDF/harvest tail is often technical — one short pointer
    s6 = ""
    if harvest_prose_tail and len(harvest_prose_tail) > len(header_excerpt) + 50:
        s6 = "Additional harvested context (README, references, or paper excerpt) is attached in the dataset row outside this narrative."
    parts = [s1, s2, s3, s4]
    if s5:
        parts.append(s5)
    if s6:
        parts.append(s6)
    return " ".join(parts)


def extract_structured_description(
    tla_text: str,
    *,
    module_name: str,
    readme_title: str = "",
    header_comment: str = "",
    harvest_prose: str = "",
    max_action_defs: int = 24,
) -> dict[str, Any]:
    """
    Return { "narrative": str, "technical": { ... } } without calling an LLM.
    """
    clean = preprocess_tla(tla_text)
    const_blob = extract_constants_block(clean)
    var_names = extract_variables_block(clean)
    init_b = pick_primary_init(clean)
    next_b = pick_next_body(clean)
    spec_b = pick_spec_line(clean)
    all_names = list_definition_names(clean)
    known = set(all_names)
    # Action-like defs: appear in Next or look like Verb phrases
    ref_from_next = names_referenced_in_next(next_b, known) if next_b else []
    action_names = [
        n for n in ref_from_next if n not in ("Init", "Next", "Spec", "vars", "Init0", "Init1")
    ][:max_action_defs]
    if not action_names:
        # fallback: first N definition names that look like transitions (heuristic)
        reserved = {
            "Init",
            "Next",
            "Spec",
            "vars",
            "ASSUME",
            "Proc",
            "Location",
            "guardE",
            "guardR1",
            "guardR2",
        }
        action_names = [n for n in all_names if n not in reserved and not n.startswith("Type")][:max_action_defs]

    actions: list[dict[str, str]] = []
    for an in action_names[:max_action_defs]:
        ab = extract_op_body(clean, an, max_chars=900)
        if not ab:
            continue
        actions.append(
            {
                "name": an,
                "intent": f"Transition {an} (extracted from definition).",
                "pre": "[see definition body]",
                "post": ab[:800],
            }
        )

    variables = [
        {
            "name": vn,
            "type": infer_variable_type(vn),
            "role": f"Declared state component `{vn}` in VARIABLES.",
        }
        for vn in var_names[:40]
    ]

    invs = guess_invariants(all_names, clean)

    decisions: list[str] = []
    if r"\E" in next_b:
        decisions.append("Uses \\E to model scheduling or choice of participating process/step.")
    if "UNCHANGED" in next_b:
        decisions.append("Uses UNCHANGED to stipulate frame conditions on shared variables.")
    if "EXCEPT" in tla_text:
        decisions.append("Uses EXCEPT updates for indexed state (functions/records).")

    tech: dict[str, Any] = {
        "algorithm": readme_title.strip() or f"TLA+ module {module_name}",
        "constants_and_processes": const_blob or "(no CONSTANTS block detected)",
        "variables": variables,
        "init": init_b or "(no Init definition matched)",
        "actions": actions,
        "next_and_fairness": (
            (next_b[:4500] if next_b else "(no Next definition matched)")
            + "\n---\nFairness / Spec fragment:\n"
            + (spec_b or "(no Spec definition matched)")
        ),
        "invariants_and_properties": invs,
        "critical_design_decisions": decisions
        or ["(No distinctive patterns flagged; review full .tla for design choices.)"],
    }

    # Narrative: avoid duplicating full harvest PDF block — user-facing prose only
    narrative = build_programmatic_narrative(
        readme_title=readme_title,
        module_name=module_name,
        header_excerpt=header_comment[:1200],
        var_names=var_names,
        next_body=next_b,
        harvest_prose_tail=harvest_prose,
        full_clean_text=clean,
    )

    return {"narrative": narrative, "technical": tech}


def merge_harvest_prose_into_narrative(structured: dict[str, Any], full_harvest_prose: str) -> dict[str, Any]:
    """
    Keep programmatic narrative as primary; prepend a short 'Source context' block
    so training still sees README/PDF strings without breaking the no-syntax narrative rule.
    """
    out = dict(structured)
    nar = out.get("narrative", "")
    ctx = full_harvest_prose.strip()
    if ctx and ctx not in nar:
        out["narrative"] = nar + "\n\n[Source context for reconstruction — may include references and excerpts:]\n" + ctx[:6000]
    return out
