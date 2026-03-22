"""
Extract structured description from a TLA+ file using SANY's built-in XMLExporter.

Calls:  java -cp tla2tools.jar tla2sany.xml.XMLExporter -I <dir> <file.tla>
Parses the XML AST and returns the dataset description schema (narrative + technical).

No custom Java code — uses the toolchain that ships with tla2tools.jar.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TLA_TOOLS_JAR = _REPO_ROOT / "src" / "shared" / "tlc" / "tla2tools.jar"

# OpDeclNode kind constants (from tla2sany source)
_KIND_CONSTANT = "2"
_KIND_VARIABLE = "3"

# Semantic level constants
_LEVEL_CONSTANT = "0"
_LEVEL_STATE = "1"
_LEVEL_ACTION = "2"
_LEVEL_TEMPORAL = "3"

_LEVEL_LABELS = {
    _LEVEL_CONSTANT: "constant",
    _LEVEL_STATE: "state",
    _LEVEL_ACTION: "action",
    _LEVEL_TEMPORAL: "temporal",
}


def _clean_comment(text: str) -> str:
    """Strip (* ... *) wrappers and excess whitespace from SANY pre-comments."""
    if not text:
        return ""
    text = re.sub(r"^\(\*+\s*|\s*\*+\)$", "", text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SanyXmlResult:
    """Parsed SANY XML output for one module."""

    def __init__(self, root_module: str):
        self.root_module = root_module
        self.constants: list[dict[str, str]] = []
        self.variables: list[dict[str, str]] = []
        self.operators: list[dict[str, Any]] = []
        self.assumes: list[str] = []
        self.extends: list[str] = []

    def find_op(self, name: str) -> Optional[dict[str, Any]]:
        for op in self.operators:
            if op["name"] == name:
                return op
        return None

    def action_ops(self) -> list[dict[str, Any]]:
        return [o for o in self.operators if o.get("level") == _LEVEL_ACTION]

    def temporal_ops(self) -> list[dict[str, Any]]:
        return [o for o in self.operators if o.get("level") == _LEVEL_TEMPORAL]

    def state_ops(self) -> list[dict[str, Any]]:
        return [o for o in self.operators if o.get("level") == _LEVEL_STATE]


def run_sany_xml(
    tla_path: Path,
    jar: Path = _TLA_TOOLS_JAR,
    include_dirs: Optional[list[Path]] = None,
    timeout: float = 60.0,
) -> Optional[str]:
    """Run SANY XMLExporter, return XML string or None on failure."""
    if not jar.exists():
        return None

    cmd = ["java", "-cp", str(jar), "tla2sany.xml.XMLExporter"]
    dirs = set()
    dirs.add(str(tla_path.parent))
    if include_dirs:
        for d in include_dirs:
            dirs.add(str(d))
    for d in sorted(dirs):
        cmd += ["-I", d]
    cmd.append(str(tla_path))

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 and not r.stdout.strip():
            return None
        return r.stdout
    except (subprocess.TimeoutExpired, OSError):
        return None


def run_sany_xml_from_string(
    tla_content: str,
    module_name: str,
    jar: Path = _TLA_TOOLS_JAR,
    include_dirs: Optional[list[Path]] = None,
    timeout: float = 60.0,
) -> Optional[str]:
    """Write TLA+ to a temp file and run SANY XMLExporter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / f"{module_name}.tla"
        p.write_text(tla_content, encoding="utf-8")
        extra = list(include_dirs or [])
        extra.append(Path(tmpdir))
        return run_sany_xml(p, jar=jar, include_dirs=extra, timeout=timeout)


def _collect_body_text(elem: ET.Element, uid_map: dict[str, str], depth: int = 0) -> str:
    """
    Recursively walk an AST subtree (body of a UserDefinedOpKind) and produce
    a compact textual representation. Not perfect TLA+ pretty-printing but
    captures the essential structure for training descriptions.
    """
    if depth > 30:
        return "..."

    tag = elem.tag

    if tag == "NumeralNode":
        return elem.findtext("IntValue", "0")

    if tag == "StringNode":
        val = elem.findtext("StringValue", "")
        return f'"{val}"'

    if tag == "OpApplNode":
        op_elem = elem.find("operator")
        op_name = ""
        if op_elem is not None:
            for child in op_elem:
                uid_ref = child.findtext("UID")
                if uid_ref and uid_ref in uid_map:
                    op_name = uid_map[uid_ref]
                    break
                # inline name
                nm = child.findtext("uniquename")
                if nm:
                    op_name = nm
                    break

        operands_elem = elem.find("operands")
        operand_texts = []
        if operands_elem is not None:
            for child in operands_elem:
                operand_texts.append(_collect_body_text(child, uid_map, depth + 1))

        # Bound quantifiers (\E, \A)
        bounds = elem.findall("boundSymbols/bound")
        if bounds:
            bound_parts = []
            for b in bounds:
                bnames = []
                for fpn in b.findall("FormalParamNode"):
                    bnames.append(fpn.findtext("uniquename", "?"))
                for fpnr in b.findall("FormalParamNodeRef"):
                    uid = fpnr.findtext("UID", "")
                    bnames.append(uid_map.get(uid, "?"))
                bound_parts.append(", ".join(bnames))
            bstr = ", ".join(bound_parts)
            if op_name.startswith("$Bounded"):
                real_op = op_name.replace("$Bounded", "\\")
                return f"{real_op} {bstr} : {' '.join(operand_texts)}"

        # Infix binary operators (common built-ins)
        _INFIX = {
            "$Eq": "=", "=": "=",
            "$Neq": "/=", "/=": "/=",
            "$Lt": "<", "<": "<",
            "$Gt": ">", ">": ">",
            "$Leq": "<=", "=<": "<=", "<=": "<=", "\\leq": "<=",
            "$Geq": ">=", ">=": ">=", "\\geq": ">=",
            "$Plus": "+", "+": "+",
            "$Minus": "-", "-": "-",
            "$Times": "*", "*": "*",
            "$Div": "\\div", "\\div": "\\div",
            "$Mod": "%", "%": "%",
            "$DotDot": "..", "..": "..",
            "$In": "\\in", "\\in": "\\in",
            "$NotIn": "\\notin", "\\notin": "\\notin",
            "$Implies": "=>", "=>": "=>",
            "$Equiv": "<=>", "<=>": "<=>",
            "$Subseteq": "\\subseteq", "\\subseteq": "\\subseteq",
            "$Supset": "\\supset",
            "$Cup": "\\cup", "\\cup": "\\cup",
            "$Cap": "\\cap", "\\cap": "\\cap",
            "$Setdiff": "\\", "\\": "\\",
            "$Circ": "\\circ", "\\circ": "\\circ",
            "$Conj": "/\\", "$Disj": "\\/",
            "$Append": "\\o", "\\o": "\\o",
            "$Concat": "\\o",
        }

        if op_name in _INFIX and len(operand_texts) == 2:
            return f"{operand_texts[0]} {_INFIX[op_name]} {operand_texts[1]}"
        # Single-char / symbol built-ins that map to known TLA+ syntax
        if op_name == "'" and len(operand_texts) == 1:
            return f"{operand_texts[0]}'"
        if op_name == "UNCHANGED" and operand_texts:
            return f"UNCHANGED {operand_texts[0]}"
        if op_name == "\\land" and len(operand_texts) == 2:
            return f"{operand_texts[0]} /\\ {operand_texts[1]}"
        if op_name == "\\lor" and len(operand_texts) == 2:
            return f"{operand_texts[0]} \\/ {operand_texts[1]}"
        if op_name == "[]" and len(operand_texts) == 1:
            return f"[]({operand_texts[0]})"
        if op_name == "<>" and len(operand_texts) == 1:
            return f"<>({operand_texts[0]})"
        if op_name == "~" and len(operand_texts) == 1:
            return f"~({operand_texts[0]})"
        if op_name in ("TRUE", "FALSE"):
            return op_name

        if op_name == "$ConjList":
            return " /\\ ".join(operand_texts)
        if op_name == "$DisjList":
            return " \\/ ".join(operand_texts)
        if op_name == "$SetEnumerate":
            return "{" + ", ".join(operand_texts) + "}"
        if op_name == "$Tuple":
            return "<<" + ", ".join(operand_texts) + ">>"
        if op_name == "$SetOfFcns":
            return f"[{operand_texts[0]} -> {operand_texts[1]}]" if len(operand_texts) == 2 else str(operand_texts)
        if op_name == "$FcnApply":
            return f"{operand_texts[0]}[{operand_texts[1]}]" if len(operand_texts) == 2 else str(operand_texts)
        if op_name == "$Except":
            return f"[{operand_texts[0]} EXCEPT {' '.join(operand_texts[1:])}]" if operand_texts else "EXCEPT"
        if op_name == "$Pair":
            return f"({', '.join(operand_texts)})"
        if op_name == "$RcdConstructor":
            pairs = []
            for i in range(0, len(operand_texts) - 1, 2):
                pairs.append(f"{operand_texts[i]} |-> {operand_texts[i+1]}")
            return "[" + ", ".join(pairs) + "]"
        if op_name == "$RcdSelect":
            return f"{operand_texts[0]}.{operand_texts[1]}" if len(operand_texts) == 2 else str(operand_texts)
        if op_name == "$FcnConstructor":
            return f"[{' '.join(operand_texts)}]" if operand_texts else "[]"
        if op_name == "$SetOfRcds":
            pairs = []
            for i in range(0, len(operand_texts) - 1, 2):
                pairs.append(f"{operand_texts[i]}: {operand_texts[i+1]}")
            return "[" + ", ".join(pairs) + "]"
        if op_name == "$SquareAct":
            return f"[{operand_texts[0]}]_{operand_texts[1]}" if len(operand_texts) == 2 else str(operand_texts)
        if op_name == "$WF":
            return f"WF_{operand_texts[0]}({operand_texts[1]})" if len(operand_texts) == 2 else "WF_..."
        if op_name == "$SF":
            return f"SF_{operand_texts[0]}({operand_texts[1]})" if len(operand_texts) == 2 else "SF_..."
        if op_name == "$TemporalForall" or op_name == "$TemporalExists":
            return f"{op_name.replace('$', '')}({', '.join(operand_texts)})"
        if op_name == "$UnchangedOp" or op_name == "$Unchanged":
            return f"UNCHANGED {operand_texts[0]}" if operand_texts else "UNCHANGED"
        if op_name == "$Prime":
            return f"{operand_texts[0]}'" if operand_texts else "'"
        if op_name == "$Negate" or op_name == "$LogicalNot":
            return f"~({operand_texts[0]})" if operand_texts else "~"
        if op_name == "$Cardinality":
            return f"Cardinality({operand_texts[0]})" if operand_texts else "Cardinality(?)"
        if op_name == "$DOMAIN":
            return f"DOMAIN {operand_texts[0]}" if operand_texts else "DOMAIN"
        if op_name == "$UNION":
            return f"UNION {operand_texts[0]}" if operand_texts else "UNION"
        if op_name == "$SUBSET":
            return f"SUBSET {operand_texts[0]}" if operand_texts else "SUBSET"
        if op_name == "$SubsetOf":
            return f"{{{operand_texts[0]} : ...}}" if operand_texts else "{...}"
        if op_name == "$SetOfAll":
            return f"{{{' '.join(operand_texts)}}}" if operand_texts else "{...}"
        if op_name == "$Case":
            return "CASE " + " [] ".join(operand_texts)
        if op_name == "$IfThenElse":
            if len(operand_texts) == 3:
                return f"IF {operand_texts[0]} THEN {operand_texts[1]} ELSE {operand_texts[2]}"
        if op_name.startswith("$"):
            clean = op_name.lstrip("$")
            if operand_texts:
                if len(operand_texts) == 2:
                    return f"{operand_texts[0]} {clean} {operand_texts[1]}"
                return f"{clean}({', '.join(operand_texts)})"
            return clean

        # User-defined operator
        if operand_texts:
            return f"{op_name}({', '.join(operand_texts)})"
        return op_name

    if tag == "FormalParamNode" or tag == "FormalParamNodeRef":
        uid = elem.findtext("UID", "")
        return uid_map.get(uid, elem.findtext("uniquename", "?"))

    if tag == "OpDeclNodeRef":
        uid = elem.findtext("UID", "")
        return uid_map.get(uid, "?")

    # LetInNode
    if tag == "LetInNode":
        body_el = elem.find("body")
        if body_el is not None and len(body_el):
            return "LET ... IN " + _collect_body_text(body_el[0], uid_map, depth + 1)

    # SubstInNode
    if tag == "SubstInNode":
        body_el = elem.find("body")
        if body_el is not None and len(body_el):
            return _collect_body_text(body_el[0], uid_map, depth + 1)

    # Fallback: try recursing children
    parts = []
    for child in elem:
        t = _collect_body_text(child, uid_map, depth + 1)
        if t:
            parts.append(t)
    return " ".join(parts) if parts else ""


def parse_sany_xml(xml_str: str, root_module_name: str) -> SanyXmlResult:
    """Parse SANY XML output into SanyXmlResult."""
    root = ET.fromstring(xml_str)
    result = SanyXmlResult(root_module_name)

    # Build UID -> name map for cross-references
    uid_map: dict[str, str] = {}
    for entry in root.findall(".//context/entry"):
        uid = entry.findtext("UID", "")
        for child in entry:
            nm = child.findtext("uniquename")
            if nm and uid:
                uid_map[uid] = nm

    # OpDeclNode: CONSTANTS (kind=2) and VARIABLES (kind=3)
    for entry in root.findall(".//context/entry"):
        decl = entry.find("OpDeclNode")
        if decl is None:
            continue
        name = decl.findtext("uniquename", "")
        kind = decl.findtext("kind", "")
        level = decl.findtext("level", "")
        if kind == _KIND_CONSTANT:
            result.constants.append({"name": name, "kind": "CONSTANT", "level": level})
        elif kind == _KIND_VARIABLE:
            result.variables.append({"name": name, "kind": "VARIABLE", "level": level})

    # UserDefinedOpKind: operator definitions
    for entry in root.findall(".//context/entry"):
        opdef = entry.find("UserDefinedOpKind")
        if opdef is None:
            continue
        name = opdef.findtext("uniquename", "")
        arity = opdef.findtext("arity", "0")
        level = opdef.findtext("level", "")
        fname_el = opdef.find(".//location/filename")
        fname = fname_el.text if fname_el is not None else ""
        pre_comment = _clean_comment(opdef.findtext("pre-comments", "") or "")
        line_el = opdef.find(".//location/line/begin")
        line = int(line_el.text) if line_el is not None and line_el.text else 0

        # Only keep definitions from the root module
        if fname != root_module_name:
            continue

        params = []
        for fpn in opdef.findall(".//params/FormalParamNode"):
            pname = fpn.findtext("uniquename", "")
            if pname and not pname.startswith("Formal_"):
                params.append(pname)

        # Extract body text
        body_el = opdef.find("body")
        body_text = ""
        if body_el is not None and len(body_el) > 0:
            body_text = _collect_body_text(body_el[0], uid_map)

        result.operators.append({
            "name": name,
            "arity": int(arity),
            "level": level,
            "level_label": _LEVEL_LABELS.get(level, "unknown"),
            "params": params,
            "comment": pre_comment,
            "body": body_text,
            "line": line,
        })

    # Sort operators by line number
    result.operators.sort(key=lambda o: o.get("line", 0))
    return result


def sany_result_to_description(
    sr: SanyXmlResult,
    *,
    module_name: str,
    readme_title: str = "",
    header_comment: str = "",
    harvest_prose: str = "",
) -> dict[str, Any]:
    """Convert SanyXmlResult into our dataset description schema."""

    # --- variables ---
    variables = [
        {
            "name": v["name"],
            "type": f"[VARIABLE, {_LEVEL_LABELS.get(v['level'], 'state')}-level]",
            "role": "State variable declared in the module.",
        }
        for v in sr.variables
    ]

    # --- constants ---
    const_names = [c["name"] for c in sr.constants]
    const_str = ", ".join(const_names) if const_names else "(no CONSTANTS declared)"

    # --- Init ---
    init_op = sr.find_op("Init") or sr.find_op("Init0")
    init_body = ""
    if init_op:
        init_body = init_op["body"][:3500] if init_op["body"] else "(Init defined but body not reconstructed)"
    else:
        # fallback: any op starting with Init
        for op in sr.operators:
            if op["name"].startswith("Init") and op["body"]:
                init_body = f"{op['name']} == {op['body'][:2500]}"
                break
    if not init_body:
        init_body = "(no Init definition found by SANY)"

    # --- Next / actions ---
    next_op = sr.find_op("Next") or sr.find_op("next")
    next_body = ""
    if next_op:
        next_body = next_op["body"][:4500] if next_op["body"] else "(Next defined but body not reconstructed)"

    # Spec / fairness
    spec_op = sr.find_op("Spec") or sr.find_op("FairSpec") or sr.find_op("LiveSpec")
    spec_body = ""
    if spec_op:
        spec_body = spec_op["body"][:2500]
    # fairness tokens
    all_bodies = " ".join(op["body"] for op in sr.operators if op["body"])
    fairness_parts = []
    if "WF_" in all_bodies or "$WF" in all_bodies:
        fairness_parts.append("Weak fairness (WF_) on action(s).")
    if "SF_" in all_bodies or "$SF" in all_bodies:
        fairness_parts.append("Strong fairness (SF_) on action(s).")
    fairness_note = " ".join(fairness_parts) or "No WF_/SF_ fairness detected."

    next_fairness = (
        (next_body or "(no Next definition found)")
        + "\n---\nSpec / fairness:\n"
        + (spec_body or "(no Spec definition found)")
        + "\n" + fairness_note
    )

    # --- actions (level=2 "action-level" operators, excluding Next itself) ---
    skip_names = {"Init", "Next", "Spec", "FairSpec", "LiveSpec", "vars", "Init0", "Init1"}
    actions = []
    for op in sr.action_ops():
        if op["name"] in skip_names:
            continue
        pstr = ", ".join(op["params"]) if op["params"] else ""
        sig = f"{op['name']}({pstr})" if pstr else op["name"]
        intent = op["comment"] or f"Action-level operator (transitions involving primed variables)."
        actions.append({
            "name": sig,
            "intent": intent,
            "pre": "[see body]",
            "post": op["body"][:800] if op["body"] else "[body not reconstructed]",
        })

    # --- invariants (state-level ops that look like TypeOK / Inv / Safety) ---
    inv_pattern = re.compile(r"Type|Inv|Safety|Invariant|Constraint", re.I)
    invariants = []
    for op in sr.state_ops():
        if inv_pattern.search(op["name"]):
            invariants.append({
                "name": op["name"],
                "assertion": op["body"][:1200] if op["body"] else "[defined]",
                "purpose": op["comment"] or "Type/safety invariant (state-level property).",
            })
    # Also temporal properties (level 3)
    prop_pattern = re.compile(r"Ltl|Liveness|Property|Live|Fairness", re.I)
    for op in sr.temporal_ops():
        if op["name"] in skip_names:
            continue
        if prop_pattern.search(op["name"]) or "Ltl" in op["name"]:
            invariants.append({
                "name": op["name"],
                "assertion": op["body"][:800] if op["body"] else "[defined]",
                "purpose": op["comment"] or "Temporal/liveness property.",
            })

    # --- design decisions ---
    decisions = []
    if any(v for v in sr.variables):
        decisions.append(f"Models {len(sr.variables)} state variables and {len(sr.constants)} constants.")
    if len(sr.action_ops()) > 3:
        decisions.append(f"Decomposes Next into {len(actions)} named action operators (modular transition structure).")
    if "UNCHANGED" in all_bodies:
        decisions.append("Uses UNCHANGED for frame conditions on non-affected variables.")
    if "EXCEPT" in all_bodies:
        decisions.append("Uses EXCEPT for functional/record updates (indexed state).")
    if "\\E" in all_bodies or "$BoundedExists" in all_bodies or "$Bounded" in all_bodies:
        decisions.append("Uses existential quantification to model nondeterministic choice (e.g. process scheduling).")

    # --- narrative ---
    title = readme_title.strip() or f"the {module_name} module"
    var_list = ", ".join(v["name"] for v in sr.variables[:8])
    n_sentences = [f"This specification formalizes {title} as a TLA+ module."]
    if header_comment:
        n_sentences.append(f"From the authors: {header_comment[:500]}")
    if var_list:
        extra = " among others" if len(sr.variables) > 8 else ""
        n_sentences.append(f"The state space is organized around variables {var_list}{extra}.")
    n_sentences.append(
        f"SANY analysis identifies {len(sr.action_ops())} action-level operators "
        f"and {len(sr.temporal_ops())} temporal-level operators."
    )
    n_sentences.append(fairness_note)

    narrative = " ".join(n_sentences)
    if harvest_prose and harvest_prose.strip() not in narrative:
        narrative += "\n\n[Source context for reconstruction:]\n" + harvest_prose.strip()[:6000]

    # --- algorithm ---
    algo = readme_title.strip() or f"TLA+ module {module_name}"

    return {
        "narrative": narrative,
        "technical": {
            "algorithm": algo,
            "constants_and_processes": const_str,
            "variables": variables,
            "init": init_body,
            "actions": actions[:30],
            "next_and_fairness": next_fairness,
            "invariants_and_properties": invariants[:15],
            "critical_design_decisions": decisions or ["(No distinctive patterns flagged by SANY analysis.)"],
        },
    }


def extract_with_sany(
    tla_path: Path,
    *,
    module_name: str,
    readme_title: str = "",
    header_comment: str = "",
    harvest_prose: str = "",
    jar: Path = _TLA_TOOLS_JAR,
    include_dirs: Optional[list[Path]] = None,
) -> tuple[Optional[dict[str, Any]], str]:
    """
    Full pipeline: run SANY XMLExporter -> parse XML -> produce description dict.

    Returns (description_dict_or_none, status_string).
    """
    xml_str = run_sany_xml(tla_path, jar=jar, include_dirs=include_dirs)
    if not xml_str or not xml_str.strip():
        return None, "sany_failed:no_xml"

    try:
        sr = parse_sany_xml(xml_str, module_name)
    except ET.ParseError as e:
        return None, f"xml_parse_error:{e}"

    if not sr.variables and not sr.operators:
        return None, "sany_empty:no_vars_or_ops"

    desc = sany_result_to_description(
        sr,
        module_name=module_name,
        readme_title=readme_title,
        header_comment=header_comment,
        harvest_prose=harvest_prose,
    )
    n_act = len([o for o in sr.operators if o.get("level") == _LEVEL_ACTION])
    status = f"ok:vars={len(sr.variables)},consts={len(sr.constants)},ops={len(sr.operators)},actions={n_act}"
    return desc, status
