"""
component_validator.py — Per-component verification signals for TLA+ specs.

The default tier signal (gold/silver/bronze) is binary at the spec level: either
TLC accepts the whole module or it does not. That collapses every kind of
near-miss into the same bucket and gives the training loop nothing to climb.

This module produces a denser, partial-credit signal by inspecting individual
spec components:

  init_present        — Init operator exists in the AST
  next_present        — Next operator exists in the AST
  init_level_ok       — Init is state-level (SANY level 1) — i.e. no primes
  next_level_ok       — Next is action-level (SANY level 2) — i.e. has primes
  invariants_declared — at least one invariant-shaped operator exists
  tlc_depth1_ok       — TLC accepts the spec at depth 1 (one-step reachability)
  tlc_full_ok         — TLC accepts the spec without bound (the legacy gold gate)

Each verdict is a bool. partial_credit is a weighted mean in [0, 1] used as a
proxy reward for the eval callback and as a bucketing key for the RL loop's
rejection-sampling pool.

The AST inspection is free (one SANY XMLExporter call) — it reuses the
existing parser in scripts/tla_description_sources/sany_extract.py rather than
building probe modules. The TLC depth-1 run is the only added I/O.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

# sany_extract lives under scripts/, not src/, so add it to sys.path on first
# use. It has no third-party deps beyond the stdlib + tla2tools.jar.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SANY_EXTRACT_DIR = _REPO_ROOT / "scripts" / "tla_description_sources"
if str(_SANY_EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_SANY_EXTRACT_DIR))

# Imported lazily inside functions so unit tests that don't touch SANY don't
# fail at import time if the script directory layout changes.

_DEFAULT_JAR = _REPO_ROOT / "src" / "shared" / "tlc" / "tla2tools.jar"

# Component weights for partial_credit. They sum to 1.0 and reflect the rough
# difficulty ladder: parsing < structure < depth-1 reachability < full TLC.
_WEIGHTS = {
    "init_present":        0.05,
    "next_present":        0.05,
    "init_level_ok":       0.10,
    "next_level_ok":       0.10,
    "invariants_declared": 0.10,
    "tlc_depth1_ok":       0.25,
    "tlc_full_ok":         0.35,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9


@dataclass
class ComponentVerdicts:
    init_present: bool = False
    next_present: bool = False
    init_level_ok: bool = False
    next_level_ok: bool = False
    invariants_declared: bool = False
    tlc_depth1_ok: bool = False
    tlc_full_ok: bool = False
    partial_credit: float = 0.0
    # Names from the AST (handy for plan extraction and debugging)
    invariant_names: list[str] = field(default_factory=list)
    action_names: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def recompute_credit(self) -> None:
        score = 0.0
        for field_name, weight in _WEIGHTS.items():
            if getattr(self, field_name):
                score += weight
        self.partial_credit = round(score, 4)


# ──────────────────────────── AST inspection ────────────────────────────

# SANY level codes (mirror sany_extract._LEVEL_*)
_LEVEL_STATE = "1"
_LEVEL_ACTION = "2"

# Names that are conventionally invariants/safety properties
_INV_PATTERN = re.compile(r"Type|Inv|Safety|Invariant|Constraint|Consistent|Correct", re.I)


def _ast_verdicts(
    tla_content: str,
    module_name: str,
    jar: Path,
) -> tuple[ComponentVerdicts, Any]:
    """Run SANY XMLExporter once and derive AST-level verdicts.

    Returns (verdicts, sany_result_or_None). The sany_result is returned so the
    caller can reuse it for plan_from_ast without paying for a second parse.
    """
    from sany_extract import (  # type: ignore
        run_sany_xml_from_string,
        parse_sany_xml,
    )

    v = ComponentVerdicts()
    xml_str = run_sany_xml_from_string(tla_content, module_name, jar=jar, timeout=30.0)
    if not xml_str:
        v.notes = "sany_xml_failed"
        return v, None

    try:
        sr = parse_sany_xml(xml_str, module_name)
    except Exception as e:  # pragma: no cover
        v.notes = f"sany_xml_parse_error:{e}"
        return v, None

    init_op = sr.find_op("Init")
    next_op = sr.find_op("Next")
    v.init_present = init_op is not None
    v.next_present = next_op is not None
    v.init_level_ok = bool(init_op and init_op.get("level") == _LEVEL_STATE)
    # Next is allowed to be either action-level (preferred) or higher
    v.next_level_ok = bool(next_op and next_op.get("level") in (_LEVEL_ACTION, "3"))

    # Action operator names (excluding Init/Next themselves)
    skip = {"Init", "Next", "Spec", "FairSpec", "LiveSpec", "vars"}
    v.action_names = [
        op["name"] for op in sr.action_ops() if op["name"] not in skip
    ]

    # Invariant-shaped state-level operators
    inv_names: list[str] = []
    for op in sr.state_ops():
        if _INV_PATTERN.search(op["name"]):
            inv_names.append(op["name"])
    v.invariant_names = inv_names
    v.invariants_declared = len(inv_names) > 0

    return v, sr


# ──────────────────────────── TLC depth-1 probe ────────────────────────────

def _tlc_depth1(
    tla_content: str,
    cfg_content: str,
    module_name: str,
    jar: Path,
    timeout: int = 15,
) -> bool:
    """Run TLC with depth bound 1 (one-step reachability) and return success.

    Depth-1 is a much weaker check than the full model-check: it only verifies
    that Init produces a non-empty initial state set and that Next can fire at
    least once without violating any declared invariant. But it's a meaningful
    intermediate signal — most syntactically-valid junk specs fail it because
    Init is unsatisfiable or Next has type errors at the action level.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tla_path = tmp / f"{module_name}.tla"
        cfg_path = tmp / f"{module_name}.cfg"
        tla_path.write_text(tla_content, encoding="utf-8")
        cfg_path.write_text(cfg_content, encoding="utf-8")

        cmd = [
            "java", "-cp", str(jar),
            "tlc2.TLC",
            "-config", str(cfg_path),
            "-dfid", "1",
            str(tla_path),
        ]
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=tmp,
            )
            stdout = r.stdout + r.stderr
            # Success heuristic: TLC reported "no error" OR generated >0 states
            # without an error/violation line. Depth-1 runs are short enough
            # that timeouts are real failures.
            if re.search(r"no error has been found", stdout, re.IGNORECASE):
                return True
            if re.search(r"(Error|violation|is violated)", stdout, re.IGNORECASE):
                return False
            # Fallback: if any state was generated and no error, call it ok
            return bool(re.search(r"\d+\s+states?\s+generated", stdout))
        except (subprocess.TimeoutExpired, OSError):
            return False


# ──────────────────────────── public entry point ────────────────────────────

def validate_components(
    tla_content: str,
    cfg_content: Optional[str] = None,
    module_name: Optional[str] = None,
    jar: Path = _DEFAULT_JAR,
    full_tlc_passed: Optional[bool] = None,
    run_depth1: bool = True,
    depth1_timeout: int = 15,
) -> ComponentVerdicts:
    """Compute per-component verdicts for a TLA+ spec.

    Parameters
    ----------
    tla_content      : str            The .tla source.
    cfg_content      : str | None     TLC config; if None, depth-1 probe is skipped.
    module_name      : str | None     Inferred from "MODULE <name>" if not given.
    jar              : Path           tla2tools.jar.
    full_tlc_passed  : bool | None    If the caller already ran full TLC, pass
                                       the result here to populate tlc_full_ok
                                       without re-running.
    run_depth1       : bool           Whether to run the depth-1 probe.
    depth1_timeout   : int            Seconds before depth-1 is killed (call it failed).
    """
    if module_name is None:
        m = re.search(r"MODULE\s+(\w+)", tla_content)
        module_name = m.group(1) if m else "Temp"

    verdicts, _ = _ast_verdicts(tla_content, module_name, jar)

    if run_depth1 and cfg_content and verdicts.init_present and verdicts.next_present:
        verdicts.tlc_depth1_ok = _tlc_depth1(
            tla_content, cfg_content, module_name, jar, timeout=depth1_timeout,
        )

    if full_tlc_passed is not None:
        verdicts.tlc_full_ok = bool(full_tlc_passed)
        # If full TLC passed, depth-1 trivially passes too (saves a probe call
        # for callers who skipped run_depth1).
        if verdicts.tlc_full_ok:
            verdicts.tlc_depth1_ok = True

    verdicts.recompute_credit()
    return verdicts


# ──────────────────────────── plan extraction ────────────────────────────

def plan_from_ast(
    tla_content: str,
    module_name: Optional[str] = None,
    jar: Path = _DEFAULT_JAR,
) -> Optional["SpecPlan"]:
    """Reverse-engineer a SpecPlan from a (parseable) TLA+ spec.

    Used to retrofit existing gold specs with structured plans for
    plan-then-spec training data, without an additional LLM curation pass.
    """
    from src.shared.schemas.spec_plan import SpecPlan, NextAction, PlannedInvariant

    if module_name is None:
        m = re.search(r"MODULE\s+(\w+)", tla_content)
        module_name = m.group(1) if m else "Temp"

    from sany_extract import run_sany_xml_from_string, parse_sany_xml  # type: ignore

    xml_str = run_sany_xml_from_string(tla_content, module_name, jar=jar, timeout=30.0)
    if not xml_str:
        return None
    try:
        sr = parse_sany_xml(xml_str, module_name)
    except Exception:
        return None

    # EXTENDS — pulled from raw text since SANY XML doesn't expose it cleanly
    extends: list[str] = []
    em = re.search(r"^\s*EXTENDS\s+(.+)$", tla_content, re.MULTILINE)
    if em:
        extends = [s.strip() for s in em.group(1).split(",") if s.strip()]

    constants = [c["name"] for c in sr.constants]
    variables = [v["name"] for v in sr.variables]

    init_op = sr.find_op("Init")
    init_sketch = ""
    if init_op and init_op.get("body"):
        # Truncate body for sketch — full body lives in the spec
        body = init_op["body"]
        init_sketch = (body[:240] + "...") if len(body) > 240 else body

    skip = {"Init", "Next", "Spec", "FairSpec", "LiveSpec", "vars"}
    next_actions: list[NextAction] = []
    for op in sr.action_ops():
        if op["name"] in skip:
            continue
        body = op.get("body", "") or ""
        next_actions.append(NextAction(
            name=op["name"],
            guard="",  # NL guards are best-effort; leave empty for AST extraction
            effect=(body[:160] + "...") if len(body) > 160 else body,
        ))

    invariants: list[PlannedInvariant] = []
    for op in sr.state_ops():
        if _INV_PATTERN.search(op["name"]):
            kind = "type" if "Type" in op["name"] else "safety"
            body = op.get("body", "") or ""
            invariants.append(PlannedInvariant(
                name=op["name"],
                statement=(body[:160] + "...") if len(body) > 160 else body,
                kind=kind,  # type: ignore[arg-type]
            ))
    for op in sr.temporal_ops():
        if op["name"] in skip:
            continue
        if re.search(r"Live|Liveness|Property|Ltl|Eventually", op["name"], re.I):
            body = op.get("body", "") or ""
            invariants.append(PlannedInvariant(
                name=op["name"],
                statement=(body[:160] + "...") if len(body) > 160 else body,
                kind="liveness",
            ))

    fairness = ""
    all_bodies = " ".join(op.get("body", "") or "" for op in sr.operators)
    if "WF_" in all_bodies or "$WF" in all_bodies:
        fairness = "weak fairness"
    if "SF_" in all_bodies or "$SF" in all_bodies:
        fairness = (fairness + " + strong fairness").strip(" +")

    return SpecPlan(
        module_name=module_name,
        extends=extends,
        constants=constants,
        variables=variables,
        init_sketch=init_sketch,
        next_actions=next_actions,
        invariants=invariants,
        fairness=fairness,
    )


# ──────────────────────────── reward convenience ────────────────────────────

def reward_from_spec(
    tla_content: str,
    module_name: Optional[str] = None,
    jar: Path = _DEFAULT_JAR,
    run_depth1: bool = True,
    run_full_tlc: bool = True,
    depth1_timeout: int = 15,
    full_tlc_timeout: int = 30,
) -> float:
    """One-call reward: normalize → auto-cfg → validate_components → partial_credit.

    Designed for use in GRPO reward functions where the caller has a raw TLA+
    string and needs a float in [0, 1].

    Parameters
    ----------
    tla_content       : Raw TLA+ spec text (may include fences, <think>, Unicode).
    module_name       : Inferred from MODULE header if None.
    run_depth1        : Whether to run TLC depth-1 probe (adds ~15s worst case).
    run_full_tlc      : Whether to run full TLC model check (adds ~30s worst case).
    full_tlc_timeout  : Seconds for full TLC.

    Returns partial_credit ∈ [0, 1].
    """
    from src.postprocess.normalize import normalize_spec
    from src.validators.tlc_validator import _autogenerate_cfg, validate_string

    if not tla_content or not tla_content.strip():
        return 0.0

    # Normalize: strip fences, <think>, Unicode→ASCII
    try:
        tla_content, _report = normalize_spec(tla_content)
    except Exception:
        return 0.0

    if module_name is None:
        m = re.search(r"MODULE\s+(\w+)", tla_content)
        module_name = m.group(1) if m else "Temp"

    # Auto-generate .cfg from spec heuristics
    cfg_content = _autogenerate_cfg(tla_content)

    # Optional full TLC pass (to populate tlc_full_ok)
    full_tlc_passed = None
    if run_full_tlc and cfg_content:
        try:
            tlc_result = validate_string(
                tla_content, cfg_content=cfg_content,
                module_name=module_name, jar=jar, timeout=full_tlc_timeout,
            )
            full_tlc_passed = tlc_result.tier == "gold"
        except Exception:
            full_tlc_passed = False

    verdicts = validate_components(
        tla_content,
        cfg_content=cfg_content,
        module_name=module_name,
        jar=jar,
        full_tlc_passed=full_tlc_passed,
        run_depth1=run_depth1,
        depth1_timeout=depth1_timeout,
    )
    return verdicts.partial_credit
