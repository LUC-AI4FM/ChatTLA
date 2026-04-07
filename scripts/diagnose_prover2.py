"""
diagnose_prover2.py — Second-round diagnosis: try to get the trained prover
adapter to actually emit its final-channel proof, by (a) increasing the
decode budget and (b) toggling the reasoning-effort instruction in the
developer prompt. Runs on two train examples (memorization check) across
four variants and reports which combination recovers the proof.

Variants:
  A: Reasoning:medium (as trained),  max_new_tokens=1024  (control)
  B: Reasoning:medium,               max_new_tokens=4096
  C: Reasoning:low,                  max_new_tokens=1024
  D: Reasoning:low,                  max_new_tokens=4096
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from src.validators.tlaps_validator import validate_string

ADAPTER = REPO / "outputs" / "checkpoints_prover" / "checkpoint-48"
BASE_MODEL = "openai/gpt-oss-20b"
TRAIN_JSONL = REPO / "data" / "processed" / "prover_train.jsonl"
REPORT = REPO / "outputs" / "prover_diagnose2.json"

_MODULE_RE = re.compile(r"MODULE\s+(\w+)")
_TLA_BLOCK_RE = re.compile(r"```tla\s*(.*?)\s*```", re.DOTALL)


def load_model():
    print(f"[diag2] loading from {ADAPTER}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(ADAPTER),
        attn_implementation="eager",
        use_cache=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def strip_to_proof(text: str) -> str:
    # Harmony channels render inline. Find the last 'final' marker, take after.
    # Fall back to first <n> bullet if no 'final' marker.
    idx = text.rfind("final")
    if idx >= 0:
        text = text[idx + len("final"):]
    m = re.search(r"(<\d+>.*)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def build_synthetic(preamble_stmt: str, proof: str) -> tuple[str, str]:
    body = preamble_stmt + "\n" + proof
    m = _MODULE_RE.search(body)
    orig = m.group(1) if m else "Generated"
    new_name = f"RTGen_{orig}"
    body = _MODULE_RE.sub(f"MODULE {new_name}", body, count=1)
    if not body.rstrip().endswith("===="):
        body = body.rstrip() + "\n" + ("=" * 78) + "\n"
    return body, new_name


@torch.no_grad()
def gen(model, tok, msgs, max_new: int) -> str:
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new, do_sample=False, temperature=None,
        pad_token_id=tok.pad_token_id,
    )
    new = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True)


def set_reasoning(dev_content: str, level: str) -> str:
    return re.sub(r"Reasoning:\s*\w+", f"Reasoning: {level}", dev_content)


def run_variant(model, tok, ex, label, reasoning, max_new):
    user_msg = next(m for m in ex["messages"] if m["role"] == "user")
    dev_msg = next(m for m in ex["messages"] if m["role"] == "developer")
    msgs = [
        {"role": "developer", "content": set_reasoning(dev_msg["content"], reasoning)},
        {"role": "user", "content": user_msg["content"]},
    ]
    raw = gen(model, tok, msgs, max_new)
    proof = strip_to_proof(raw)
    ptla = _TLA_BLOCK_RE.search(user_msg["content"])
    synth, name = build_synthetic(ptla.group(1).strip(), proof)
    try:
        r = validate_string(synth, module_name=name, timeout=60)
        vres = {"tier": r.tier, "proved": r.obligations_proved, "total": r.obligations_total}
    except Exception as e:
        vres = {"tier": "exception", "error": str(e)}
    has_final = "final" in raw
    has_bullet = bool(re.search(r"<\d+>", proof))
    return {
        "label": label,
        "reasoning": reasoning,
        "max_new": max_new,
        "raw_len": len(raw),
        "has_final_marker": has_final,
        "proof_has_step_bullet": has_bullet,
        "proof_preview": proof[:500],
        "raw_tail": raw[-500:],
        "verify": vres,
    }


def main():
    model, tok = load_model()
    train_rows = [json.loads(l) for l in TRAIN_JSONL.open()][:2]
    variants = [
        ("A_med_1024", "medium", 1024),
        ("B_med_4096", "medium", 4096),
        ("C_low_1024", "low", 1024),
        ("D_low_4096", "low", 4096),
    ]
    out = []
    for i, ex in enumerate(train_rows):
        print(f"\n=== ex[{i}]  {ex.get('_source_file')}:L{ex.get('_theorem_line')} ===")
        gold_obligations = ex.get("_obligations_proved")
        print(f"  gold obligations: {gold_obligations}")
        for label, r, mn in variants:
            res = run_variant(model, tok, ex, label, r, mn)
            v = res["verify"]
            print(f"  {label}: raw_len={res['raw_len']:>5} has_final={res['has_final_marker']} has_bullet={res['proof_has_step_bullet']} tier={v.get('tier')} proved={v.get('proved','?')}/{v.get('total','?')}")
            out.append({"ex_idx": i, "source_file": ex.get("_source_file"), "theorem_line": ex.get("_theorem_line"), "gold_obligations": gold_obligations, **res})

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2))
    print(f"\n[diag2] wrote {REPORT}")


if __name__ == "__main__":
    main()
