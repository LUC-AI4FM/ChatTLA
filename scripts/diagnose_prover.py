"""
diagnose_prover.py — Load the trained prover LoRA and inspect what it actually
generates on (a) a training example [memorization check] and (b) all eval
examples. Runs tlapm on each generation, prints the raw model output alongside
the verification result, and writes a JSON report.

This is the script to run when you need to understand WHY the held-out metrics
look the way they do, not just what the numbers are.
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
EVAL_JSONL = REPO / "data" / "processed" / "prover_eval.jsonl"
REPORT = REPO / "outputs" / "prover_diagnose.json"

_MODULE_RE = re.compile(r"MODULE\s+(\w+)")
_TLA_BLOCK_RE = re.compile(r"```tla\s*(.*?)\s*```", re.DOTALL)


def load_model():
    # AutoPeftModelForCausalLM loads base + adapter together and bypasses the
    # peft/accelerate get_balanced_memory code path that crashes on gpt-oss
    # (TypeError: unhashable type: 'set').
    print(f"[diagnose] loading adapter+base from {ADAPTER}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(ADAPTER),
        attn_implementation="eager",
        use_cache=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def extract_tla_from_user(user_content: str) -> str:
    m = _TLA_BLOCK_RE.search(user_content)
    return m.group(1).strip() if m else ""


def build_synthetic(preamble_plus_stmt: str, proof: str) -> tuple[str, str]:
    body = preamble_plus_stmt + "\n" + proof
    m = _MODULE_RE.search(body)
    orig = m.group(1) if m else "Generated"
    new_name = f"RTGen_{orig}"
    body = _MODULE_RE.sub(f"MODULE {new_name}", body, count=1)
    if not body.rstrip().endswith("===="):
        body = body.rstrip() + "\n" + ("=" * 78) + "\n"
    return body, new_name


def strip_to_proof(text: str) -> str:
    """Mirror of what TLAPSEvalCallback does: strip analysis channel, take from
    first <n> bullet."""
    if "final" in text:
        text = text[text.index("final") + len("final"):]
    m = re.search(r"(<\d+>.*)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


@torch.no_grad()
def generate(model, tokenizer, example: dict, max_new_tokens: int = 1024) -> tuple[str, str]:
    """Return (raw_text, extracted_proof)."""
    msgs = [m for m in example["messages"] if m["role"] in ("developer", "user")]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    new = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new, skip_special_tokens=True)
    return raw, strip_to_proof(raw)


def run_one(model, tokenizer, example: dict, tag: str) -> dict:
    user = next(m["content"] for m in example["messages"] if m["role"] == "user")
    preamble_stmt = extract_tla_from_user(user)
    if not preamble_stmt:
        return {"tag": tag, "error": "no tla block in user content"}

    gold_proof = next(
        (m["content"] for m in example["messages"]
         if m["role"] == "assistant" and m.get("channel") == "final"),
        "",
    )

    raw, proof = generate(model, tokenizer, example)

    synth, module_name = build_synthetic(preamble_stmt, proof)
    try:
        r = validate_string(synth, module_name=module_name, timeout=60)
        vres = {
            "tier": r.tier,
            "proved": r.obligations_proved,
            "total": r.obligations_total,
            "failed": r.obligations_failed,
            "errors": r.errors[:2],
        }
    except Exception as e:
        vres = {"tier": "exception", "error": str(e)}

    return {
        "tag": tag,
        "source_file": example.get("_source_file"),
        "theorem_line": example.get("_theorem_line"),
        "gold_proof_preview": gold_proof[:300],
        "gold_obligations": example.get("_obligations_proved"),
        "raw_generation_len": len(raw),
        "raw_generation_preview": raw[:800],
        "extracted_proof_preview": proof[:800],
        "verification": vres,
    }


def main():
    model, tokenizer = load_model()

    train_rows = [json.loads(l) for l in TRAIN_JSONL.open()]
    eval_rows = [json.loads(l) for l in EVAL_JSONL.open()]

    # Memorization sanity: take 2 train examples
    train_probe = train_rows[:2]

    results = {"train_memorization": [], "eval_heldout": []}

    print(f"[diagnose] MEMORIZATION check — 2 train examples")
    for i, ex in enumerate(train_probe):
        r = run_one(model, tokenizer, ex, tag=f"train[{i}]")
        results["train_memorization"].append(r)
        v = r["verification"]
        print(f"  [{i}] {r['source_file']}:L{r['theorem_line']}  tier={v.get('tier')} proved={v.get('proved','?')}/{v.get('total','?')}  raw_len={r['raw_generation_len']}")

    print(f"\n[diagnose] HELD-OUT eval — {len(eval_rows)} examples")
    for i, ex in enumerate(eval_rows):
        r = run_one(model, tokenizer, ex, tag=f"eval[{i}]")
        results["eval_heldout"].append(r)
        v = r["verification"]
        print(f"  [{i}] {r['source_file']}:L{r['theorem_line']}  tier={v.get('tier')} proved={v.get('proved','?')}/{v.get('total','?')}  raw_len={r['raw_generation_len']}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2))
    print(f"\n[diagnose] wrote {REPORT}")

    # Aggregate
    def agg(rs):
        n = len(rs)
        parse = sum(1 for r in rs if r["verification"].get("tier") != "parse_error")
        any_p = sum(1 for r in rs if (r["verification"].get("proved") or 0) > 0)
        full = sum(1 for r in rs
                   if (r["verification"].get("proved") or 0) >= (r.get("gold_obligations") or 1))
        return f"parse={parse}/{n} any_proved={any_p}/{n} full_proved={full}/{n}"
    print(f"[diagnose] train: {agg(results['train_memorization'])}")
    print(f"[diagnose] eval:  {agg(results['eval_heldout'])}")


if __name__ == "__main__":
    main()
