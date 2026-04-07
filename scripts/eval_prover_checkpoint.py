"""
eval_prover_checkpoint.py — Manually run the prover model on the eval set
and dump generated proofs + tlapm verdicts so we can actually see what's
happening.

Usage:
    python scripts/eval_prover_checkpoint.py \
        --checkpoint outputs/checkpoints_prover/checkpoint-48 \
        --out outputs/prover_eval_dump.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.validators.tlaps_validator import validate_string  # noqa: E402

EVAL_JSONL = REPO / "data" / "processed" / "prover_eval.jsonl"
BASE_MODEL = "openai/gpt-oss-20b"

MODULE_RE = re.compile(r"MODULE\s+(\w+)")
TLA_BLOCK_RE = re.compile(r"```tla\s*(.*?)\s*```", re.DOTALL)


def build_synthetic(preamble_plus_stmt: str, proof: str) -> tuple[str, str]:
    body = preamble_plus_stmt + "\n" + proof
    m = MODULE_RE.search(body)
    orig = m.group(1) if m else "Generated"
    new_name = f"RTGen_{orig}"
    body = MODULE_RE.sub(f"MODULE {new_name}", body, count=1)
    if not body.rstrip().endswith("===="):
        body = body.rstrip() + "\n" + ("=" * 78) + "\n"
    return body, new_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=str(REPO / "outputs" / "prover_eval_dump.json"))
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in EVAL_JSONL.open()]
    print(f"[eval] {len(rows)} eval examples")

    print(f"[eval] loading base {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        attn_implementation="eager",
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"[eval] loading adapter {args.checkpoint}")
    model = PeftModel.from_pretrained(base, args.checkpoint)
    model.eval()

    dump = []
    summary = {"n": 0, "parse": 0, "any": 0, "full": 0, "sum_proved": 0}

    for i, ex in enumerate(rows, 1):
        msgs = [m for m in ex["messages"] if m["role"] in ("developer", "user")]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Strip harmony tags / <think> / fences via the canonical normalizer.
        try:
            from src.postprocess import strip_reasoning_artifacts, NormalizationReport
            proof_text = strip_reasoning_artifacts(raw, NormalizationReport())
        except Exception:
            proof_text = raw
        if "final" in proof_text:
            proof_text = proof_text[proof_text.index("final") + len("final"):]
        m = re.search(r"(<\d+>.*)", proof_text, re.DOTALL)
        if m:
            proof = m.group(1).strip()
        else:
            proof = proof_text.strip()

        user_text = next(m for m in ex["messages"] if m["role"] == "user")["content"]
        tla_block_m = TLA_BLOCK_RE.search(user_text)
        preamble = tla_block_m.group(1).strip() if tla_block_m else ""

        synth, mod_name = build_synthetic(preamble, proof)
        try:
            res = validate_string(synth, module_name=mod_name, timeout=60)
            tier = res.tier
            proved = res.obligations_proved
            total = res.obligations_total
        except Exception as e:
            tier, proved, total = f"exception:{e}", 0, 0

        gold_total = int(ex.get("_obligations_total") or 0)
        summary["n"] += 1
        if tier != "parse_error":
            summary["parse"] += 1
        if proved > 0:
            summary["any"] += 1
        if gold_total > 0 and proved >= gold_total:
            summary["full"] += 1
        summary["sum_proved"] += proved

        print(f"[{i}] {ex.get('_source_file','?')}:L{ex.get('_theorem_line','?')}  "
              f"tier={tier}  proved={proved}/{total}  (gold {ex.get('_obligations_proved','?')}/{gold_total})")
        dump.append({
            "source_file": ex.get("_source_file"),
            "theorem_line": ex.get("_theorem_line"),
            "gold_proved": ex.get("_obligations_proved"),
            "gold_total": gold_total,
            "raw_generation": raw,
            "extracted_proof": proof,
            "synthetic_module": synth,
            "tier": tier,
            "proved": proved,
            "total": total,
        })

    n = max(summary["n"], 1)
    print("\n[eval] summary:")
    print(f"  parse_rate: {summary['parse']/n:.2f}")
    print(f"  any_proved: {summary['any']/n:.2f}")
    print(f"  full_proved: {summary['full']/n:.2f}")
    print(f"  avg_obligations: {summary['sum_proved']/n:.1f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": dump}, indent=2))
    print(f"[eval] dumped {out_path}")


if __name__ == "__main__":
    main()
