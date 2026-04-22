#!/usr/bin/env python3
"""eval_canary_tla.py — compare base vs fine-tuned model on per-action TLC harnesses.

Usage:
    python -m scripts.eval_canary_tla
    python -m scripts.eval_canary_tla --model outputs/checkpoints_canary_tla
    python -m scripts.eval_canary_tla --compare  # base vs fine-tuned side by side
    python -m scripts.eval_canary_tla --n 20     # eval on first N harnesses only
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rlvr_canary.tla_dataset import load_tla_action_prompts
from src.rlvr_canary.tla_reward import _extract_next_body
from src.validators.per_action_tlc import validate_action

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FT_MODEL   = str(_REPO / "outputs" / "checkpoints_canary_tla")

TIMEOUT = int(os.getenv("CHATTLA_REWARD_TLC_TIMEOUT", "20"))



def load_model(model_path: str, device: str = "cuda"):
    print(f"[eval] loading {model_path} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def generate_next(model, tok, prompt_messages: list[dict], max_new_tokens: int = 300) -> str:
    text = tok.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — reproducible
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def score_one(harness, candidate: str) -> dict:
    # Apply same preprocessing as tla_reward.py: strip fences, <think>, etc.
    extracted = _extract_next_body(candidate) or ""
    result = validate_action(harness, extracted, timeout=TIMEOUT)
    # ActionResult fields: tier ("gold"/"silver"/"bronze"), sany_ok, tlc_ok, reward
    tlc_pass  = result.tlc_ok
    sany_pass = result.sany_ok
    reward    = result.reward  # 1.0 / 0.5 / 0.0
    if not candidate.strip():
        tier = "empty"
        reward = 0.0
    elif tlc_pass:
        tier = "tlc"
    elif sany_pass:
        tier = "sany"
    else:
        tier = "bronze"
    return {
        "tlc_pass": tlc_pass,
        "sany_pass": sany_pass,
        "reward": reward,
        "tier": tier,
    }


def eval_model(model, tok, examples, label: str) -> dict:
    results = []
    for i, ex in enumerate(examples):
        t0 = time.time()
        candidate = generate_next(model, tok, ex.prompt)
        score = score_one(ex.harness, candidate)
        elapsed = time.time() - t0
        score["prompt_id"] = ex.prompt_id
        score["elapsed_s"] = round(elapsed, 1)
        results.append(score)
        tier_sym = {"tlc": "✓", "sany": "S", "norm": "n", "emit": ".", "empty": "×"}.get(score["tier"], "?")
        print(f"  [{i+1:3d}/{len(examples)}] {tier_sym}  {ex.prompt_id[:40]:<40}  r={score['reward']:.2f}  {elapsed:.1f}s", flush=True)

    n = len(results)
    tlc  = sum(r["tlc_pass"]  for r in results)
    sany = sum(r["sany_pass"] for r in results)
    mean_r = sum(r["reward"] for r in results) / max(n, 1)

    print(f"\n[{label}] n={n}  TLC={tlc}/{n} ({tlc/n:.0%})  SANY={sany}/{n} ({sany/n:.0%})  "
          f"mean_reward={mean_r:.3f}")

    return {
        "label": label,
        "n": n,
        "tlc_pass": tlc,
        "sany_pass": sany,
        "mean_reward": mean_r,
        "per_example": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=FT_MODEL, help="Path or HF id of model to eval (default: fine-tuned)")
    ap.add_argument("--compare", action="store_true", help="Run base model too and compare")
    ap.add_argument("--n", type=int, default=None, help="Limit to first N harnesses")
    ap.add_argument("--corpus", default="data/processed/diamond_curated.jsonl")
    ap.add_argument("--out", default=None, help="Write JSON results to this path")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    examples = load_tla_action_prompts(args.corpus)
    if args.n:
        examples = examples[: args.n]
    print(f"[eval] {len(examples)} harnesses from {args.corpus}", flush=True)

    all_results = []

    if args.compare:
        base_model, base_tok = load_model(BASE_MODEL, args.device)
        print(f"\n=== BASE MODEL ({BASE_MODEL}) ===")
        base_res = eval_model(base_model, base_tok, examples, label="base")
        all_results.append(base_res)
        del base_model, base_tok
        torch.cuda.empty_cache()

    ft_model, ft_tok = load_model(args.model, args.device)
    label = "fine-tuned" if args.model == FT_MODEL else Path(args.model).name
    print(f"\n=== FINE-TUNED MODEL ({args.model}) ===")
    ft_res = eval_model(ft_model, ft_tok, examples, label=label)
    all_results.append(ft_res)

    if args.compare and len(all_results) == 2:
        base, ft = all_results
        delta_tlc  = ft["tlc_pass"]  - base["tlc_pass"]
        delta_sany = ft["sany_pass"] - base["sany_pass"]
        delta_r    = ft["mean_reward"] - base["mean_reward"]
        print(f"\n=== DELTA (fine-tuned - base) ===")
        print(f"  TLC  : {base['tlc_pass']}/{base['n']} → {ft['tlc_pass']}/{ft['n']}  ({delta_tlc:+d})")
        print(f"  SANY : {base['sany_pass']}/{base['n']} → {ft['sany_pass']}/{ft['n']}  ({delta_sany:+d})")
        print(f"  mean_reward: {base['mean_reward']:.3f} → {ft['mean_reward']:.3f}  ({delta_r:+.3f})")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
