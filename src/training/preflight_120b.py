"""120b fine-tune preflight — run on a Sophia login/compute node BEFORE qsub.

Per the HPC preflight checklist: dry-import under the exact job env, verify
every blocker fix, and prove the LoRA adapter attaches to gpt-oss-120b with
nonzero expert coverage — without launching a full train.

    python preflight_120b.py --model /path/to/gpt-oss-120b [--meta-only]

--meta-only builds the model on the meta device (no weights read) so it runs
on a login node in seconds; drop it on a compute node for the real load.

Checks (all must PASS; exits nonzero on the first failure):
  1. imports: torch / transformers / peft / accelerate / datasets import clean
  2. config: model config loads, is MoE (num_local_experts>1)
  3. lora: resolve_lora_config picks MoE branch, get_peft_model attaches,
     trainable%% >= 0.1 floor, expert target_parameters present in the
     trainable parameter names
  4. no_hardcode: train.py no longer sets CUDA_VISIBLE_DEVICES defaults,
     no unconditional Mxfp4Config, no use_cache kwarg passed to from_pretrained
  5. pbs: every *.pbs job script propagates TRAIN_EXIT (no `exit 0` after it)
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path


def ok(name, msg=""):
    print(f"PASS {name} {msg}")


def fail(name, msg):
    print(f"FAIL {name}: {msg}")
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-dir", default=".", help="dir containing train.py and *.pbs")
    ap.add_argument("--meta-only", action="store_true")
    args = ap.parse_args()

    # 1. imports
    try:
        import torch, transformers, peft, accelerate, datasets  # noqa: F401
    except Exception as e:
        fail("imports", repr(e))
    ok("imports", f"torch {torch.__version__}, transformers {transformers.__version__}, peft {peft.__version__}")

    # 2. config
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    from lora_resolver import is_moe
    if not is_moe(cfg):
        fail("config", f"{args.model} is not MoE (num_local_experts missing) — wrong model path?")
    ok("config", f"MoE, num_local_experts={getattr(cfg, 'num_local_experts', '?')}")

    # 3. lora attach
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model
    from lora_resolver import resolve_lora_config, assert_trainable_floor
    yaml_cfg = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "bias": "none"}
    lora_cfg = resolve_lora_config(cfg, yaml_cfg)
    if not lora_cfg.target_parameters:
        fail("lora", "MoE model but resolver produced no target_parameters")
    if args.meta_only:
        import torch
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="bfloat16", device_map="cpu")
    model = get_peft_model(model, lora_cfg)
    pct = assert_trainable_floor(model)  # exits 3 below floor
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not any("experts" in n for n in trainable_names):
        fail("lora", "no expert tensors in trainable set — target_parameters did not match")
    ok("lora", f"trainable {pct:.4f}%, expert coverage confirmed "
               f"({sum('experts' in n for n in trainable_names)} expert tensors)")

    # 4. hardcode scan on train.py
    tp = Path(args.train_dir) / "train.py"
    if not tp.exists():
        hits = glob.glob(str(Path(args.train_dir) / "**" / "train.py"), recursive=True)
        if not hits:
            fail("no_hardcode", f"train.py not found under {args.train_dir}")
        tp = Path(hits[0])
    src = tp.read_text()
    problems = []
    if re.search(r"setdefault\(\s*['\"]CUDA_VISIBLE_DEVICES", src):
        problems.append("CUDA_VISIBLE_DEVICES setdefault still present")
    if re.search(r"Mxfp4Config", src) and not re.search(r"if\b[^\n]*(mxfp4|quant|moe|gpt.oss)", src, re.I):
        problems.append("Mxfp4Config applied unconditionally")
    if re.search(r"from_pretrained\([^)]*use_cache", src, re.S):
        problems.append("use_cache passed as from_pretrained kwarg")
    if problems:
        fail("no_hardcode", f"{tp}: " + "; ".join(problems))
    ok("no_hardcode", str(tp))

    # 5. pbs exit propagation
    bad = []
    for pbs in glob.glob(str(Path(args.train_dir) / "**" / "*.pbs"), recursive=True):
        text = Path(pbs).read_text()
        if "TRAIN_EXIT" in text:
            after = text.split("TRAIN_EXIT")[-1]
            if re.search(r"^\s*exit\s+0\s*$", after, re.M) and not re.search(r"exit\s+\$\{?TRAIN_EXIT", after):
                bad.append(pbs)
    if bad:
        fail("pbs", f"exit 0 after TRAIN_EXIT in: {bad}")
    ok("pbs", "all job scripts propagate TRAIN_EXIT")

    print("\nALL PREFLIGHT CHECKS PASSED")


if __name__ == "__main__":
    main()
