"""RLVR validation canary.

Per `project_rlvr_validation_plan.md`, we don't trust LoRA + RL on a hard
target like TLA+ until we've validated the RL stack on a simple, well-known
binary-verifiable task. This package is that canary:

  Phase 1 (here): full FT, Llama-3.2-1B-Instruct, GSM8K, GRPO, binary reward
  Phase 2: same setup + LoRA — confirm performance is preserved
  Phase 3: swap GSM8K → TLA+ spec generation with the per-action TLC reward
           from src.validators.per_action_tlc

The canary is intentionally narrow. If it doesn't reach the published GRPO
baseline on GSM8K (~50%+ pass@1 from a 1B base after a few hundred steps),
the RL stack itself is broken and there is no point adding LoRA or TLA+ on
top. See the kalomaze "RL Learning with LoRA" blog post (memory:
reference_lora_rl_blogs.md) for the gotchas this is meant to flush out.
"""

from .gsm8k_dataset import extract_gold_answer, load_gsm8k_prompts
from .reward import binary_correctness_reward, extract_model_answer

__all__ = [
    "binary_correctness_reward",
    "extract_gold_answer",
    "extract_model_answer",
    "load_gsm8k_prompts",
]
