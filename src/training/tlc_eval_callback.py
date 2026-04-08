"""
tlc_eval_callback.py — TLC-based evaluation callback for SFTTrainer.

Standard language model evaluation (perplexity, loss) is insufficient for
TLA+ fine-tuning because a low-loss model can still generate syntactically
invalid or semantically broken specs.  This callback runs TLC model checking
on generated outputs at every evaluation step, providing two hard metrics:

Metrics logged to MLflow
------------------------
tlc/sany_parse_rate         — fraction of generated specs that SANY accepts
tlc/tlc_clean_rate          — fraction that TLC model-checks with no violations
tlc/eval_count              — number of specs evaluated in this step
tlc/init_present_rate       — fraction whose AST contains an Init operator
tlc/next_present_rate       — fraction whose AST contains a Next operator
tlc/init_level_ok_rate      — Init exists AND is state-level (no primes)
tlc/next_level_ok_rate      — Next exists AND is action-level (has primes)
tlc/invariants_decl_rate    — at least one invariant-shaped operator declared
tlc/depth1_rate             — TLC depth-1 reachability check passes
tlc/partial_credit_mean     — weighted mean ∈ [0,1] across all components

`tlc_clean_rate` remains the headline metric. `partial_credit_mean` is the
denser proxy that moves earlier in training (DeepSeek-Prover-V2-style binary
verifier feedback is too sparse for ChatTLA's scale; partial credit gives the
gradient something to climb on near-misses).

Implementation notes
--------------------
- We sample n_samples prompts from the eval set at each eval step.
  This is deterministic (seeded sample) for reproducibility.
- We use the spec_generation task format only: NL description → TLA+ spec.
  This is the most end-to-end task.
- We cap TLC runtime to 30s per spec to keep eval under ~5 min total.
- The callback does NOT affect training loss or gradients.

Research note
-------------
This pattern — using a domain-specific verifier as a training signal —
is the key differentiator of ChatTLA vs generic code fine-tuning.  Future
work could close the loop by adding TLC feedback as a reward signal in RLHF.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TLCEvalCallback(TrainerCallback):
    """
    At each evaluation step:
    1. Sample n_samples examples from eval_dataset (spec_generation task)
    2. Generate TLA+ specs using the current model checkpoint
    3. Run SANY + TLC on each generated spec
    4. Log sany_parse_rate and tlc_clean_rate to MLflow
    """

    def __init__(
        self,
        eval_dataset,
        tokenizer,
        n_samples: int = 50,
        max_new_tokens: int = 1024,
        tlc_timeout: int = 30,
        seed: int = 42,
    ):
        self.eval_dataset   = eval_dataset
        self.tokenizer      = tokenizer
        self.n_samples      = n_samples
        self.max_new_tokens = max_new_tokens
        self.tlc_timeout    = tlc_timeout
        self.seed           = seed
        self._eval_prompts: list[dict] | None = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs: Any,
    ) -> None:
        if model is None:
            return

        print(f"\n[TLCEvalCallback] Running TLC eval at step {state.global_step}...")

        prompts = self._get_eval_prompts()
        if not prompts:
            print("[TLCEvalCallback] No spec_generation examples found in eval set; skipping.")
            return

        n_sany_pass = 0
        n_tlc_pass  = 0
        n_total     = 0
        # Component sums for the denser per-step reward signal
        n_init_present     = 0
        n_next_present     = 0
        n_init_level_ok    = 0
        n_next_level_ok    = 0
        n_invariants_decl  = 0
        n_depth1_pass      = 0
        partial_credit_sum = 0.0

        model.eval()
        with torch.no_grad():
            for prompt_msgs in prompts:
                user_content = self._extract_user_content(prompt_msgs)
                if not user_content:
                    continue

                generated = self._generate(model, user_content)
                if not generated:
                    n_total += 1
                    continue

                n_total += 1
                try:
                    tier, semantic = self._run_tlc(generated)
                except Exception as exc:
                    print(f"[TLCEvalCallback] TLC error: {exc}")
                    tier, semantic = "bronze", None

                if tier in ("silver", "gold"):
                    n_sany_pass += 1
                if tier == "gold":
                    n_tlc_pass += 1
                if semantic is not None:
                    n_init_present    += int(semantic.init_present)
                    n_next_present    += int(semantic.next_present)
                    n_init_level_ok   += int(semantic.init_level_ok)
                    n_next_level_ok   += int(semantic.next_level_ok)
                    n_invariants_decl += int(semantic.invariants_declared)
                    n_depth1_pass     += int(semantic.tlc_depth1_ok)
                    partial_credit_sum += semantic.partial_credit

        denom = n_total if n_total else 1
        sany_rate = n_sany_pass / denom
        tlc_rate  = n_tlc_pass  / denom

        mlflow.log_metrics(
            {
                "tlc/sany_parse_rate":      sany_rate,
                "tlc/tlc_clean_rate":       tlc_rate,
                "tlc/eval_count":           n_total,
                "tlc/init_present_rate":    n_init_present     / denom,
                "tlc/next_present_rate":    n_next_present     / denom,
                "tlc/init_level_ok_rate":   n_init_level_ok    / denom,
                "tlc/next_level_ok_rate":   n_next_level_ok    / denom,
                "tlc/invariants_decl_rate": n_invariants_decl  / denom,
                "tlc/depth1_rate":          n_depth1_pass      / denom,
                "tlc/partial_credit_mean":  partial_credit_sum / denom,
            },
            step=state.global_step,
        )

        print(
            f"[TLCEvalCallback] step={state.global_step} | "
            f"sany={sany_rate:.3f} | tlc={tlc_rate:.3f} | "
            f"depth1={n_depth1_pass / denom:.3f} | "
            f"partial={partial_credit_sum / denom:.3f} | n={n_total}"
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_eval_prompts(self) -> list[list[dict]]:
        """Cache and return up to n_samples spec_generation prompts from eval set."""
        if self._eval_prompts is not None:
            return self._eval_prompts

        import random
        rng = random.Random(self.seed)
        spec_gen_examples: list[list[dict]] = []

        for example in self.eval_dataset:
            msgs = example.get("messages", [])
            # Spec-generation examples have the user message asking to "Write a TLA+"
            user_msgs = [m for m in msgs if m.get("role") == "user"]
            if user_msgs and "Write a TLA+" in user_msgs[0].get("content", ""):
                spec_gen_examples.append(msgs)

        rng.shuffle(spec_gen_examples)
        self._eval_prompts = spec_gen_examples[: self.n_samples]
        return self._eval_prompts

    def _extract_user_content(self, messages: list[dict]) -> str:
        """Extract the user prompt text from a messages list."""
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _generate(self, model, user_content: str) -> str:
        """Generate a TLA+ spec from a user prompt."""
        # Use the same harmony chat template that SFT training uses, so eval
        # measures performance on the actual input distribution (not a bare prompt).
        from src.training.dataset_builder import _DEVELOPER_PROMPT

        messages = [
            {"role": "developer", "content": _DEVELOPER_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback if chat template not available on this tokenizer
            prompt = f"Write a TLA+ specification for:\n{user_content}\n\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # Decode only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception as exc:
            print(f"[TLCEvalCallback] Generation error: {exc}")
            return ""

    def _run_tlc(self, generated_text: str):
        """
        Extract TLA+ spec from generated text and run TLC validation.
        Returns (tier, SemanticInfo|None). The semantic carries per-component
        verdicts that the eval loop aggregates into the partial_credit metric.
        """
        tla_content = _extract_tla_block(generated_text)
        if not tla_content:
            return "bronze", None

        from src.validators.tlc_validator import validate_string
        m = re.search(r"----\s*MODULE\s+(\w+)", tla_content)
        module_name = m.group(1) if m else "Generated"

        result = validate_string(
            tla_content,
            module_name=module_name,
            timeout=self.tlc_timeout,
        )
        return result.tier, result.semantic


def _extract_tla_block(text: str) -> str:
    """
    Extract a normalized TLA+ module block from generated text.

    Runs the canonical post-processor (src.postprocess.normalize_spec) which
    closes the five FormaLLM hallucination categories — Unicode operators,
    semicolon/backtick injection, <think>/harmony leakage, missing/duplicate
    MODULE headers, and missing ==== terminators — before SANY/TLC see the
    output. Falls back to returning the full text if normalization fails.
    """
    try:
        from src.postprocess import normalize_spec
        cleaned, _ = normalize_spec(text)
        return cleaned.strip() if cleaned else text.strip()
    except Exception:
        m = re.search(r"(----\s*MODULE\b.*?====)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()
