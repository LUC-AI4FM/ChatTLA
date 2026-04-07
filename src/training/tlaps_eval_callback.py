"""
tlaps_eval_callback.py — TLAPS-based evaluation callback for the prover SFT.

Sibling of tlc_eval_callback.py, but for proof generation instead of spec
generation. At each evaluation step, it samples prover examples from the
eval set, generates proofs with the current model, splices them back into
the original module, and runs tlapm.

Metrics logged to MLflow
------------------------
tlaps/parse_rate     — fraction of generated proofs whose synthetic module
                       parses (i.e. tlapm did not return parse_error)
tlaps/any_proved     — fraction of generated proofs that discharge >=1
                       obligation
tlaps/full_proved    — fraction of generated proofs that match-or-exceed the
                       original obligation count for that theorem (hard win)
tlaps/avg_obligations_proved — mean #obligations discharged per generated proof
tlaps/eval_count     — number of proofs evaluated this step

The primary research metric is `tlaps/any_proved`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mlflow
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.validators.tlaps_validator import validate_string

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Same MODULE-rename trick the round-trip script uses to keep the synthetic
# module from colliding with on-disk files of the same name.
_MODULE_RE = re.compile(r"MODULE\s+(\w+)")
_TLA_BLOCK_RE = re.compile(r"```tla\s*(.*?)\s*```", re.DOTALL)


class TLAPSEvalCallback(TrainerCallback):
    """
    At each evaluation step:
    1. Sample n_samples prover examples from eval_dataset
    2. Generate the proof body with the current model checkpoint
    3. Splice (preamble + statement + generated_proof + ====) into a synthetic
       module and run tlapm via validate_string
    4. Log parse_rate / any_proved / full_proved / avg_obligations to MLflow
    """

    def __init__(
        self,
        eval_dataset,
        tokenizer,
        n_samples: int = 6,
        max_new_tokens: int = 1024,
        tlapm_timeout: int = 60,
        seed: int = 42,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.max_new_tokens = max_new_tokens
        self.tlapm_timeout = tlapm_timeout
        self.seed = seed
        self._eval_examples: list[dict] | None = None

    # ----------------------------------------------------------------------

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs: Any,
    ) -> None:
        # Full runs have eval_strategy="no" (vocab cross_entropy OOM during
        # eval-mode forward). Run TLAPS eval once at the very end so we get
        # at least one measurement of the final adapter.
        self._run_eval(state, model, label="train_end")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs: Any,
    ) -> None:
        self._run_eval(state, model, label="evaluate")

    def _run_eval(self, state: TrainerState, model, label: str) -> None:
        if model is None:
            return

        examples = self._get_eval_examples()
        if not examples:
            print("[TLAPSEvalCallback] No prover examples found in eval set; skipping.")
            return

        print(f"\n[TLAPSEvalCallback/{label}] step={state.global_step} on {len(examples)} examples...")

        n_total = 0
        n_parsed = 0
        n_any = 0
        n_full = 0
        sum_proved = 0

        model.eval()
        with torch.no_grad():
            for ex in examples:
                user_content = self._extract_user_content(ex)
                if not user_content:
                    continue

                preamble_plus_stmt = self._extract_tla_from_user(user_content)
                if not preamble_plus_stmt:
                    continue

                generated_proof = self._generate(model, ex)
                if not generated_proof:
                    n_total += 1
                    continue

                n_total += 1
                synth, module_name = self._build_synthetic(preamble_plus_stmt, generated_proof)
                try:
                    result = validate_string(
                        synth,
                        module_name=module_name,
                        timeout=self.tlapm_timeout,
                    )
                except Exception as exc:
                    print(f"[TLAPSEvalCallback] validator error: {exc}")
                    continue

                if result.tier != "parse_error":
                    n_parsed += 1
                if result.obligations_proved > 0:
                    n_any += 1
                sum_proved += result.obligations_proved

                gold_total = int(ex.get("_obligations_total") or 0)
                if gold_total > 0 and result.obligations_proved >= gold_total:
                    n_full += 1

        denom = max(n_total, 1)
        metrics = {
            "tlaps/parse_rate": n_parsed / denom,
            "tlaps/any_proved": n_any / denom,
            "tlaps/full_proved": n_full / denom,
            "tlaps/avg_obligations_proved": sum_proved / denom,
            "tlaps/eval_count": n_total,
        }
        mlflow.log_metrics(metrics, step=state.global_step)

        print(
            f"[TLAPSEvalCallback] step={state.global_step} | "
            f"parse={metrics['tlaps/parse_rate']:.2f} "
            f"any={metrics['tlaps/any_proved']:.2f} "
            f"full={metrics['tlaps/full_proved']:.2f} "
            f"avg_obs={metrics['tlaps/avg_obligations_proved']:.1f} "
            f"n={n_total}"
        )

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------

    def _get_eval_examples(self) -> list[dict]:
        if self._eval_examples is not None:
            return self._eval_examples
        import random
        rng = random.Random(self.seed)
        prover_rows: list[dict] = []
        for ex in self.eval_dataset:
            msgs = ex.get("messages", [])
            user = next((m for m in msgs if m.get("role") == "user"), None)
            if user and "TLAPS proof" in user.get("content", ""):
                prover_rows.append(ex)
        rng.shuffle(prover_rows)
        self._eval_examples = prover_rows[: self.n_samples]
        return self._eval_examples

    def _extract_user_content(self, ex: dict) -> str:
        for m in ex.get("messages", []):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    def _extract_tla_from_user(self, user_content: str) -> str:
        """The user message contains a ```tla ... ``` fenced block with
        preamble + theorem statement. Extract just that block."""
        m = _TLA_BLOCK_RE.search(user_content)
        return m.group(1).strip() if m else ""

    def _generate(self, model, example: dict) -> str:
        # Build a prompt from developer + user only (assistant is what we generate)
        msgs = [m for m in example.get("messages", []) if m.get("role") in ("developer", "user")]
        try:
            prompt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt = self._extract_user_content(example)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception as exc:
            print(f"[TLAPSEvalCallback] generation error: {exc}")
            return ""

        # Strip harmony tags, <think> blocks, and markdown fences via the
        # canonical normalizer (Unicode op replacement also applies — useful
        # for proof bullets that quote ∧/∨).
        try:
            from src.postprocess import strip_reasoning_artifacts, NormalizationReport
            text = strip_reasoning_artifacts(text, NormalizationReport())
        except Exception:
            pass
        # Strip harmony "analysis" channel prose. Channels render inline as
        # `analysis...final...`, so peel off everything up to and including
        # "final" before extracting the first <n> bullet.
        if "final" in text:
            text = text[text.index("final") + len("final"):]
        m = re.search(r"(<\d+>.*)", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    def _build_synthetic(self, preamble_plus_stmt: str, proof: str) -> tuple[str, str]:
        body = preamble_plus_stmt + "\n" + proof
        m = _MODULE_RE.search(body)
        orig = m.group(1) if m else "Generated"
        new_name = f"RTGen_{orig}"
        body = _MODULE_RE.sub(f"MODULE {new_name}", body, count=1)
        if not body.rstrip().endswith("===="):
            body = body.rstrip() + "\n" + ("=" * 78) + "\n"
        return body, new_name
