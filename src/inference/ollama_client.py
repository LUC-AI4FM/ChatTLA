"""
ollama_client.py — Local Ollama inference client for ChatTLA.

Wraps the Ollama Python SDK to provide a clean API for TLA+ spec generation
using the fine-tuned ChatTLA model (or the base gpt-oss:20b for comparison).

The client always applies the gpt-oss harmony format via openai-harmony.
Without this the model produces degraded output.

Models
------
  chattla:20b      — Fine-tuned ChatTLA model (after convert_to_gguf.py)
  gpt-oss:20b      — Base model baseline

Reasoning levels
----------------
  low    — fast responses, suitable for interactive use
  medium — balanced; default for spec generation
  high   — deep analysis; use for complex distributed systems specs

Usage
-----
    from src.inference.ollama_client import ChatTLAClient

    client = ChatTLAClient()
    spec = client.generate_spec("A distributed read-write lock with N readers and 1 writer.")
    print(spec)

    # Async:
    import asyncio
    spec = asyncio.run(client.agenerate_spec("Two-phase commit protocol."))
"""

from __future__ import annotations

import os
import re
from typing import Optional

_OLLAMA_HOST   = os.getenv("OLLAMA_HOST",   "http://localhost:11434")
_DEFAULT_MODEL = os.getenv("CHATTLA_MODEL", "chattla:20b")

_DEVELOPER_PROMPT = """\
You are ChatTLA, Respond only the TLA+ module, no commentary
1. Start the module with ---- MODULE <ModuleName> ----
2. End with ====
3. Include EXTENDS, VARIABLES, Init, Next, and Spec operators
4. After the TLA+ module, append a TLC configuration block:
   SPECIFICATION Spec
   INVARIANT TypeOK   (if TypeOK is defined)
\
"""


class ChatTLAClient:
    """
    Ollama client for TLA+ spec generation via the locally hosted ChatTLA model.

    Parameters
    ----------
    model         : str   Ollama model tag (default: chattla:20b)
    host          : str   Ollama server host (default: http://localhost:11434)
    reasoning     : str   Harmony reasoning level: "low", "medium", or "high"
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host:  str = _OLLAMA_HOST,
        reasoning: str = "medium",
    ):
        import ollama  # lazy import

        self.model     = model
        self.reasoning = reasoning
        self._client   = ollama.Client(host=host)

    def generate_spec(
        self,
        nl_description: str,
        module_name: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """
        Generate a TLA+ specification from a natural-language description.

        Parameters
        ----------
        nl_description : str   Plain-English description of the system to model.
        module_name    : str   Desired module name (hinted to the model if provided).
        temperature    : float Sampling temperature; lower = more deterministic.

        Returns
        -------
        str   The extracted TLA+ module text (---- MODULE ... ====).
              Returns the raw output if delimiters cannot be found.
        """
        user_content = nl_description.strip()
        if module_name:
            user_content += f"\n\nUse module name: {module_name}"

        system_content = f"{_DEVELOPER_PROMPT}\nReasoning: {self.reasoning}"

        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_content},
            ],
            options={"temperature": temperature},
        )
        raw = response["message"]["content"]
        return _extract_tla(raw)

    def validate_and_generate(
        self,
        nl_description: str,
        max_retries: int = 3,
    ) -> tuple[str, str]:
        """
        Generate a spec and run TLC validation.  If TLC reports errors,
        feed the error back to the model for self-correction (up to max_retries).

        Returns
        -------
        (spec: str, tier: str)   Final spec text and validation tier ("gold"|"silver"|"bronze").
        """
        from src.validators.tlc_validator import validate_string

        spec = self.generate_spec(nl_description)
        for attempt in range(max_retries):
            m = re.search(r"----\s*MODULE\s+(\w+)", spec)
            module_name = m.group(1) if m else "Generated"
            result = validate_string(spec, module_name=module_name)

            if result.tier == "gold":
                return spec, "gold"
            if result.tier == "silver":
                return spec, "silver"

            # Bronze: feed TLC error back to model for correction
            error_summary = "\n".join(result.tlc_violations[:5])
            spec = self._self_correct(spec, error_summary)

        # Last attempt result
        m = re.search(r"----\s*MODULE\s+(\w+)", spec)
        module_name = m.group(1) if m else "Generated"
        result = validate_string(spec, module_name=module_name)
        return spec, result.tier

    def _self_correct(self, buggy_spec: str, error_msg: str) -> str:
        """Ask the model to fix a spec given a TLC error message."""
        system_content = f"{_DEVELOPER_PROMPT}\nReasoning: {self.reasoning}"
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": (
                        f"This TLA+ spec has errors:\n```\n{error_msg}\n```\n\n"
                        f"Buggy spec:\n{buggy_spec}\n\nFix the spec."
                    ),
                },
            ],
            options={"temperature": 0.1},
        )
        return _extract_tla(response["message"]["content"])


def _extract_tla(text: str) -> str:
    """Extract ---- MODULE ... ==== block from model output."""
    m = re.search(r"(----\s*MODULE\b.*?====)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Convenience CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a TLA+ spec via ChatTLA")
    parser.add_argument("description", help="Natural-language system description")
    parser.add_argument("--model",     default=_DEFAULT_MODEL)
    parser.add_argument("--reasoning", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--validate",  action="store_true", help="Run TLC validation after generation")
    args = parser.parse_args()

    client = ChatTLAClient(model=args.model, reasoning=args.reasoning)
    if args.validate:
        spec, tier = client.validate_and_generate(args.description)
        print(spec)
        print(f"\n[Validation tier: {tier}]")
    else:
        print(client.generate_spec(args.description))
