"""Deterministic post-processing for LLM-generated TLA+ specs.

The FormaLLM evaluation (docs/formallm.md) identified five recurring
hallucination categories that account for a large fraction of SANY failures.
All five are deterministically fixable before the parser is invoked. This
module is the single source of truth for that normalization so that inference,
training-time eval, and dataset curation all behave identically.
"""

from .normalize import (
    NormalizationReport,
    extract_module_block,
    normalize_spec,
    strip_reasoning_artifacts,
)

__all__ = [
    "NormalizationReport",
    "extract_module_block",
    "normalize_spec",
    "strip_reasoning_artifacts",
]
