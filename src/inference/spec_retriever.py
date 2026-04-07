"""Lightweight BM25-style retriever over verified TLA+ specs.

The FormaLLM discussion (docs/formallm.md §"Hallucinations Are Systematic
and Mitigable") points to RAG over verified specs as a way to ground operator
usage in concrete examples — directly addressing the "phantom operator"
hallucination class. This is the operator-level analogue of package
hallucination in code generation.

Why a BM25 token index instead of embeddings:
  - The query side is structured (NL description + maybe a partial spec).
  - The corpus is small (hundreds of specs), so the index fits in memory.
  - We want to retrieve by *technical vocabulary* ("mutex", "leader",
    "consensus", "EXCEPT", "bag") not paraphrase similarity. Token IDF
    weights that exact-match vocabulary directly.
  - Zero extra dependencies on top of stdlib + numpy/jsonl that the rest
    of the project already uses.

Usage:
    from src.inference.spec_retriever import SpecRetriever
    r = SpecRetriever.from_jsonl("data/processed/diamond_curated.jsonl")
    hits = r.retrieve("mutual exclusion among N processes", k=3)
    for h in hits:
        print(h.score, h.module_name, h.snippet[:80])
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
# TLA+ surface keywords we don't want dominating the IDF scores.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "have",
    "has", "but", "not", "all", "any", "you", "use", "uses", "using", "use",
    "MODULE", "EXTENDS", "VARIABLES", "VARIABLE", "CONSTANTS", "CONSTANT",
    "Init", "Next", "Spec", "TypeOK", "Invariant", "TLA",
}


def _tokenize(text: str) -> list[str]:
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if t not in _STOPWORDS and len(t) > 1
    ]


@dataclass
class RetrievalHit:
    prompt_id: str
    module_name: str
    score: float
    snippet: str          # First ~12 lines of the spec body, ready to inline.
    description: str      # The NL description from the user message.


@dataclass
class _Doc:
    prompt_id: str
    module_name: str
    description: str
    spec: str
    tokens: list[str]


class SpecRetriever:
    """Tiny BM25 over (NL description + spec body) tokens."""

    K1 = 1.5
    B = 0.75

    def __init__(self, docs: list[_Doc]) -> None:
        self.docs = docs
        if not docs:
            self._idf: dict[str, float] = {}
            self._dl: list[int] = []
            self._avgdl = 0.0
            self._tfs: list[Counter] = []
            return
        N = len(docs)
        df: dict[str, int] = defaultdict(int)
        self._tfs = []
        self._dl = []
        for d in docs:
            tf = Counter(d.tokens)
            self._tfs.append(tf)
            self._dl.append(len(d.tokens))
            for tok in tf:
                df[tok] += 1
        self._avgdl = sum(self._dl) / max(N, 1)
        self._idf = {
            tok: math.log((N - n + 0.5) / (n + 0.5) + 1.0)
            for tok, n in df.items()
        }

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "SpecRetriever":
        path = Path(path)
        docs: list[_Doc] = []
        if not path.exists():
            return cls(docs)
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            spec = ""
            description = ""
            for m in rec.get("messages", []):
                if m.get("role") == "user":
                    description = m.get("content", "")
                if m.get("role") == "assistant" and m.get("channel") == "final":
                    spec = m.get("content", "")
            if not spec:
                continue
            mod_m = re.search(r"^-{2,}\s*MODULE\s+(\w+)", spec, re.MULTILINE)
            module_name = mod_m.group(1) if mod_m else "Unknown"
            tokens = _tokenize(description) + _tokenize(spec)
            docs.append(_Doc(
                prompt_id=rec.get("_prompt_id", "?"),
                module_name=module_name,
                description=description,
                spec=spec,
                tokens=tokens,
            ))
        return cls(docs)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievalHit]:
        if not self.docs:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores: list[float] = []
        for i, tf in enumerate(self._tfs):
            s = 0.0
            dl = self._dl[i]
            for q in q_tokens:
                if q not in tf:
                    continue
                idf = self._idf.get(q, 0.0)
                f = tf[q]
                denom = f + self.K1 * (1 - self.B + self.B * dl / max(self._avgdl, 1.0))
                s += idf * (f * (self.K1 + 1)) / max(denom, 1e-9)
            scores.append(s)
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        out: list[RetrievalHit] = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                break
            d = self.docs[i]
            snippet_lines = d.spec.strip().splitlines()[:12]
            snippet = "\n".join(snippet_lines)
            out.append(RetrievalHit(
                prompt_id=d.prompt_id,
                module_name=d.module_name,
                score=scores[i],
                snippet=snippet,
                description=d.description.strip()[:280],
            ))
        return out

    def format_for_prompt(self, hits: Iterable[RetrievalHit]) -> str:
        """Render hits as an in-context block for the user message.

        Format keeps it short and operator-grounding-focused — the goal is to
        anchor the model on real verified vocabulary, not to copy-paste."""
        chunks: list[str] = []
        for h in hits:
            chunks.append(
                f"Reference: {h.module_name}\n"
                f"Goal: {h.description}\n"
                f"Spec excerpt:\n{h.snippet}"
            )
        if not chunks:
            return ""
        return (
            "Below are 1-3 verified TLA+ specs from the project corpus that "
            "use related vocabulary and idioms. Use them as grounding for "
            "operator names and structural patterns; do NOT copy them.\n\n"
            + "\n\n---\n\n".join(chunks)
        )
