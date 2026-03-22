"""
Download PDFs from URLs and extract text using pypdf (already in ChatTLA requirements).

Caches raw bytes + extracted text under data/derived/.pdf_cache/ by URL hash
so re-runs are fast and idempotent.
"""

from __future__ import annotations

import hashlib
import io
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

# Lazy import pypdf — clear error if missing
def _pypdf_reader(data: bytes):
    from pypdf import PdfReader

    return PdfReader(io.BytesIO(data))


def looks_like_pdf_url(url: str) -> bool:
    if not url or not url.startswith(("http://", "https://")):
        return False
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:24]


def _clean_pdf_text(raw: str, max_chars: int) -> str:
    if not raw:
        return ""
    # Normalize whitespace; drop excessive blank lines
    lines = [ln.strip() for ln in raw.splitlines()]
    out = []
    prev_empty = False
    for ln in lines:
        empty = not ln
        if empty and prev_empty:
            continue
        out.append(ln)
        prev_empty = empty
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars].strip()


def extract_text_from_pdf_bytes(data: bytes, max_pages: int = 3) -> str:
    reader = _pypdf_reader(data)
    parts: list[str] = []
    n = min(len(reader.pages), max_pages)
    for i in range(n):
        try:
            t = reader.pages[i].extract_text()
        except Exception:
            t = ""
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def fetch_pdf_bytes(url: str, timeout: float = 45.0, max_bytes: int = 25 * 1024 * 1024) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ChatTLA-tla-description-harvest/1.0 (+https://github.com/tlaplus)",
            "Accept": "application/pdf,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"PDF larger than {max_bytes} bytes")
    return data


def extract_pdf_excerpt_from_url(
    url: str,
    cache_dir: Path,
    max_pages: int = 3,
    max_chars: int = 4500,
    verbose: bool = False,
) -> tuple[str, str]:
    """
    Returns (excerpt_text, status) where status is 'ok', 'cache_hit', 'skip', or 'error: ...'.
    """
    if not looks_like_pdf_url(url):
        return "", "skip:not_pdf"

    cache_dir.mkdir(parents=True, exist_ok=True)
    h = _url_hash(url)
    txt_path = cache_dir / f"{h}.txt"
    pdf_path = cache_dir / f"{h}.pdf"

    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        return text, "cache_hit"

    try:
        raw = fetch_pdf_bytes(url)
        pdf_path.write_bytes(raw)
        raw_text = extract_text_from_pdf_bytes(raw, max_pages=max_pages)
        text = _clean_pdf_text(raw_text, max_chars)
        txt_path.write_text(text, encoding="utf-8")
        return text, "ok"
    except urllib.error.HTTPError as e:
        msg = f"error:http_{e.code}"
        if verbose:
            print(f"  PDF {url[:80]}... -> {msg}", flush=True)
        return "", msg
    except Exception as e:
        msg = f"error:{type(e).__name__}:{e!s}"[:200]
        if verbose:
            print(f"  PDF {url[:80]}... -> {msg}", flush=True)
        return "", msg


def collect_pdf_excerpts_for_sources(
    sources: list[str],
    cache_dir: Path,
    max_pdfs: int = 2,
    max_pages: int = 3,
    max_chars_per_pdf: int = 3500,
    verbose: bool = False,
) -> tuple[str, list[dict[str, str]]]:
    """
    Fetch up to `max_pdfs` PDF URLs from `sources`, concatenate excerpts with headings.
    Returns (combined_text, per_url_status_list).
    """
    pdf_urls = [u for u in sources if looks_like_pdf_url(u)][:max_pdfs]
    if not pdf_urls:
        return "", []

    chunks: list[str] = []
    details: list[dict[str, str | int]] = []
    for url in pdf_urls:
        excerpt, status = extract_pdf_excerpt_from_url(
            url, cache_dir, max_pages=max_pages, max_chars=max_chars_per_pdf, verbose=verbose
        )
        details.append({"url": url, "status": status, "chars": len(excerpt)})
        if excerpt:
            chunks.append(f"(from {url})\n{excerpt}")

    combined = "\n\n---\n\n".join(chunks) if chunks else ""
    return combined, details
