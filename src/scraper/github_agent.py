"""
github_agent.py — GitHub API scraper for TLA+ specifications (Tier-2 sources).

Systematically sweeps GitHub using Code Search to find .tla files beyond the
FormaLLM seed corpus.  Handles:
  - Multiple search queries covering common TLA+ patterns
  - Rate limiting (authenticated: 30 req/min; backed off gracefully)
  - Multiple GitHub token rotation via GITHUB_TOKEN_1..4 env vars
  - Fetching the raw .tla AND any sibling .cfg files
  - Per-repo license detection (MIT/Apache/BSD only by default)

Usage
-----
    from src.scraper.github_agent import GitHubAgent
    agent = GitHubAgent()
    for raw_spec in agent.iter_specs():
        print(raw_spec.tla_content[:120])

    # Or CLI:
    python -m src.scraper.github_agent --output data/raw/github.jsonl

Environment variables
---------------------
    GITHUB_TOKEN_1  (required — authenticated search has much higher rate limit)
    GITHUB_TOKEN_2  (optional — additional token for rotation)
    GITHUB_TOKEN_3  (optional)
    GITHUB_TOKEN_4  (optional)
"""

from __future__ import annotations

import os
import time
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import requests

from src.shared.schemas.dataset_schema import DatasetRecord


# ---------------------------------------------------------------------------
# Search queries — targeting all common TLA+ signals on GitHub
# ---------------------------------------------------------------------------
_SEARCH_QUERIES = [
    "language:tlaplus",
    'extension:tla "EXTENDS TLC"',
    'extension:tla "EXTENDS Sequences"',
    'extension:tla "Init == "',
    'extension:tla "SPECIFICATION Spec"',
    'extension:tla "\\A " "\\E "',
    'extension:tla "INVARIANT"',
]

# Allowed SPDX license identifiers for training data (copyleft excluded)
_ALLOWED_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "0bsd"}


@dataclass
class RawSpec:
    """Lightweight container for a spec fetched from GitHub before full validation."""
    tla_content: str
    cfg_content: Optional[str]
    source: str          # "github:owner/repo/path"
    license: str
    metadata: dict


class GitHubAgent:
    """
    Fetches TLA+ specs from GitHub using the Code Search API.

    Token rotation: tokens are cycled round-robin.  When a 403/429 is hit
    we switch tokens immediately and back off with exponential delay.
    """

    def __init__(
        self,
        allowed_licenses: set[str] = _ALLOWED_LICENSES,
        min_stars: int = 0,
    ):
        self.allowed_licenses = allowed_licenses
        self.min_stars = min_stars
        self._tokens = self._load_tokens()
        self._token_idx = 0
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def iter_specs(self, queries: list[str] = _SEARCH_QUERIES) -> Iterator[RawSpec]:
        """
        Yield RawSpec objects for all unique .tla files matching the given queries.

        Deduplication at this level is by raw GitHub blob URL — deeper content
        deduplication happens in dedup_agent.py.
        """
        seen_urls: set[str] = set()
        for query in queries:
            print(f"[github_agent] Query: {query!r}")
            try:
                for item in self._search_code(query):
                    url = item.get("html_url", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    spec = self._fetch_spec(item)
                    if spec is not None:
                        yield spec
            except Exception as exc:
                print(f"[github_agent] Query failed: {exc}")

    def to_dataset_records(self, specs: Iterator[RawSpec]) -> Iterator[DatasetRecord]:
        """Convert RawSpec objects to DatasetRecord objects."""
        for spec in specs:
            record = DatasetRecord(
                id=DatasetRecord.make_id(spec.tla_content),
                source=spec.source,
                license=spec.license,
                tla_content=spec.tla_content,
                cfg_content=spec.cfg_content,
                metadata=spec.metadata,
            )
            yield record

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _token(self) -> str:
        return self._tokens[self._token_idx % len(self._tokens)]

    def _rotate_token(self) -> None:
        self._token_idx = (self._token_idx + 1) % len(self._tokens)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token()}"}

    def _search_code(self, query: str, per_page: int = 100) -> Iterator[dict]:
        """Paginate through GitHub code search results."""
        page = 1
        while True:
            resp = self._get(
                "https://api.github.com/search/code",
                params={"q": query, "per_page": per_page, "page": page},
            )
            if resp is None:
                break
            data = resp.json()
            items = data.get("items", [])
            if not items:
                break
            yield from items
            # GitHub caps code search at 1000 results (10 pages of 100)
            if len(items) < per_page or page >= 10:
                break
            page += 1
            time.sleep(2)  # be polite between pages

    def _fetch_spec(self, item: dict) -> Optional[RawSpec]:
        """
        Fetch raw .tla content and optional sibling .cfg for a search result item.
        Returns None if license is not allowed or fetch fails.
        """
        repo = item.get("repository", {})
        owner = repo.get("owner", {}).get("login", "unknown")
        repo_name = repo.get("name", "unknown")
        full_name = f"{owner}/{repo_name}"
        file_path: str = item.get("path", "")

        # Check repo license
        license_key = self._get_repo_license(full_name)
        if license_key not in self.allowed_licenses:
            return None

        # Fetch raw .tla
        raw_url = f"https://raw.githubusercontent.com/{full_name}/HEAD/{file_path}"
        resp = self._get(raw_url)
        if resp is None or not resp.ok:
            return None
        tla_content = resp.text

        # Try sibling .cfg
        cfg_content: Optional[str] = None
        cfg_path = file_path.rsplit(".", 1)[0] + ".cfg"
        cfg_url = f"https://raw.githubusercontent.com/{full_name}/HEAD/{cfg_path}"
        cfg_resp = self._get(cfg_url)
        if cfg_resp is not None and cfg_resp.ok:
            cfg_content = cfg_resp.text

        stars = repo.get("stargazers_count", 0)
        if stars < self.min_stars:
            return None

        return RawSpec(
            tla_content=tla_content,
            cfg_content=cfg_content,
            source=f"github:{full_name}/{file_path}",
            license=license_key,
            metadata={
                "repo": full_name,
                "path": file_path,
                "stars": stars,
                "repo_url": f"https://github.com/{full_name}",
            },
        )

    def _get_repo_license(self, full_name: str) -> str:
        """Return SPDX license key for a repo, cached in memory."""
        if not hasattr(self, "_license_cache"):
            self._license_cache: dict[str, str] = {}
        if full_name in self._license_cache:
            return self._license_cache[full_name]
        resp = self._get(f"https://api.github.com/repos/{full_name}")
        key = "unknown"
        if resp and resp.ok:
            lic = resp.json().get("license") or {}
            key = (lic.get("spdx_id") or "unknown").lower()
        self._license_cache[full_name] = key
        return key

    def _get(self, url: str, params: Optional[dict] = None, retries: int = 3) -> Optional[requests.Response]:
        """GET with automatic token rotation and exponential backoff on rate limit."""
        delay = 2
        for attempt in range(retries):
            try:
                resp = self._session.get(url, headers=self._headers(), params=params, timeout=20)
                if resp.status_code in (403, 429):
                    retry_after = int(resp.headers.get("Retry-After", delay))
                    print(f"[github_agent] Rate limited (token {self._token_idx}), waiting {retry_after}s")
                    self._rotate_token()
                    time.sleep(retry_after)
                    delay *= 2
                    continue
                return resp
            except requests.RequestException as exc:
                print(f"[github_agent] Request error: {exc}, retry {attempt+1}/{retries}")
                time.sleep(delay)
                delay *= 2
        return None

    @staticmethod
    def _load_tokens() -> list[str]:
        tokens = []
        for i in range(1, 5):
            t = os.getenv(f"GITHUB_TOKEN_{i}") or os.getenv("GITHUB_TOKEN")
            if t:
                tokens.append(t)
        if not tokens:
            raise EnvironmentError(
                "No GitHub token found. Set GITHUB_TOKEN or GITHUB_TOKEN_1..4 in your .env"
            )
        return list(dict.fromkeys(tokens))  # deduplicate, preserve order


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Scrape TLA+ specs from GitHub")
    parser.add_argument("--output", default="data/raw/github.jsonl", help="Output JSONL path")
    parser.add_argument("--min-stars", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agent = GitHubAgent(min_stars=args.min_stars)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for record in agent.to_dataset_records(agent.iter_specs()):
            f.write(record.to_json(indent=None) + "\n")
            count += 1
            if count % 10 == 0:
                print(f"[github_agent] {count} records written...")

    print(f"[github_agent] Done. {count} records → {out_path}")
