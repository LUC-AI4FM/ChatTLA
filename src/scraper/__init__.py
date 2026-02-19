"""
scraper — Phase 1 data collection package.

Entrypoints:
  src.scraper.pipeline          — full orchestrated pipeline
  src.scraper.ingest_formalllm  — seed ingestion from FormaLLM submodule
  src.scraper.github_agent      — GitHub code search scraper
  src.scraper.dedup_agent       — MinHash LSH near-deduplication
  src.scraper.annotate          — local Ollama annotation pass
"""
