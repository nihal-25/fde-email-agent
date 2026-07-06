"""Docs/support/GitHub ingestion for the platform-Q&A RAG store (Phase 2 case 2).

crawl  -> chunk -> embed -> store(pgvector). Polite by construction (rate-limited,
identifies itself, honors robots where present). See pipeline.py for the CLI.
"""
