"""
scraper/__init__.py
===================
SugamGov Daily Live Scheme Scraper Package.

Pipeline:
  1. myscheme_scraper.py  — Fetch latest schemes from myscheme.gov.in JSON API
  2. pmindia_scraper.py   — Fetch PM-level schemes from pmindia.gov.in
  3. diff.py              — Checksum-based change detection (skip unchanged)
  4. embedder.py          — Chunk + embed new/changed schemes
  5. db_writer.py         — UPSERT into PostgreSQL (same schema as existing RAG)
  6. run_daily.py         — Daily cron entry point + report logger
"""
