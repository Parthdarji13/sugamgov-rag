"""
scraper/diff.py
================
Change Detection Engine for SugamGov Live Scheme Scraper.

How it works:
  1. For each scraped scheme, compute a SHA-256 checksum of its content fields.
  2. Query the DB for the existing checksum (stored in schemes.scrape_checksum).
  3. If no row exists → NEW scheme (action: INSERT).
  4. If checksums differ → CHANGED scheme (action: UPDATE).
  5. If checksums match → UNCHANGED (action: SKIP — no DB write, no re-embedding).

This saves:
  - DB writes: only write what changed
  - Embedding compute: only re-embed new/changed schemes
  - Time: skip 95%+ of schemes on a normal day
"""

import hashlib
import json
import logging
from typing import Dict, Any, List, Literal, Tuple
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger("sugamgov_scraper.diff")

# Fields used to compute the content checksum.
# Only changes in these fields trigger a re-embed.
CHECKSUM_FIELDS = [
    "scheme_name",
    "details",
    "benefits",
    "eligibility",
    "application",
    "documents",
    "level",
    "state",
]

Action = Literal["INSERT", "UPDATE", "SKIP"]


@dataclass
class DiffResult:
    """Result of comparing one scraped scheme to the DB."""
    scheme_id: str
    scheme_name: str
    action: Action
    scraped_checksum: str
    db_checksum: str = ""
    changed_fields: List[str] = field(default_factory=list)


def compute_checksum(scheme: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 checksum of the scheme's content fields.
    Uses sorted JSON serialization for consistency regardless of dict order.
    """
    content = {
        field: str(scheme.get(field) or "").strip()
        for field in CHECKSUM_FIELDS
    }
    content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content_str.encode("utf-8")).hexdigest()


def _fetch_existing_checksums(engine: Engine, scheme_ids: List[str]) -> Dict[str, str]:
    """
    Batch-fetches existing checksums from the DB for a list of scheme_ids.
    Returns {scheme_id: scrape_checksum} for existing rows.
    """
    if not scheme_ids:
        return {}

    # Use ANY() for efficient bulk lookup
    sql = text("""
        SELECT scheme_id, scrape_checksum
        FROM schemes
        WHERE scheme_id = ANY(:ids)
    """)

    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"ids": scheme_ids}).fetchall()
        return {row.scheme_id: (row.scrape_checksum or "") for row in rows}
    except Exception as e:
        logger.error("Failed to fetch checksums from DB: %s", e)
        return {}


def _detect_changed_fields(old_scheme: Dict[str, Any], new_scheme: Dict[str, Any]) -> List[str]:
    """Returns a list of field names that differ between old and new data."""
    changed = []
    for f in CHECKSUM_FIELDS:
        old_val = str(old_scheme.get(f) or "").strip()
        new_val = str(new_scheme.get(f) or "").strip()
        if old_val != new_val:
            changed.append(f)
    return changed


def _fetch_existing_scheme(engine: Engine, scheme_id: str) -> Dict[str, Any]:
    """Fetches the existing scheme row for field-level diff reporting."""
    sql = text("""
        SELECT scheme_name, details, benefits, eligibility, application,
               documents, level, state, scrape_checksum
        FROM schemes
        WHERE scheme_id = :sid
        LIMIT 1
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(sql, {"sid": scheme_id}).fetchone()
        if row:
            return dict(row._mapping)
        return {}
    except Exception:
        return {}


def diff_schemes(
    scraped: List[Dict[str, Any]],
    engine: Engine,
) -> Tuple[List[DiffResult], Dict[str, str]]:
    """
    Compares a list of scraped schemes against what's in the DB.

    Args:
        scraped: List of scheme dicts from the scraper.
        engine: SQLAlchemy engine connected to the PostgreSQL DB.

    Returns:
        Tuple of:
          - List[DiffResult]: one result per scheme with action INSERT/UPDATE/SKIP
          - Dict[str, str]: {scheme_id: checksum} for all scraped schemes
    """
    if not scraped:
        return [], {}

    # 1. Compute checksums for all scraped schemes
    checksums: Dict[str, str] = {}
    for scheme in scraped:
        sid = scheme.get("scheme_id", "")
        if sid:
            checksums[sid] = compute_checksum(scheme)

    scheme_ids = list(checksums.keys())

    # 2. Batch-fetch existing checksums from DB
    db_checksums = _fetch_existing_checksums(engine, scheme_ids)

    # 3. Compare and classify each scheme
    results: List[DiffResult] = []

    for scheme in scraped:
        sid = scheme.get("scheme_id", "")
        if not sid:
            continue

        new_checksum = checksums.get(sid, "")
        db_checksum = db_checksums.get(sid, "")

        if not db_checksum:
            # Not in DB at all → INSERT
            action: Action = "INSERT"
            changed_fields: List[str] = []
            logger.debug("NEW scheme: %s — %s", sid, scheme.get("scheme_name", "")[:60])

        elif db_checksum == new_checksum:
            # Checksum matches → completely unchanged → SKIP
            action = "SKIP"
            changed_fields = []

        else:
            # Checksum differs → something changed → UPDATE
            action = "UPDATE"
            existing = _fetch_existing_scheme(engine, sid)
            changed_fields = _detect_changed_fields(existing, scheme)
            logger.info(
                "CHANGED scheme: %s — fields: %s",
                scheme.get("scheme_name", "")[:60],
                changed_fields,
            )

        results.append(DiffResult(
            scheme_id=sid,
            scheme_name=scheme.get("scheme_name", ""),
            action=action,
            scraped_checksum=new_checksum,
            db_checksum=db_checksum,
            changed_fields=changed_fields,
        ))

    new_count = sum(1 for r in results if r.action == "INSERT")
    update_count = sum(1 for r in results if r.action == "UPDATE")
    skip_count = sum(1 for r in results if r.action == "SKIP")

    logger.info(
        "Diff complete: %d new | %d updated | %d unchanged (skipped)",
        new_count, update_count, skip_count,
    )

    return results, checksums
