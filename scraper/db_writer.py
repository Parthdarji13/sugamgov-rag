"""
scraper/db_writer.py
=====================
Database UPSERT Writer for SugamGov Live Scheme Scraper.

Handles:
  - INSERT new schemes into 'schemes' table
  - INSERT new chunks into 'scheme_chunks' table
  - UPDATE changed schemes (update fields + checksum + last_scraped_at)
  - DELETE old chunks for updated schemes and re-insert fresh ones
  - Update scrape_runs log table with run statistics

All operations use transactions — partial failures roll back cleanly.
Uses the same PostgreSQL instance as your RAG backend.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from scraper.diff import DiffResult, Action

logger = logging.getLogger("sugamgov_scraper.db_writer")


def _scheme_to_row(scheme: Dict[str, Any], checksum: str) -> Dict[str, Any]:
    """Maps a scraped scheme dict to the exact columns in the 'schemes' table."""
    now = datetime.now(timezone.utc)
    return {
        "scheme_id":       scheme["scheme_id"],
        "scheme_name":     scheme.get("scheme_name", ""),
        "slug":            scheme.get("slug") or None,
        "details":         scheme.get("details") or None,
        "benefits":        scheme.get("benefits") or None,
        "eligibility":     scheme.get("eligibility") or None,
        "application":     scheme.get("application") or None,
        "documents":       scheme.get("documents") or None,
        "level":           scheme.get("level") or None,
        "state":           scheme.get("state") or None,
        "states":          json.dumps(scheme.get("states") or [], ensure_ascii=False),
        "categories":      json.dumps(scheme.get("categories") or [], ensure_ascii=False),
        "tags":            scheme.get("tags") or None,
        "normalized_tags": json.dumps([], ensure_ascii=False),
        "source_url":      scheme.get("source_url") or None,
        "data_source":     scheme.get("data_source", "live_scraper"),
        "last_scraped_at": now,
        "scrape_checksum": checksum,
        "created_at":      now,
        "updated_at":      now,
    }


def _insert_scheme(conn, row: Dict[str, Any]) -> None:
    """Inserts a new scheme row into the 'schemes' table."""
    sql = text("""
        INSERT INTO schemes (
            scheme_id, scheme_name, slug, details, benefits, eligibility,
            application, documents, level, state, states, categories,
            tags, normalized_tags, source_url, data_source,
            last_scraped_at, scrape_checksum, created_at, updated_at
        ) VALUES (
            :scheme_id, :scheme_name, :slug, :details, :benefits, :eligibility,
            :application, :documents, :level, :state,
            CAST(:states AS jsonb), CAST(:categories AS jsonb),
            :tags, CAST(:normalized_tags AS jsonb), :source_url, :data_source,
            :last_scraped_at, :scrape_checksum, :created_at, :updated_at
        )
        ON CONFLICT (scheme_id) DO NOTHING
    """)
    conn.execute(sql, row)


def _update_scheme(conn, row: Dict[str, Any]) -> None:
    """Updates an existing scheme row (content + metadata fields)."""
    sql = text("""
        UPDATE schemes SET
            scheme_name     = :scheme_name,
            details         = :details,
            benefits        = :benefits,
            eligibility     = :eligibility,
            application     = :application,
            documents       = :documents,
            level           = :level,
            state           = :state,
            states          = CAST(:states AS jsonb),
            categories      = CAST(:categories AS jsonb),
            tags            = :tags,
            source_url      = :source_url,
            data_source     = :data_source,
            last_scraped_at = :last_scraped_at,
            scrape_checksum = :scrape_checksum,
            updated_at      = :updated_at
        WHERE scheme_id = :scheme_id
    """)
    conn.execute(sql, row)


def _delete_chunks_for_scheme(conn, scheme_id: str) -> int:
    """Deletes all existing chunks for a scheme (before re-inserting fresh ones)."""
    result = conn.execute(
        text("DELETE FROM scheme_chunks WHERE scheme_id = :sid"),
        {"sid": scheme_id},
    )
    return result.rowcount


def _insert_chunks(conn, chunks: List[Dict[str, Any]]) -> int:
    """Inserts chunk rows into 'scheme_chunks'. Skips chunks with duplicate chunk_id."""
    if not chunks:
        return 0

    inserted = 0
    sql = text("""
        INSERT INTO scheme_chunks (
            scheme_id, chunk_id, field_name, chunk_text, chunk_index,
            embedding_local, metadata, created_at
        ) VALUES (
            :scheme_id, :chunk_id, :field_name, :chunk_text, :chunk_index,
            :embedding_local, CAST(:metadata AS jsonb), NOW()
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            chunk_text      = EXCLUDED.chunk_text,
            embedding_local = EXCLUDED.embedding_local,
            metadata        = EXCLUDED.metadata
    """)

    for chunk in chunks:
        emb = chunk.get("embedding_local")
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except Exception:
                emb = None
        elif isinstance(emb, (list, tuple)):
            emb = [float(x) for x in emb]
        else:
            emb = None

        meta = chunk.get("metadata", "{}")
        if isinstance(meta, dict):
            meta = json.dumps(meta)

        conn.execute(sql, {
            "scheme_id":       chunk["scheme_id"],
            "chunk_id":        chunk["chunk_id"],
            "field_name":      chunk["field_name"],
            "chunk_text":      chunk["chunk_text"],
            "chunk_index":     chunk["chunk_index"],
            "embedding_local": emb,
            "metadata":        meta,
        })
        inserted += 1

    return inserted


def write_schemes(
    engine: Engine,
    diff_results: List[DiffResult],
    checksums: Dict[str, str],
    scheme_map: Dict[str, Dict[str, Any]],                    # scheme_id → scheme dict
    chunk_map: Dict[str, List[Dict[str, Any]]],               # scheme_id → chunks list
) -> Dict[str, int]:
    """
    Writes new and updated schemes + their chunks to PostgreSQL.

    Args:
        engine:       SQLAlchemy engine for the PostgreSQL database.
        diff_results: List of DiffResult from diff.diff_schemes().
        checksums:    {scheme_id: checksum} from diff.diff_schemes().
        scheme_map:   {scheme_id: scraped_scheme_dict}
        chunk_map:    {scheme_id: [chunk_dict, ...]}

    Returns:
        Dict with counts: schemes_inserted, schemes_updated, chunks_inserted.
    """
    counts = {
        "schemes_inserted": 0,
        "schemes_updated":  0,
        "chunks_inserted":  0,
        "errors":           0,
    }

    # Process only INSERT and UPDATE — skip SKIP
    to_process = [r for r in diff_results if r.action in ("INSERT", "UPDATE")]
    if not to_process:
        logger.info("No new or changed schemes to write.")
        return counts

    logger.info("Writing %d schemes to DB (%d to process)...",
                len(to_process), len(to_process))

    for diff in to_process:
        sid = diff.scheme_id
        scheme = scheme_map.get(sid)
        chunks = chunk_map.get(sid, [])

        if not scheme:
            logger.warning("No scheme dict found for %s — skipping", sid)
            continue

        checksum = checksums.get(sid, "")
        row = _scheme_to_row(scheme, checksum)

        try:
            with engine.begin() as conn:  # begin() auto-commits or rolls back
                if diff.action == "INSERT":
                    _insert_scheme(conn, row)
                elif diff.action == "UPDATE":
                    _delete_chunks_for_scheme(conn, sid)
                    _update_scheme(conn, row)

                # Insert chunks for both INSERT and UPDATE
                n = _insert_chunks(conn, chunks)
                counts["chunks_inserted"] += n

                if diff.action == "INSERT":
                    counts["schemes_inserted"] += 1
                    logger.info("INSERT: %s — %s", sid, scheme.get("scheme_name", "")[:60])
                elif diff.action == "UPDATE":
                    counts["schemes_updated"] += 1
                    logger.info(
                        "UPDATE: %s — changed fields: %s",
                        scheme.get("scheme_name", "")[:60],
                        diff.changed_fields,
                    )

        except Exception as e:
            counts["errors"] += 1
            logger.error("Failed to write scheme %s: %s", sid, e, exc_info=True)

    logger.info(
        "DB write complete: +%d new schemes | ~%d updated | +%d chunks | %d errors",
        counts["schemes_inserted"],
        counts["schemes_updated"],
        counts["chunks_inserted"],
        counts["errors"],
    )
    return counts


def update_scrape_run(
    engine: Engine,
    run_date: str,
    source: str,
    started_at: datetime,
    stats: Dict[str, int],
    status: str = "success",
    error_message: Optional[str] = None,
) -> None:
    """
    Upserts a row in the scrape_runs log table for monitoring.
    Call this at the end of each scraper's run.
    """
    sql = text("""
        INSERT INTO scrape_runs (
            run_date, started_at, completed_at, source,
            schemes_found, schemes_added, schemes_updated, schemes_skipped,
            errors, status, error_message
        ) VALUES (
            CAST(:run_date AS DATE), :started_at, NOW(), :source,
            :schemes_found, :schemes_added, :schemes_updated, :schemes_skipped,
            :errors, :status, :error_message
        )
        ON CONFLICT (run_date, source) DO UPDATE SET
            completed_at    = NOW(),
            schemes_found   = EXCLUDED.schemes_found,
            schemes_added   = EXCLUDED.schemes_added,
            schemes_updated = EXCLUDED.schemes_updated,
            schemes_skipped = EXCLUDED.schemes_skipped,
            errors          = EXCLUDED.errors,
            status          = EXCLUDED.status,
            error_message   = EXCLUDED.error_message
    """)
    try:
        with engine.begin() as conn:
            conn.execute(sql, {
                "run_date":        run_date,
                "started_at":      started_at,
                "source":          source,
                "schemes_found":   stats.get("schemes_found", 0),
                "schemes_added":   stats.get("schemes_inserted", 0),
                "schemes_updated": stats.get("schemes_updated", 0),
                "schemes_skipped": stats.get("schemes_skipped", 0),
                "errors":          stats.get("errors", 0),
                "status":          status,
                "error_message":   error_message,
            })
    except Exception as e:
        logger.error("Failed to update scrape_runs log: %s", e)
