"""
scraper/run_daily.py
=====================
Daily Orchestrator for SugamGov Live Scheme Scraper.

This is the single entry point run by the Windows Task Scheduler every morning.

Pipeline order:
  1. Load DB engine from .env DATABASE_URL
  2. Run myscheme.gov.in API scraper
  3. Run pmindia.gov.in HTML scraper
  4. Merge all scraped schemes (deduplicate by scheme_id)
  5. Diff against DB (find new/changed schemes)
  6. Chunk + embed only new/changed schemes
  7. UPSERT to PostgreSQL
  8. Write scrape_runs log entries
  9. Print daily report to console + log file

Usage:
  python -m scraper.run_daily              # Full run
  python -m scraper.run_daily --dry-run    # Scrape + diff only, no DB writes
  python -m scraper.run_daily --source myscheme  # Only run one scraper
  python -m scraper.run_daily --max-pages 5      # Limit pages (for testing)

Schedule (Windows Task Scheduler):
  Action: python -m scraper.run_daily >> scraper/logs/cron.log 2>&1
  Trigger: Daily at 06:00 AM
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 encoding on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Ensure project root is in sys.path so 'src.*' imports work ─────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ── Setup logging (console + rotating log file) ─────────────────────────────
LOG_DIR = PROJECT_ROOT / "scraper" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

today_str = datetime.now().strftime("%Y-%m-%d")
log_file = LOG_DIR / f"scrape_{today_str}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("sugamgov_scraper.daily")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SugamGov Daily Scheme Scraper")
    p.add_argument("--dry-run",    action="store_true", help="Scrape + diff only, no DB writes or embedding")
    p.add_argument("--source",     choices=["myscheme", "pmindia", "all"], default="all",
                   help="Which scraper(s) to run (default: all)")
    p.add_argument("--max-pages",  type=int, default=500, help="Max schemes to fetch (default: 500)")
    p.add_argument("--offset",     type=int, default=0, help="Starting catalog offset (default: 0)")
    p.add_argument("--recheck-all", action="store_true", help="Do not skip schemes already in database (check all for updates)")
    p.add_argument("--no-embed",   action="store_true", help="Skip embedding step (faster, stores NULL embeddings)")
    return p.parse_args()


def _banner(msg: str) -> None:
    border = "=" * 70
    logger.info("\n%s\n  %s\n%s", border, msg, border)


def run(args: argparse.Namespace) -> int:
    """
    Main scraper orchestration function.
    Returns exit code: 0 = success, 1 = partial errors, 2 = fatal error.
    """
    started_at = datetime.now(timezone.utc)
    _banner(f"SugamGov Daily Scheme Scraper — {today_str}")

    # ── Load env ────────────────────────────────────────────────────────────
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set in .env — cannot connect to PostgreSQL")
        return 2

    engine = create_engine(db_url)
    logger.info("Connected to PostgreSQL database")

    # ── Pre-fetch existing scheme IDs from DB to fast-skip known schemes ────
    existing_ids = set()
    if not args.recheck_all:
        try:
            with engine.connect() as conn:
                existing_ids = set(conn.execute(text("SELECT scheme_id FROM schemes")).scalars())
            logger.info("Loaded %d existing scheme IDs from DB to auto-skip already collected schemes", len(existing_ids))
        except Exception as e:
            logger.warning("Could not pre-fetch existing scheme IDs: %s", e)

    # ── Step 1: Scrape ───────────────────────────────────────────────────────
    all_scraped = []

    if args.source in ("myscheme", "all"):
        try:
            from scraper.scrapers.myscheme_scraper import MySchemeScraper
            scraper = MySchemeScraper(
                max_schemes=args.max_pages,
                offset=args.offset,
                skip_existing=(not args.recheck_all),
                existing_ids=existing_ids,
            )
            schemes = scraper.fetch_and_validate()
            logger.info("myscheme.gov.in: %d valid schemes fetched", len(schemes))
            all_scraped.extend(schemes)
        except Exception as e:
            logger.error("myscheme scraper failed: %s", e, exc_info=True)

    if args.source in ("pmindia", "all"):
        try:
            from scraper.scrapers.pmindia_scraper import PmIndiaScraper
            scraper = PmIndiaScraper()
            schemes = scraper.fetch_and_validate()
            logger.info("pmindia.gov.in: %d valid schemes fetched", len(schemes))
            all_scraped.extend(schemes)
        except Exception as e:
            logger.error("pmindia scraper failed: %s", e, exc_info=True)

    if not all_scraped:
        logger.warning("No schemes scraped from any source — nothing to do")
        return 1

    # Deduplicate: if same scheme_id from multiple scrapers, keep first
    seen_ids = set()
    deduped = []
    for s in all_scraped:
        sid = s.get("scheme_id", "")
        if sid and sid not in seen_ids:
            seen_ids.add(sid)
            deduped.append(s)
    logger.info("Total unique scraped schemes: %d (from %d raw)", len(deduped), len(all_scraped))

    # ── Step 2: Diff ─────────────────────────────────────────────────────────
    from scraper.diff import diff_schemes
    diff_results, checksums = diff_schemes(deduped, engine)

    new_count     = sum(1 for r in diff_results if r.action == "INSERT")
    updated_count = sum(1 for r in diff_results if r.action == "UPDATE")
    skipped_count = sum(1 for r in diff_results if r.action == "SKIP")

    _banner(f"Diff Result: {new_count} NEW  |  {updated_count} UPDATED  |  {skipped_count} UNCHANGED")

    if args.dry_run:
        logger.info("DRY RUN mode — no DB writes or embedding. Exiting.")
        _print_report(started_at, new_count, updated_count, skipped_count, 0, 0, dry_run=True)
        return 0

    if new_count == 0 and updated_count == 0:
        logger.info("Nothing changed today — database is already up to date.")
        _print_report(started_at, 0, 0, skipped_count, 0, 0)
        return 0

    # ── Step 3: Chunk + Embed ─────────────────────────────────────────────────
    # Only process schemes that are new or changed
    to_process = {
        r.scheme_id for r in diff_results if r.action in ("INSERT", "UPDATE")
    }
    schemes_to_embed = [s for s in deduped if s["scheme_id"] in to_process]

    logger.info("Chunking + embedding %d new/changed schemes...", len(schemes_to_embed))

    from scraper.embedder import process_schemes
    processed = process_schemes(schemes_to_embed, embed=not args.no_embed)

    # Build maps for db_writer
    scheme_map = {scheme["scheme_id"]: scheme for scheme, _ in processed}
    chunk_map  = {scheme["scheme_id"]: chunks  for scheme, chunks in processed}

    total_chunks = sum(len(c) for c in chunk_map.values())
    logger.info("Generated %d chunks across %d schemes", total_chunks, len(processed))

    # ── Step 4: Write to DB ──────────────────────────────────────────────────
    from scraper.db_writer import write_schemes, update_scrape_run
    write_stats = write_schemes(engine, diff_results, checksums, scheme_map, chunk_map)

    write_stats["schemes_found"]  = len(deduped)
    write_stats["schemes_skipped"] = skipped_count

    # Log run stats to scrape_runs table
    for source_name in (["myscheme_api"] if args.source == "myscheme"
                        else ["pmindia_gov"] if args.source == "pmindia"
                        else ["myscheme_api", "pmindia_gov"]):
        update_scrape_run(
            engine=engine,
            run_date=today_str,
            source=source_name,
            started_at=started_at,
            stats=write_stats,
            status="success",
        )

    # ── Step 5: Daily Report ─────────────────────────────────────────────────
    _print_report(
        started_at,
        write_stats["schemes_inserted"],
        write_stats["schemes_updated"],
        skipped_count,
        total_chunks,
        write_stats["errors"],
    )

    return 0 if write_stats["errors"] == 0 else 1


def _print_report(
    started_at: datetime,
    added: int,
    updated: int,
    skipped: int,
    chunks: int,
    errors: int,
    dry_run: bool = False,
) -> None:
    """Prints the end-of-run summary report."""
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    mins, secs = divmod(int(elapsed), 60)

    mode = "[DRY RUN] " if dry_run else ""
    border = "*" * 70
    logger.info(
        "\n%s\n"
        "  %sSugamGov Daily Scrape Report — %s\n"
        "%s\n"
        "  [+] New schemes added   : %d\n"
        "  [*] Schemes updated     : %d\n"
        "  [-] Unchanged (skipped) : %d\n"
        "  [#] Chunks written      : %d\n"
        "  [!] Errors              : %d\n"
        "  [T] Duration            : %dm %ds\n"
        "%s",
        border, mode, today_str, border,
        added, updated, skipped, chunks, errors,
        mins, secs, border,
    )


if __name__ == "__main__":
    args = parse_args()
    exit_code = run(args)
    sys.exit(exit_code)
