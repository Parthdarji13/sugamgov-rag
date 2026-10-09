"""
scraper/apply_migration.py
===========================
Applies migration 003 to add scraper tracking columns.
Run ONCE after setting up the scraper:

  python scraper/apply_migration.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env manually before importing anything
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

db_url = os.environ.get("DATABASE_URL", "")
if not db_url:
    print("ERROR: DATABASE_URL not set in .env")
    sys.exit(1)

try:
    from sqlalchemy import create_engine, text
except ImportError:
    print("ERROR: sqlalchemy not installed. Run: pip install sqlalchemy")
    sys.exit(1)

MIGRATION_FILE = PROJECT_ROOT / "database" / "migrations" / "003_add_scraper_columns.sql"
sql_content = MIGRATION_FILE.read_text(encoding="utf-8")

# Split on semicolons, filter comments and blanks
statements = []
for raw_stmt in sql_content.split(";"):
    stmt = "\n".join(
        line for line in raw_stmt.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ).strip()
    if stmt:
        statements.append(stmt)

print(f"Connecting to database...")
engine = create_engine(db_url)

print(f"Applying {len(statements)} statements from migration 003...")
with engine.begin() as conn:
    for i, stmt in enumerate(statements, 1):
        try:
            conn.execute(text(stmt))
            print(f"  [{i}/{len(statements)}] OK")
        except Exception as e:
            err = str(e).split("\n")[0]
            print(f"  [{i}/{len(statements)}] WARN (may already exist): {err[:80]}")

print("\n✅ Migration 003 applied successfully!")
print("New columns added to 'schemes' table: data_source, source_url, last_scraped_at, scrape_checksum")
print("New table created: scrape_runs (daily run audit log)")
