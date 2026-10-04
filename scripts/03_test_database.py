"""
scripts/03_test_database.py
===========================
Automated verification test for the SugamGov PostgreSQL database.

This script:
  1. Loads DATABASE_URL from .env using python-dotenv.
  2. Establishes a connection using SQLAlchemy 2.0 and psycopg (psycopg3).
  3. Executes a test query (SELECT 1).
  4. Verifies the pgvector ('vector') extension is installed and active.
  5. Applies the initial schema migration (database/migrations/001_initial_schema.sql)
     if tables are not yet present, or when run with '--migrate'.
  6. Inspects the schema for the four expected tables:
     - schemes
     - sources
     - scheme_versions
     - scheme_chunks
  7. Prints table columns and verifies structure.
  8. Counts rows in each table and asserts that all tables contain exactly 0 rows.
  9. Emits the exact requested summary report.
"""

import os
import sys
import warnings
import argparse
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SAWarning

# Filter dialect reflection warning for custom pgvector type
warnings.filterwarnings("ignore", category=SAWarning, message=".*Did not recognize type 'vector'.*")

# Reconfigure stdout for UTF-8 on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Path & Environment Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
MIGRATION_PATH = PROJECT_ROOT / "database" / "migrations" / "001_initial_schema.sql"

EXPECTED_TABLES = [
    "schemes",
    "sources",
    "scheme_versions",
    "scheme_chunks"
]


def apply_migration_file(engine):
    """Executes the DDL statements in 001_initial_schema.sql."""
    if not MIGRATION_PATH.exists():
        raise FileNotFoundError(f"Migration file not found at: {MIGRATION_PATH}")

    print(f"\n[*] Applying schema migration from: {MIGRATION_PATH.relative_to(PROJECT_ROOT)}...")
    with open(MIGRATION_PATH, "r", encoding="utf-8") as f:
        sql_content = f.read()

    # Use raw connection to allow multi-statement DDL execution cleanly
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        cursor.execute(sql_content)
        raw_conn.commit()
        cursor.close()
        print("    ✓ Migration 001_initial_schema.sql applied successfully.")
    finally:
        raw_conn.close()


def test_database():
    parser = argparse.ArgumentParser(description="Test SugamGov database setup")
    parser.add_argument("--migrate", action="store_true", help="Apply initial migration before testing")
    args = parser.parse_args()

    print("=" * 72)
    print("  sugamgov-rag | Step 3.6: Database & Schema Verification")
    print("=" * 72)

    # 1. Load environment variables
    if not ENV_PATH.exists():
        print(f"\n[!] .env file not found at: {ENV_PATH.relative_to(PROJECT_ROOT)}")
        print("    Please create a .env file with your local DATABASE_URL, for example:")
        print("    DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/sugamgov")
        print("\nResult: FAILURE")
        sys.exit(1)

    load_dotenv(dotenv_path=ENV_PATH)
    database_url = os.getenv("DATABASE_URL")

    if not database_url or "YOUR_PASSWORD" in database_url:
        print("\n[!] Please configure your actual password in .env for DATABASE_URL.")
        print("    Format: DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/sugamgov")
        print("\nResult: FAILURE")
        sys.exit(1)

    # Mask password for safe logging
    masked_url = re_mask_url(database_url)
    print(f"\n[1/4] Connecting to database: {masked_url}")

    # 2. Connect via SQLAlchemy 2.0 + psycopg
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # 3. Test basic query
            result = conn.execute(text("SELECT 1")).scalar()
            if result != 1:
                raise ValueError("Query 'SELECT 1' did not return 1")
            print("      ✓ PostgreSQL connection established successfully.")

        # Check existing tables
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        # If any expected table is missing or --migrate requested, apply migration
        missing_tables = [tbl for tbl in EXPECTED_TABLES if tbl not in existing_tables]
        if args.migrate or missing_tables:
            apply_migration_file(engine)
            # Re-inspect tables after migration
            inspector = inspect(engine)
            existing_tables = inspector.get_table_names()

        # 4. Check pgvector extension
        print("\n[2/4] Verifying 'vector' (pgvector) extension...")
        with engine.connect() as conn:
            ext_query = text(
                "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
            )
            ext_row = conn.execute(ext_query).fetchone()

            if not ext_row:
                print("      ✗ 'vector' extension is NOT active in database.")
                print("\nResult: FAILURE")
                sys.exit(1)

            print(f"      ✓ Extension '{ext_row[0]}' is active (version: {ext_row[1]}).")

        # 5. Inspect database tables and columns
        print("\n[3/4] Inspecting schema tables & columns...")
        all_tables_found = True
        for tbl in EXPECTED_TABLES:
            if tbl in existing_tables:
                cols = [col["name"] for col in inspector.get_columns(tbl)]
                print(f"      ✓ Table '{tbl}' ({len(cols)} columns):")
                print(f"        Columns: {', '.join(cols[:8])}...")
            else:
                print(f"      ✗ Table '{tbl}' is MISSING.")
                all_tables_found = False

        if not all_tables_found:
            print("\n[ERROR] One or more expected tables are missing.")
            print("\nResult: FAILURE")
            sys.exit(1)

        # 6. Verify row counts (should be exactly 0)
        print("\n[4/4] Verifying table row counts...")
        all_empty = True
        row_counts = {}
        with engine.connect() as conn:
            for tbl in EXPECTED_TABLES:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl};")).scalar()
                row_counts[tbl] = count
                if count == 0:
                    print(f"      ✓ Table '{tbl}': {count} rows (empty)")
                else:
                    print(f"      ✗ Table '{tbl}': {count} rows (expected 0 rows)")
                    all_empty = False

        if not all_empty:
            print("\n[ERROR] Tables must be empty in Step 3.")
            print("\nResult: FAILURE")
            sys.exit(1)

        # 7. Final output matching requested format
        print("\n" + "=" * 72)
        print("  STEP 3.6 SCHEMA VERIFICATION REPORT")
        print("=" * 72)
        print("PostgreSQL connection: OK")
        print("pgvector extension: OK")
        print("schemes table: OK")
        print("sources table: OK")
        print("scheme_versions table: OK")
        print("scheme_chunks table: OK")
        print("")
        print("Rows in schemes: 0")
        print("Rows in sources: 0")
        print("Rows in scheme_versions: 0")
        print("Rows in scheme_chunks: 0")
        print("=" * 72)
        print("Result: SUCCESS\n")

    except Exception as e:
        print(f"\n[ERROR] Database operation failed: {e}")
        print("\nResult: FAILURE")
        sys.exit(1)


def re_mask_url(url: str) -> str:
    """Masks password inside database URL for safe logging."""
    import re
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)


if __name__ == "__main__":
    test_database()
