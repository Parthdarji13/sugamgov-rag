"""
scripts/04_ingest_schemes.py
============================
Step 4.1: Scheme Data Validation and Ingestion Pipeline for 'sugamgov-rag'.

This script:
  1. Loads configuration from .env (DATABASE_URL).
  2. Reads the enriched dataset: data/processed/enriched_schemes.parquet.
  3. Executes strict pre-ingestion validation checks:
     - Parquet file exists and is accessible.
     - Row count is exactly 3,397.
     - 'scheme_id' and 'scheme_name' columns exist.
     - All 'scheme_id' values are strictly unique.
     - 'scheme_id' values cover S0001 through S3397 without gaps or duplicates.
     - All 16 required schema fields exist in the dataset.
     - No nulls in non-nullable database fields.
     - Target database table 'schemes' exists and contains all required columns.
  4. Formats and prepares records for PostgreSQL insertion:
     - Converts numpy.ndarray array columns (categories, normalized_tags, states)
       into valid JSON-serialized strings for PostgreSQL JSONB columns.
     - Handles missing/null values safely (e.g. state -> None for SQL NULL).
     - Preserves all 16 enriched columns verbatim without truncating or modifying text.
  5. Safely ingests data into the PostgreSQL 'schemes' table in batches using an
     atomic transaction with ON CONFLICT (scheme_id) DO NOTHING (idempotent).
  6. Leaves all other tables ('sources', 'scheme_versions', 'scheme_chunks') untouched.
  7. Verifies final table row counts and prints a comprehensive audit report.

Usage:
  python scripts/04_ingest_schemes.py
  python scripts/04_ingest_schemes.py --dry-run
  python scripts/04_ingest_schemes.py --batch-size 500
"""

import os
import sys
import re
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np
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
# Constants & Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "enriched_schemes.parquet"
ENV_PATH = PROJECT_ROOT / ".env"

EXPECTED_ROW_COUNT = 3397
EXPECTED_ID_PREFIX = "S"
EXPECTED_ID_MIN = 1
EXPECTED_ID_MAX = 3397

REQUIRED_COLUMNS = [
    "scheme_id",
    "scheme_name",
    "slug",
    "details",
    "benefits",
    "eligibility",
    "application",
    "documents",
    "level",
    "scheme_category",
    "tags",
    "categories",
    "normalized_tags",
    "state",
    "states",
    "state_extraction_method",
]

JSONB_COLUMNS = ["categories", "normalized_tags", "states"]

OTHER_TABLES = ["sources", "scheme_versions", "scheme_chunks"]


def mask_url_password(url: str) -> str:
    """Masks credentials inside database URL for safe console display."""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Loads the enriched parquet dataset using pandas."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Enriched parquet file not found at: {file_path.relative_to(PROJECT_ROOT).as_posix() if file_path.is_relative_to(PROJECT_ROOT) else str(file_path)}"
        )
    return pd.read_parquet(file_path)


def validate_dataset(df: pd.DataFrame, file_path: Path) -> List[Tuple[str, bool, str]]:
    """
    Executes pre-ingestion validation checks on the dataset.
    Returns a list of tuples: (check_name, passed_boolean, detail_message).
    """
    checks = []

    # Check 1: File exists
    file_exists = file_path.is_file()
    checks.append((
        "File exists and is readable",
        file_exists,
        file_path.relative_to(PROJECT_ROOT).as_posix() if file_path.is_relative_to(PROJECT_ROOT) else str(file_path)
    ))

    # Check 2: Row count is exactly 3,397
    actual_rows = len(df)
    row_count_ok = (actual_rows == EXPECTED_ROW_COUNT)
    checks.append((
        f"Row count is exactly {EXPECTED_ROW_COUNT}",
        row_count_ok,
        f"Detected {actual_rows:,} rows (expected {EXPECTED_ROW_COUNT:,})"
    ))

    # Check 3: scheme_id column exists
    scheme_id_col_ok = "scheme_id" in df.columns
    checks.append((
        "'scheme_id' column exists",
        scheme_id_col_ok,
        "Column found" if scheme_id_col_ok else "MISSING 'scheme_id' column"
    ))

    # Check 4: scheme_name column exists
    scheme_name_col_ok = "scheme_name" in df.columns
    checks.append((
        "'scheme_name' column exists",
        scheme_name_col_ok,
        "Column found" if scheme_name_col_ok else "MISSING 'scheme_name' column"
    ))

    # Check 5: All scheme_id values are strictly unique
    if scheme_id_col_ok:
        unique_ids = df["scheme_id"].nunique()
        unique_ok = (unique_ids == EXPECTED_ROW_COUNT)
        checks.append((
            "All 'scheme_id' values are strictly unique",
            unique_ok,
            f"{unique_ids:,} unique IDs found" if unique_ok else f"Found only {unique_ids} unique IDs (expected {EXPECTED_ROW_COUNT})"
        ))
    else:
        checks.append(("All 'scheme_id' values are strictly unique", False, "Skipped due to missing column"))

    # Check 6: scheme_id sequence covers S0001 through S3397 exactly
    if scheme_id_col_ok:
        expected_ids = {f"{EXPECTED_ID_PREFIX}{i:04d}" for i in range(EXPECTED_ID_MIN, EXPECTED_ID_MAX + 1)}
        actual_ids = set(df["scheme_id"])
        missing_ids = expected_ids - actual_ids
        extra_ids = actual_ids - expected_ids
        sequence_ok = (len(missing_ids) == 0 and len(extra_ids) == 0)
        detail = "Exact match S0001 through S3397"
        if missing_ids:
            detail = f"Missing {len(missing_ids)} IDs (e.g. {sorted(list(missing_ids))[:3]})"
        elif extra_ids:
            detail = f"Unexpected extra {len(extra_ids)} IDs (e.g. {sorted(list(extra_ids))[:3]})"
        checks.append((
            f"scheme_id sequence covers S{EXPECTED_ID_MIN:04d} through S{EXPECTED_ID_MAX:04d}",
            sequence_ok,
            detail
        ))
    else:
        checks.append(("scheme_id sequence covers S0001 through S3397", False, "Skipped due to missing column"))

    # Check 7: All 16 required columns exist
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    cols_ok = len(missing_cols) == 0
    checks.append((
        f"All {len(REQUIRED_COLUMNS)} required schema columns exist in dataset",
        cols_ok,
        f"All {len(REQUIRED_COLUMNS)} columns present" if cols_ok else f"Missing columns: {missing_cols}"
    ))

    # Check 8: Non-nullable fields have no nulls
    null_issues = []
    if scheme_id_col_ok and df["scheme_id"].isna().sum() > 0:
        null_issues.append(f"scheme_id has {df['scheme_id'].isna().sum()} nulls")
    if scheme_name_col_ok and df["scheme_name"].isna().sum() > 0:
        null_issues.append(f"scheme_name has {df['scheme_name'].isna().sum()} nulls")
    non_null_ok = len(null_issues) == 0
    checks.append((
        "Non-nullable fields ('scheme_id', 'scheme_name') have no nulls",
        non_null_ok,
        "No nulls detected" if non_null_ok else "; ".join(null_issues)
    ))

    return checks


def prepare_records_for_insertion(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Transforms dataframe rows into database-ready dictionaries:
      - numpy.ndarray lists converted to python lists and serialized to JSON.
      - NaNs / empty state values safely converted to None for SQL NULL.
      - String columns preserved without alteration.
    """
    records = []
    # Using df.to_dict('records') for high performance
    for raw in df.to_dict(orient="records"):
        # JSONB column 1: categories
        cat_val = raw.get("categories")
        if isinstance(cat_val, np.ndarray):
            cat_list = cat_val.tolist()
        elif isinstance(cat_val, list):
            cat_list = cat_val
        elif pd.isna(cat_val) or cat_val is None:
            cat_list = []
        else:
            cat_list = [str(cat_val)]
        categories_json = json.dumps(cat_list, ensure_ascii=False)

        # JSONB column 2: normalized_tags
        tags_val = raw.get("normalized_tags")
        if isinstance(tags_val, np.ndarray):
            tags_list = tags_val.tolist()
        elif isinstance(tags_val, list):
            tags_list = tags_val
        elif pd.isna(tags_val) or tags_val is None:
            tags_list = []
        else:
            tags_list = [str(tags_val)]
        normalized_tags_json = json.dumps(tags_list, ensure_ascii=False)

        # JSONB column 3: states
        states_val = raw.get("states")
        if isinstance(states_val, np.ndarray):
            states_list = states_val.tolist()
        elif isinstance(states_val, list):
            states_list = states_val
        elif pd.isna(states_val) or states_val is None:
            states_list = []
        else:
            states_list = [str(states_val)]
        states_json = json.dumps(states_list, ensure_ascii=False)

        # Helper for text fields (preserve empty string, convert NaN/None to None)
        def to_nullable_text(v: Any) -> Any:
            if v is None or pd.isna(v):
                return None
            return str(v)

        record = {
            "scheme_id": str(raw["scheme_id"]).strip(),
            "scheme_name": str(raw["scheme_name"]).strip(),
            "slug": to_nullable_text(raw.get("slug")),
            "details": to_nullable_text(raw.get("details")),
            "benefits": to_nullable_text(raw.get("benefits")),
            "eligibility": to_nullable_text(raw.get("eligibility")),
            "application": to_nullable_text(raw.get("application")),
            "documents": to_nullable_text(raw.get("documents")),
            "level": to_nullable_text(raw.get("level")),
            "scheme_category": to_nullable_text(raw.get("scheme_category")),
            "tags": to_nullable_text(raw.get("tags")),
            "categories": categories_json,
            "normalized_tags": normalized_tags_json,
            "state": to_nullable_text(raw.get("state")),
            "states": states_json,
            "state_extraction_method": to_nullable_text(raw.get("state_extraction_method")),
        }
        records.append(record)

    return records


def ingest_schemes(file_path: Path, batch_size: int = 500, dry_run: bool = False):
    """Main execution orchestrator for Step 4.1 scheme ingestion."""
    print("=" * 76)
    print("  sugamgov-rag | Step 4.1: Schemes Ingestion Pipeline")
    print("=" * 76)

    # -------------------------------------------------------------------------
    # Phase 1: Read parquet dataset
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Loading Enriched Parquet File...")
    rel_path = file_path.relative_to(PROJECT_ROOT).as_posix() if file_path.is_relative_to(PROJECT_ROOT) else str(file_path)
    print(f"      Input file: {rel_path}")

    if not file_path.exists():
        print(f"\n[ERROR] File not found: {file_path}")
        print("Validation Result: FAILED")
        sys.exit(1)

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"      File size:  {file_size_mb:.2f} MB")

    df = load_dataset(file_path)
    print(f"      Rows loaded: {len(df):,}")
    print(f"      Columns detected ({len(df.columns)}):")
    for col in df.columns:
        print(f"        - {col}")

    # -------------------------------------------------------------------------
    # Phase 2: Validate dataset integrity before any database interaction
    # -------------------------------------------------------------------------
    print(f"\n[2/5] Running Pre-Ingestion Data Validation...")
    checks = validate_dataset(df, file_path)

    all_passed = True
    for idx, (check_name, passed, detail) in enumerate(checks, 1):
        status_symbol = "✓" if passed else "✗"
        status_label = "PASSED" if passed else "FAILED"
        print(f"      {status_symbol} Check {idx}: {check_name} -> [{status_label}] ({detail})")
        if not passed:
            all_passed = False

    if not all_passed:
        print("\n" + "!" * 76)
        print("[ERROR] Pre-ingestion validation failed. Ingestion aborted.")
        print("No database connection was established, and no records were inserted.")
        print("!" * 76)
        print("Result: FAILURE")
        sys.exit(1)

    print(f"      Validation Result: ALL PRE-INGESTION CHECKS PASSED ({len(checks)}/{len(checks)})")

    # -------------------------------------------------------------------------
    # Phase 3: Connect to Database & Verify Target Table Structure
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Connecting to Database & Verifying Schema...")
    if not ENV_PATH.exists():
        print(f"\n[ERROR] .env file not found at: {ENV_PATH.relative_to(PROJECT_ROOT).as_posix()}")
        print("Result: FAILURE")
        sys.exit(1)

    load_dotenv(dotenv_path=ENV_PATH)
    database_url = os.getenv("DATABASE_URL")

    if not database_url or "YOUR_PASSWORD" in database_url:
        print("\n[ERROR] DATABASE_URL in .env is missing or unconfigured.")
        print("Result: FAILURE")
        sys.exit(1)

    masked_url = mask_url_password(database_url)
    print(f"      Database URL: {masked_url}")

    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("      ✓ PostgreSQL connection established.")

        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        if "schemes" not in existing_tables:
            print("\n[ERROR] Target table 'schemes' does not exist in the database.")
            print("Please run database migration first: python scripts/03_test_database.py --migrate")
            print("Result: FAILURE")
            sys.exit(1)

        db_columns = {c["name"] for c in inspector.get_columns("schemes")}
        missing_db_columns = [col for col in REQUIRED_COLUMNS if col not in db_columns]
        if missing_db_columns:
            print(f"\n[ERROR] Target table 'schemes' is missing columns: {missing_db_columns}")
            print("Result: FAILURE")
            sys.exit(1)

        print(f"      ✓ Table 'schemes' detected with all {len(REQUIRED_COLUMNS)} required columns.")

        # Check existing row counts
        with engine.connect() as conn:
            schemes_initial_count = conn.execute(text("SELECT COUNT(*) FROM schemes;")).scalar()
            sources_initial_count = conn.execute(text("SELECT COUNT(*) FROM sources;")).scalar()
            versions_initial_count = conn.execute(text("SELECT COUNT(*) FROM scheme_versions;")).scalar()
            chunks_initial_count = conn.execute(text("SELECT COUNT(*) FROM scheme_chunks;")).scalar()

        print("      Current table row counts:")
        print(f"        - schemes:         {schemes_initial_count:,} rows")
        print(f"        - sources:         {sources_initial_count:,} rows")
        print(f"        - scheme_versions: {versions_initial_count:,} rows")
        print(f"        - scheme_chunks:   {chunks_initial_count:,} rows")

    except Exception as e:
        print(f"\n[ERROR] Database connection/inspection failed: {e}")
        print("Result: FAILURE")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Phase 4: Prepare records
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Preparing Records for Ingestion...")
    records = prepare_records_for_insertion(df)
    print(f"      Records prepared: {len(records):,}")
    print(f"      JSONB fields formatted: {', '.join(JSONB_COLUMNS)}")
    print(f"      Number of records ready for insertion: {len(records):,}")

    if dry_run:
        print("\n" + "=" * 76)
        print("  DRY RUN COMPLETED (No records were inserted into the database)")
        print("=" * 76)
        print(f"Input file:             {rel_path}")
        print(f"Rows validated:         {len(df):,}")
        print(f"Validation status:      PASSED (all {len(checks)} checks)")
        print(f"Records ready:          {len(records):,}")
        print(f"Target table:           schemes (currently {schemes_initial_count:,} rows)")
        print("Mode:                   --dry-run")
        print("Result: SUCCESS (DRY RUN)")
        print("=" * 76)
        return

    # Check if already fully populated
    if schemes_initial_count == EXPECTED_ROW_COUNT:
        print(f"\n[*] Table 'schemes' already contains exactly {EXPECTED_ROW_COUNT:,} records.")
        print("    No additional records will be inserted (idempotent check).")

    # -------------------------------------------------------------------------
    # Phase 5: Batch Ingestion
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Ingesting Records into 'schemes' Table (batch size: {batch_size})...")

    insert_sql = text("""
        INSERT INTO schemes (
            scheme_id,
            scheme_name,
            slug,
            details,
            benefits,
            eligibility,
            application,
            documents,
            level,
            scheme_category,
            tags,
            categories,
            normalized_tags,
            state,
            states,
            state_extraction_method
        ) VALUES (
            :scheme_id,
            :scheme_name,
            :slug,
            :details,
            :benefits,
            :eligibility,
            :application,
            :documents,
            :level,
            :scheme_category,
            :tags,
            :categories,
            :normalized_tags,
            :state,
            :states,
            :state_extraction_method
        )
        ON CONFLICT (scheme_id) DO NOTHING;
    """)

    num_batches = (len(records) + batch_size - 1) // batch_size

    try:
        with engine.begin() as trans_conn:
            for b_idx in range(num_batches):
                start_i = b_idx * batch_size
                end_i = min((b_idx + 1) * batch_size, len(records))
                batch_data = records[start_i:end_i]

                trans_conn.execute(insert_sql, batch_data)
                print(f"      Batch {b_idx + 1}/{num_batches}: Processed records {start_i + 1:,} to {end_i:,} [OK]")

        print("      ✓ Database transaction committed successfully.")

    except Exception as e:
        print(f"\n[ERROR] Ingestion failed during batch execution: {e}")
        print("Transaction automatically rolled back. No partial records committed.")
        print("Result: FAILURE")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Post-Ingestion Verification
    # -------------------------------------------------------------------------
    print(f"\nVerifying final database state...")
    try:
        with engine.connect() as conn:
            schemes_final_count = conn.execute(text("SELECT COUNT(*) FROM schemes;")).scalar()
            sources_final_count = conn.execute(text("SELECT COUNT(*) FROM sources;")).scalar()
            versions_final_count = conn.execute(text("SELECT COUNT(*) FROM scheme_versions;")).scalar()
            chunks_final_count = conn.execute(text("SELECT COUNT(*) FROM scheme_chunks;")).scalar()

            # Verify sample record retrieved from DB
            sample_row = conn.execute(
                text("SELECT scheme_id, scheme_name, categories, states FROM schemes WHERE scheme_id = 'S0001';")
            ).fetchone()

        newly_inserted = schemes_final_count - schemes_initial_count

        print("\n" + "=" * 76)
        print("  STEP 4.1 SCHEMES INGESTION REPORT")
        print("=" * 76)
        print(f"Input file:                         {rel_path}")
        print(f"Rows loaded from parquet:           {len(df):,}")
        print(f"Pre-ingestion validation:           PASSED ({len(checks)}/{len(checks)} checks)")
        print(f"Records prepared for insertion:     {len(records):,}")
        print(f"Initial rows in schemes table:      {schemes_initial_count:,}")
        print(f"Newly inserted rows in schemes:     {newly_inserted:,}")
        print(f"Final rows in schemes table:        {schemes_final_count:,}")
        print(f"Verification sample (S0001):")
        if sample_row:
            print(f"  - scheme_id:   {sample_row[0]}")
            print(f"  - scheme_name: {sample_row[1][:60]}...")
            print(f"  - categories:  {sample_row[2]}")
            print(f"  - states:      {sample_row[3]}")
        print("")
        print("Untouched tables verification:")
        print(f"  - sources:         {sources_final_count} rows (expected: 0)")
        print(f"  - scheme_versions: {versions_final_count} rows (expected: 0)")
        print(f"  - scheme_chunks:   {chunks_final_count} rows (expected: 0)")
        print("=" * 76)

        # Integrity assertions
        if schemes_final_count != EXPECTED_ROW_COUNT:
            print(f"\n[ERROR] schemes count ({schemes_final_count}) does not match expected ({EXPECTED_ROW_COUNT})")
            print("Result: FAILURE")
            sys.exit(1)

        if sources_final_count != 0 or versions_final_count != 0 or chunks_final_count != 0:
            print("\n[ERROR] Other tables must remain untouched (0 rows).")
            print("Result: FAILURE")
            sys.exit(1)

        print("Result: SUCCESS\n")

    except Exception as e:
        print(f"\n[ERROR] Post-ingestion verification failed: {e}")
        print("Result: FAILURE")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Step 4.1: Ingest enriched government schemes into PostgreSQL"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_PARQUET_PATH,
        help=f"Path to input parquet file (default: {DEFAULT_PARQUET_PATH.relative_to(PROJECT_ROOT).as_posix()})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for database insertion (default: 500)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and test connection without inserting records"
    )

    args = parser.parse_args()

    ingest_schemes(
        file_path=args.file,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
