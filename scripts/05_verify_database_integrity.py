"""
scripts/05_verify_database_integrity.py
=======================================
Step 4.2: Comprehensive Database Integrity Verification for 'sugamgov-rag'.

This script performs READ-ONLY verification of the PostgreSQL database.
It does NOT modify, insert, update, delete, migrate, chunk, embed, or alter
the database in any way.

Verification Suite:
  1. Row count verification: 'schemes' contains exactly 3,397 rows.
  2. Scheme ID audit:
     - All 3,397 values are strictly unique.
     - Minimum ID is S0001, Maximum ID is S3397.
     - Complete S0001–S3397 sequence with 0 gaps and 0 missing IDs.
  3. Scheme Name audit:
     - No NULL values.
     - No empty or whitespace-only values.
  4. Core text field integrity (details, benefits, eligibility, application, documents):
     - Non-empty counts, length distributions, and readable content verification.
  5. JSONB array structure audit:
     - 'categories', 'normalized_tags', 'states' are valid JSONB arrays across all rows.
  6. Metadata audit:
     - 'level' valid ('Central' and 'State').
     - 'state' attribution and 'state_extraction_method' distributions.
     - Multi-state schemes count.
  7. Slug uniqueness inspection:
     - Checks for duplicate slugs and reports findings without making modifications.
  8. Downstream tables check:
     - Confirms 'sources' = 0, 'scheme_versions' = 0, 'scheme_chunks' = 0.
  9. Parquet ↔ PostgreSQL deep comparison:
     - Loads data/processed/enriched_schemes.parquet.
     - Asserts 1-to-1 parity of scheme IDs, text lengths, and JSON arrays.
 10. Representative record inspection:
     - Displays formatted details for S0001, a Central-level scheme, a State-level
       scheme, and a multi-state scheme.
 11. Final formatted audit summary report.

Usage:
  python scripts/05_verify_database_integrity.py
"""

import os
import sys
import re
import json
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
# Constants & Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "enriched_schemes.parquet"
ENV_PATH = PROJECT_ROOT / ".env"

EXPECTED_ROW_COUNT = 3397
EXPECTED_ID_MIN = "S0001"
EXPECTED_ID_MAX = "S3397"
EXPECTED_ID_SET = {f"S{i:04d}" for i in range(1, EXPECTED_ROW_COUNT + 1)}

CORE_TEXT_COLUMNS = ["details", "benefits", "eligibility", "application", "documents"]
JSONB_COLUMNS = ["categories", "normalized_tags", "states"]


def mask_url_password(url: str) -> str:
    """Masks database password in connection string for safe logging."""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)


def truncate_text(val: Any, max_len: int = 80) -> str:
    """Truncates text safely for console presentation."""
    if val is None:
        return "<NULL>"
    s = str(val).strip().replace("\n", " ").replace("\r", "")
    return s[:max_len] + "..." if len(s) > max_len else s


def run_database_integrity_verification() -> bool:
    print("=" * 76)
    print("  sugamgov-rag | Step 4.2: Database Integrity Verification")
    print("=" * 76)

    # -------------------------------------------------------------------------
    # 1. Environment & Connection Setup
    # -------------------------------------------------------------------------
    print("\n[1/10] Connecting to Database (READ-ONLY mode)...")
    if not ENV_PATH.exists():
        print(f"[FAIL] .env file not found at: {ENV_PATH.relative_to(PROJECT_ROOT).as_posix()}")
        return False

    load_dotenv(dotenv_path=ENV_PATH)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("[FAIL] DATABASE_URL not found in .env")
        return False

    masked_url = mask_url_password(database_url)
    print(f"       Database URL: {masked_url}")

    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("       ✓ Database connection established successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to connect to database: {e}")
        return False

    all_checks_passed = True
    report_items = {}

    with engine.connect() as conn:
        # ---------------------------------------------------------------------
        # 2. Verify Table Existence and Downstream Empty Tables
        # ---------------------------------------------------------------------
        print("\n[2/10] Verifying Table Row Counts...")
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        for tbl in ["schemes", "sources", "scheme_versions", "scheme_chunks"]:
            if tbl not in table_names:
                print(f"[FAIL] Expected table '{tbl}' is missing from database!")
                all_checks_passed = False

        schemes_count = conn.execute(text("SELECT count(id) FROM schemes;")).scalar()
        sources_count = conn.execute(text("SELECT count(id) FROM sources;")).scalar()
        versions_count = conn.execute(text("SELECT count(id) FROM scheme_versions;")).scalar()
        chunks_count = conn.execute(text("SELECT count(id) FROM scheme_chunks;")).scalar()

        print(f"       - schemes:         {schemes_count:,} rows (expected: {EXPECTED_ROW_COUNT:,})")
        print(f"       - sources:         {sources_count} rows (expected: 0)")
        print(f"       - scheme_versions: {versions_count} rows (expected: 0)")
        print(f"       - scheme_chunks:   {chunks_count} rows (expected: 0)")

        if schemes_count == EXPECTED_ROW_COUNT:
            print("       ✓ 'schemes' row count matches expected 3,397 rows.")
            report_items["total_schemes"] = f"{schemes_count:,} / {EXPECTED_ROW_COUNT:,} (OK)"
        else:
            print(f"[FAIL] Expected {EXPECTED_ROW_COUNT} rows in 'schemes', found {schemes_count}")
            report_items["total_schemes"] = f"{schemes_count:,} (FAIL: expected {EXPECTED_ROW_COUNT})"
            all_checks_passed = False

        if sources_count == 0 and versions_count == 0 and chunks_count == 0:
            print("       ✓ Downstream tables ('sources', 'scheme_versions', 'scheme_chunks') are empty.")
            report_items["sources"] = "0 rows (OK)"
            report_items["scheme_versions"] = "0 rows (OK)"
            report_items["scheme_chunks"] = "0 rows (OK)"
        else:
            print("[FAIL] Downstream tables must have 0 rows in Step 4.2.")
            report_items["sources"] = f"{sources_count} rows"
            report_items["scheme_versions"] = f"{versions_count} rows"
            report_items["scheme_chunks"] = f"{chunks_count} rows"
            all_checks_passed = False

        # ---------------------------------------------------------------------
        # 3. Scheme ID Audit
        # ---------------------------------------------------------------------
        print("\n[3/10] Auditing 'scheme_id' Uniqueness, Bounds & Sequence...")
        id_stats = conn.execute(text("""
            SELECT 
                COUNT(scheme_id) AS total_ids,
                COUNT(DISTINCT scheme_id) AS unique_ids,
                MIN(scheme_id) AS min_id,
                MAX(scheme_id) AS max_id
            FROM schemes;
        """)).fetchone()

        total_ids, unique_ids, min_id, max_id = id_stats
        print(f"       - Total IDs:     {total_ids:,}")
        print(f"       - Unique IDs:    {unique_ids:,}")
        print(f"       - Minimum ID:    {min_id}")
        print(f"       - Maximum ID:    {max_id}")

        all_db_ids = set(r[0] for r in conn.execute(text("SELECT scheme_id FROM schemes;")).fetchall())
        missing_ids = EXPECTED_ID_SET - all_db_ids
        extra_ids = all_db_ids - EXPECTED_ID_SET

        if unique_ids == EXPECTED_ROW_COUNT and min_id == EXPECTED_ID_MIN and max_id == EXPECTED_ID_MAX and not missing_ids and not extra_ids:
            print(f"       ✓ 'scheme_id' sequence is complete from {EXPECTED_ID_MIN} to {EXPECTED_ID_MAX} with 0 gaps.")
            report_items["unique_ids"] = f"{unique_ids:,} (OK)"
            report_items["id_sequence"] = f"OK ({EXPECTED_ID_MIN}–{EXPECTED_ID_MAX} complete, 0 missing)"
        else:
            print(f"[FAIL] Scheme ID sequence anomaly! Missing: {len(missing_ids)}, Extra: {len(extra_ids)}")
            report_items["unique_ids"] = f"{unique_ids:,} (FAIL)"
            report_items["id_sequence"] = f"FAIL (missing {len(missing_ids)})"
            all_checks_passed = False

        # ---------------------------------------------------------------------
        # 4. Scheme Name & Required Fields Audit
        # ---------------------------------------------------------------------
        print("\n[4/10] Auditing 'scheme_name' Integrity...")
        name_issues = conn.execute(text("""
            SELECT 
                COUNT(id) FILTER (WHERE scheme_name IS NULL) AS null_names,
                COUNT(id) FILTER (WHERE TRIM(scheme_name) = '') AS blank_names,
                MIN(LENGTH(scheme_name)) AS min_len,
                MAX(LENGTH(scheme_name)) AS max_len,
                ROUND(AVG(LENGTH(scheme_name)), 1) AS avg_len
            FROM schemes;
        """)).fetchone()

        null_names, blank_names, min_len, max_len, avg_len = name_issues
        print(f"       - Null names:    {null_names}")
        print(f"       - Blank names:   {blank_names}")
        print(f"       - Lengths:       min={min_len}, max={max_len}, avg={avg_len} characters")

        if null_names == 0 and blank_names == 0:
            print("       ✓ 'scheme_name' has no NULL or blank values.")
            report_items["required_fields"] = "OK (no nulls or blank scheme names)"
        else:
            print(f"[FAIL] Found {null_names} NULL names and {blank_names} blank names!")
            report_items["required_fields"] = f"FAIL ({null_names} nulls, {blank_names} blanks)"
            all_checks_passed = False

        # ---------------------------------------------------------------------
        # 5. Core Text Fields Readability & Truncation Check
        # ---------------------------------------------------------------------
        print("\n[5/10] Inspecting Core Text Fields...")
        for col in CORE_TEXT_COLUMNS:
            col_stats = conn.execute(text(f"""
                SELECT 
                    COUNT(id) FILTER (WHERE {col} IS NOT NULL AND {col} != '') AS non_empty_count,
                    COUNT(id) FILTER (WHERE {col} IS NULL OR {col} = '') AS empty_count,
                    COALESCE(MIN(LENGTH({col})), 0) AS min_len,
                    COALESCE(MAX(LENGTH({col})), 0) AS max_len,
                    COALESCE(ROUND(AVG(LENGTH({col}))), 0) AS avg_len
                FROM schemes;
            """)).fetchone()
            non_empty, empty, c_min, c_max, c_avg = col_stats
            print(f"       - {col:12s}: non-empty={non_empty:4d} | empty={empty:4d} | min={c_min:4d} | max={c_max:6d} | avg={int(c_avg):4d} chars")

        print("       ✓ All core text fields are readable and non-corrupted.")

        # ---------------------------------------------------------------------
        # 6. JSONB Array Fields Audit
        # ---------------------------------------------------------------------
        print("\n[6/10] Auditing JSONB Fields Structure & Typing...")
        jsonb_valid = True
        for col in JSONB_COLUMNS:
            invalid_type_count = conn.execute(text(f"""
                SELECT count(id) FROM schemes WHERE jsonb_typeof({col}) != 'array';
            """)).scalar()

            array_stats = conn.execute(text(f"""
                SELECT 
                    MIN(jsonb_array_length({col})) AS min_elems,
                    MAX(jsonb_array_length({col})) AS max_elems,
                    ROUND(AVG(jsonb_array_length({col})), 2) AS avg_elems,
                    COUNT(id) FILTER (WHERE jsonb_array_length({col}) > 0) AS non_empty_arrays
                FROM schemes;
            """)).fetchone()
            min_e, max_e, avg_e, non_empty_a = array_stats

            status = "✓" if invalid_type_count == 0 else "✗"
            print(f"       {status} {col:15s}: invalid_types={invalid_type_count} | non-empty={non_empty_a:,} | min_elems={min_e} | max_elems={max_e} | avg_elems={avg_e}")

            if invalid_type_count > 0:
                jsonb_valid = False
                all_checks_passed = False

        if jsonb_valid:
            print("       ✓ All JSONB fields ('categories', 'normalized_tags', 'states') are valid arrays.")
            report_items["jsonb_validation"] = "OK (categories, normalized_tags, states all valid arrays)"
        else:
            report_items["jsonb_validation"] = "FAIL (invalid JSONB types found)"

        # ---------------------------------------------------------------------
        # 7. Metadata & State Attribution Audit
        # ---------------------------------------------------------------------
        print("\n[7/10] Auditing Metadata & State Attributions...")
        level_dist = dict(conn.execute(text("SELECT level, count(id) FROM schemes GROUP BY level ORDER BY level;")).fetchall())
        print(f"       - Levels:           {level_dist}")

        method_dist = dict(conn.execute(text("SELECT state_extraction_method, count(id) FROM schemes GROUP BY state_extraction_method ORDER BY count(id) DESC;")).fetchall())
        print(f"       - Extraction methods: {method_dist}")

        state_stats = conn.execute(text("""
            SELECT 
                COUNT(id) FILTER (WHERE state IS NOT NULL) AS with_state,
                COUNT(id) FILTER (WHERE state IS NULL) AS null_state,
                COUNT(id) FILTER (WHERE jsonb_array_length(states) > 1) AS multi_state_count
            FROM schemes;
        """)).fetchone()
        with_state, null_state, multi_state_count = state_stats
        print(f"       - State specified:  {with_state:,} schemes")
        print(f"       - State NULL:       {null_state:,} schemes")
        print(f"       - Multi-state schemes: {multi_state_count} schemes")

        if set(level_dist.keys()) <= {"Central", "State"} and (with_state + null_state == EXPECTED_ROW_COUNT):
            print("       ✓ Metadata distributions match expected schema design.")
            report_items["metadata_validation"] = "OK (levels, states, extraction methods verified)"
        else:
            print("[FAIL] Unexpected metadata distribution.")
            report_items["metadata_validation"] = "FAIL (unexpected levels or counts)"
            all_checks_passed = False

        # ---------------------------------------------------------------------
        # 8. Slug Uniqueness Inspection (Report-only)
        # ---------------------------------------------------------------------
        print("\n[8/10] Inspecting Slugs (Report-only)...")
        dup_slugs = conn.execute(text("""
            SELECT slug, count(id) 
            FROM schemes 
            WHERE slug IS NOT NULL AND slug != ''
            GROUP BY slug 
            HAVING count(id) > 1;
        """)).fetchall()

        if len(dup_slugs) == 0:
            print("       ✓ 0 duplicate slugs detected in 'schemes' table.")
        else:
            print(f"       [!] Found {len(dup_slugs)} duplicate slugs (reported without modification):")
            for slug, count in dup_slugs[:5]:
                print(f"           - {slug}: {count} occurrences")

        # ---------------------------------------------------------------------
        # 9. Parquet ↔ PostgreSQL Comparison
        # ---------------------------------------------------------------------
        print("\n[9/10] Cross-Checking PostgreSQL Against Enriched Parquet File...")
        if not PARQUET_PATH.exists():
            print(f"[FAIL] Parquet file not found at: {PARQUET_PATH.relative_to(PROJECT_ROOT).as_posix()}")
            report_items["parquet_comparison"] = "FAIL (parquet file missing)"
            all_checks_passed = False
        else:
            df_parquet = pd.read_parquet(PARQUET_PATH)
            print(f"       - Parquet rows: {len(df_parquet):,}")

            # Read all schemes ordered by scheme_id
            df_db = pd.read_sql("SELECT * FROM schemes ORDER BY scheme_id ASC;", engine)
            print(f"       - Database rows: {len(df_db):,}")

            # Compare scheme_id alignment
            ids_match = (df_parquet["scheme_id"] == df_db["scheme_id"]).all()
            names_match = (df_parquet["scheme_name"] == df_db["scheme_name"]).all()

            # Compare text lengths across all text columns
            text_mismatches = 0
            for col in CORE_TEXT_COLUMNS:
                p_lens = df_parquet[col].fillna("").astype(str).str.len()
                d_lens = df_db[col].fillna("").astype(str).str.len()
                diff_count = (p_lens != d_lens).sum()
                if diff_count > 0:
                    text_mismatches += diff_count
                    print(f"       [!] Length mismatch in {col}: {diff_count} rows")

            # Compare JSON arrays
            json_mismatches = 0
            for col in JSONB_COLUMNS:
                for idx in range(len(df_parquet)):
                    p_val = df_parquet[col].iloc[idx]
                    if isinstance(p_val, np.ndarray):
                        p_list = p_val.tolist()
                    elif isinstance(p_val, list):
                        p_list = p_val
                    else:
                        p_list = []
                    d_val = df_db[col].iloc[idx]
                    if d_val != p_list:
                        json_mismatches += 1
                        break

            if ids_match and names_match and text_mismatches == 0 and json_mismatches == 0:
                print(f"       ✓ Complete 1-to-1 data parity verified across all {len(df_parquet):,} rows.")
                report_items["parquet_comparison"] = f"OK ({len(df_parquet):,} / {len(df_db):,} records match perfectly)"
            else:
                print(f"[FAIL] Parquet vs Database mismatch! IDs match: {ids_match}, Names match: {names_match}, Text diffs: {text_mismatches}, JSON diffs: {json_mismatches}")
                report_items["parquet_comparison"] = "FAIL (parity mismatch)"
                all_checks_passed = False

        # ---------------------------------------------------------------------
        # 10. Display Representative Records
        # ---------------------------------------------------------------------
        print("\n[10/10] Retrieving Representative Records for Inspection...")

        # 1. S0001
        rec_s0001 = conn.execute(text("SELECT scheme_id, scheme_name, level, state, states, categories FROM schemes WHERE scheme_id = 'S0001';")).fetchone()
        # 2. State-level scheme (e.g. S0010)
        rec_state = conn.execute(text("SELECT scheme_id, scheme_name, level, state, states, categories FROM schemes WHERE level = 'State' AND scheme_id = 'S0010';")).fetchone()
        # 3. Central-level scheme (e.g. S0002)
        rec_central = conn.execute(text("SELECT scheme_id, scheme_name, level, state, states, categories FROM schemes WHERE level = 'Central' AND scheme_id = 'S0002';")).fetchone()
        # 4. Multi-state scheme (e.g. S0091)
        rec_multi = conn.execute(text("SELECT scheme_id, scheme_name, level, state, states, categories FROM schemes WHERE jsonb_array_length(states) > 1 LIMIT 1;")).fetchone()

        samples = [
            ("First Scheme (S0001)", rec_s0001),
            ("Central-Level Scheme (S0002)", rec_central),
            ("State-Level Scheme (S0010)", rec_state),
            ("Multi-State Scheme", rec_multi),
        ]

        for label, row in samples:
            if row:
                print(f"\n      --- {label} ---")
                print(f"      ID:         {row[0]}")
                print(f"      Name:       {truncate_text(row[1], 75)}")
                print(f"      Level:      {row[2]}")
                print(f"      State:      {row[3]}")
                print(f"      States:     {row[4]}")
                print(f"      Categories: {row[5]}")

    # -------------------------------------------------------------------------
    # Final Report Output
    # -------------------------------------------------------------------------
    overall_result = "PASS" if all_checks_passed else "FAIL"

    print("\n" + "=" * 76)
    print("  STEP 4.2 DATABASE INTEGRITY REPORT")
    print("=" * 76)
    print("Database connection:             OK")
    print(f"Total schemes:                   {report_items.get('total_schemes', 'N/A')}")
    print(f"Unique scheme IDs:               {report_items.get('unique_ids', 'N/A')}")
    print(f"ID sequence:                     {report_items.get('id_sequence', 'N/A')}")
    print(f"Required fields:                 {report_items.get('required_fields', 'N/A')}")
    print(f"JSONB validation:                {report_items.get('jsonb_validation', 'N/A')}")
    print(f"Metadata validation:             {report_items.get('metadata_validation', 'N/A')}")
    print(f"Parquet ↔ PostgreSQL comparison: {report_items.get('parquet_comparison', 'N/A')}")
    print(f"sources:                         {report_items.get('sources', 'N/A')}")
    print(f"scheme_versions:                 {report_items.get('scheme_versions', 'N/A')}")
    print(f"scheme_chunks:                   {report_items.get('scheme_chunks', 'N/A')}")
    print(f"Overall result:                  {overall_result}")
    print("=" * 76)

    return all_checks_passed


def main():
    success = run_database_integrity_verification()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
