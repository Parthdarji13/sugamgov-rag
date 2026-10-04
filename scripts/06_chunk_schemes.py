"""
scripts/06_chunk_schemes.py
===========================
Step 5.1: RAG Document Chunking Pipeline for 'sugamgov-rag'.

This script:
  1. Connects to PostgreSQL and reads all scheme records from the 'schemes' table.
  2. Generates deterministic, retrieval-friendly text chunks from 6 textual fields:
     - scheme_name
     - details
     - benefits
     - eligibility
     - application
     - documents
  3. Prepends contextual headers (scheme name, section, metadata) to each chunk so
     that every chunk contains complete standalone context for dense & lexical retrieval.
  4. Applies deterministic chunk splitting for long sections (targeting 500–800 words
     with ~60 words overlap), preserving sentence and paragraph boundaries without cutting words.
  5. Assigns deterministic chunk IDs: {scheme_id}_{field_name}_{chunk_index}.
  6. Attaches rich metadata (JSONB) with strict RFC 8259 JSON compliance:
     - Recursively sanitizes all NaN, NA, Infinity, and null values into JSON null.
     - Guarantees missing states are serialized as `"state": null` (never NaN or "nan").
     - Serializes with allow_nan=False to strictly prevent non-compliant tokens.
  7. Supports '--dry-run' for end-to-end chunk generation, distribution analysis, and
     strict JSON validation without modifying the database.
  8. Safely and atomically inserts chunks into 'scheme_chunks' with embedding = NULL.
  9. Verifies table integrity post-insertion and outputs the required Step 5.1 report.

Usage:
  python scripts/06_chunk_schemes.py --dry-run
  python scripts/06_chunk_schemes.py
  python scripts/06_chunk_schemes.py --batch-size 1000
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
ENV_PATH = PROJECT_ROOT / ".env"

CHUNK_TARGET_WORDS = 600
CHUNK_MAX_WORDS = 800
CHUNK_OVERLAP_WORDS = 60

TEXT_FIELDS = [
    "scheme_name",
    "details",
    "benefits",
    "eligibility",
    "application",
    "documents",
]


def mask_url_password(url: str) -> str:
    """Masks database password in connection string for safe logging."""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)


def sanitize_json_value(val: Any) -> Any:
    """
    Recursively sanitizes values for strict RFC 8259 JSON compliance:
    - Converts pandas NaN, numpy.nan, float('nan'), pandas.NA, None to Python None (JSON null).
    - Converts float('inf'), float('-inf') to None.
    - Sanitizes strings (e.g. 'nan', 'none' case-insensitive representations).
    - Recursively processes lists, tuples, numpy arrays, and dictionaries.
    """
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    if isinstance(val, (list, tuple, np.ndarray)):
        cleaned = [sanitize_json_value(item) for item in val]
        return [item for item in cleaned if item is not None]
    if isinstance(val, dict):
        return {str(k): sanitize_json_value(v) for k, v in val.items()}
    if isinstance(val, str):
        s = val.strip()
        if s.lower() in ("nan", "none") and s != "None":
            return None
        return s
    return val


def split_text_into_chunks(text_content: str, max_words: int = CHUNK_TARGET_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    """
    Deterministically splits text into chunks of approximately max_words.
    Preserves sentence boundaries as much as possible and includes overlap.
    Extremely short sections are not split.
    """
    words = text_content.split()
    if len(words) <= max_words:
        return [text_content]

    # Split into candidate sentences/segments
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_content) if s.strip()]

    # If any single sentence exceeds max_words, break it into word-level sub-segments
    units = []
    for s in raw_sentences:
        s_words = s.split()
        if len(s_words) > max_words:
            step = max_words - overlap_words if max_words > overlap_words else max_words
            for i in range(0, len(s_words), step):
                sub_part = " ".join(s_words[i:i + max_words])
                units.append(sub_part)
        else:
            units.append(s)

    chunks = []
    current_units = []
    current_word_count = 0

    for unit in units:
        unit_words = len(unit.split())
        if current_word_count + unit_words > max_words and current_units:
            chunks.append(" ".join(current_units))
            # Determine overlap units from the tail of current_units
            overlap_units = []
            overlap_count = 0
            for u in reversed(current_units):
                u_len = len(u.split())
                if overlap_count + u_len <= overlap_words:
                    overlap_units.insert(0, u)
                    overlap_count += u_len
                else:
                    break
            current_units = overlap_units + [unit]
            current_word_count = overlap_count + unit_words
        else:
            current_units.append(unit)
            current_word_count += unit_words

    if current_units:
        chunks.append(" ".join(current_units))

    return chunks


def build_chunks_for_scheme(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transforms a single scheme database row into a list of retrieval-ready chunk dictionaries.
    Each chunk contains:
      - scheme_id
      - chunk_id
      - field_name
      - chunk_text (with contextual prefix)
      - chunk_index
      - metadata (strict JSON string)
    """
    scheme_id = str(row["scheme_id"]).strip()
    scheme_name = str(row["scheme_name"]).strip()

    # Clean level
    raw_level = row.get("level")
    if pd.isna(raw_level) or str(raw_level).strip().lower() in ("nan", "none", ""):
        level = None
    else:
        level = str(raw_level).strip()

    # Clean state: missing must be None, never NaN or "nan"
    raw_state = row.get("state")
    if pd.isna(raw_state) or str(raw_state).strip().lower() in ("nan", "none", ""):
        state = None
    else:
        state = str(raw_state).strip()

    # Clean states list
    raw_states = row.get("states")
    if isinstance(raw_states, np.ndarray):
        raw_states = raw_states.tolist()
    elif not isinstance(raw_states, list):
        raw_states = []
    states = [
        str(s).strip()
        for s in raw_states
        if pd.notna(s) and str(s).strip() and str(s).strip().lower() not in ("nan", "none")
    ]

    # Clean categories list
    raw_categories = row.get("categories")
    if isinstance(raw_categories, np.ndarray):
        raw_categories = raw_categories.tolist()
    elif not isinstance(raw_categories, list):
        raw_categories = []
    categories = [
        str(c).strip()
        for c in raw_categories
        if pd.notna(c) and str(c).strip() and str(c).strip().lower() not in ("nan", "none")
    ]

    # Clean normalized_tags list
    raw_tags = row.get("normalized_tags")
    if isinstance(raw_tags, np.ndarray):
        raw_tags = raw_tags.tolist()
    elif not isinstance(raw_tags, list):
        raw_tags = []
    normalized_tags = [
        str(t).strip()
        for t in raw_tags
        if pd.notna(t) and str(t).strip() and str(t).strip().lower() not in ("nan", "none")
    ]

    def create_metadata_json(field_name: str, chunk_index: int, total_field_chunks: int, chunk_text: str) -> str:
        raw_metadata = {
            "scheme_id": scheme_id,
            "scheme_name": scheme_name,
            "level": level,
            "state": state,
            "states": states,
            "categories": categories,
            "normalized_tags": normalized_tags,
            "source_field": field_name,
            "chunk_index": chunk_index,
            "total_field_chunks": total_field_chunks,
            "word_count": len(chunk_text.split()),
            "char_count": len(chunk_text),
        }
        sanitized = sanitize_json_value(raw_metadata)
        # allow_nan=False guarantees strict JSON and raises error if any NaN remains
        return json.dumps(sanitized, ensure_ascii=False, allow_nan=False)

    chunks = []

    # -------------------------------------------------------------------------
    # 1. Scheme Identity & Overview Chunk (field_name = 'scheme_name')
    # -------------------------------------------------------------------------
    name_lines = [f"Scheme: {scheme_name}"]
    loc_parts = []
    if level:
        loc_parts.append(f"Level: {level}")
    if state:
        loc_parts.append(f"State: {state}")
    elif states:
        loc_parts.append(f"States: {', '.join(states[:5])}")
    if loc_parts:
        name_lines.append(" | ".join(loc_parts))
    if categories:
        name_lines.append(f"Categories: {', '.join(categories)}")
    if normalized_tags:
        name_lines.append(f"Tags: {', '.join(normalized_tags[:10])}")

    name_chunk_text = "\n".join(name_lines)
    name_chunk_id = f"{scheme_id}_scheme_name_0"

    chunks.append({
        "scheme_id": scheme_id,
        "chunk_id": name_chunk_id,
        "field_name": "scheme_name",
        "chunk_text": name_chunk_text,
        "chunk_index": 0,
        "metadata": create_metadata_json("scheme_name", 0, 1, name_chunk_text),
    })

    # -------------------------------------------------------------------------
    # 2. Section Chunks (details, benefits, eligibility, application, documents)
    # -------------------------------------------------------------------------
    section_fields = ["details", "benefits", "eligibility", "application", "documents"]

    for field in section_fields:
        val = row.get(field)
        if val is None:
            continue
        text_content = str(val).strip()
        if not text_content:
            continue

        field_splits = split_text_into_chunks(text_content, max_words=CHUNK_TARGET_WORDS, overlap_words=CHUNK_OVERLAP_WORDS)
        total_fc = len(field_splits)

        for c_idx, raw_chunk_text in enumerate(field_splits):
            part_suffix = f" (Part {c_idx + 1} of {total_fc})" if total_fc > 1 else ""
            section_title = field.title()
            header = f"Scheme: {scheme_name}\nSection: {section_title}{part_suffix}\n\n"
            full_chunk_text = header + raw_chunk_text
            chunk_id = f"{scheme_id}_{field}_{c_idx}"

            chunks.append({
                "scheme_id": scheme_id,
                "chunk_id": chunk_id,
                "field_name": field,
                "chunk_text": full_chunk_text,
                "chunk_index": c_idx,
                "metadata": create_metadata_json(field, c_idx, total_fc, full_chunk_text),
            })

    return chunks


def process_and_ingest_chunks(batch_size: int = 1000, dry_run: bool = False, rechunk: bool = False):
    print("=" * 76)
    print("  sugamgov-rag | Step 5.1: RAG Document Chunking Pipeline")
    print("=" * 76)

    # -------------------------------------------------------------------------
    # 1. Connect to PostgreSQL
    # -------------------------------------------------------------------------
    print("\n[1/6] Connecting to PostgreSQL...")
    if not ENV_PATH.exists():
        print(f"[FAIL] .env file not found at: {ENV_PATH.relative_to(PROJECT_ROOT).as_posix()}")
        sys.exit(1)

    load_dotenv(dotenv_path=ENV_PATH)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("[FAIL] DATABASE_URL not found in .env")
        sys.exit(1)

    masked_url = mask_url_password(database_url)
    print(f"      Database URL: {masked_url}")

    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("      ✓ PostgreSQL connection established.")
    except Exception as e:
        print(f"[FAIL] Database connection error: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. Read Schemes from PostgreSQL
    # -------------------------------------------------------------------------
    print("\n[2/6] Reading Schemes from PostgreSQL...")
    query = """
        SELECT 
            scheme_id,
            scheme_name,
            details,
            benefits,
            eligibility,
            application,
            documents,
            level,
            state,
            states,
            categories,
            normalized_tags
        FROM schemes
        ORDER BY scheme_id ASC;
    """
    df_schemes = pd.read_sql(query, engine)
    total_schemes = len(df_schemes)
    print(f"      ✓ Retrieved {total_schemes:,} schemes from 'schemes' table.")

    if total_schemes == 0:
        print("[FAIL] 'schemes' table is empty. Ingest schemes before chunking.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Generate Chunks
    # -------------------------------------------------------------------------
    print("\n[3/6] Generating Deterministic Text Chunks...")
    all_chunks = []
    schemes_with_zero_chunks = []
    chunks_by_field: Dict[str, int] = {f: 0 for f in TEXT_FIELDS}
    chunk_lengths_words = []
    chunk_lengths_chars = []
    seen_chunk_ids = set()
    duplicate_chunk_ids = []

    for row in df_schemes.to_dict(orient="records"):
        scheme_chunks = build_chunks_for_scheme(row)
        if not scheme_chunks:
            schemes_with_zero_chunks.append(row["scheme_id"])
            continue

        for chk in scheme_chunks:
            cid = chk["chunk_id"]
            if cid in seen_chunk_ids:
                duplicate_chunk_ids.append(cid)
            seen_chunk_ids.add(cid)

            chunks_by_field[chk["field_name"]] = chunks_by_field.get(chk["field_name"], 0) + 1
            w_len = len(chk["chunk_text"].split())
            c_len = len(chk["chunk_text"])
            chunk_lengths_words.append(w_len)
            chunk_lengths_chars.append(c_len)

        all_chunks.extend(scheme_chunks)

    total_chunks = len(all_chunks)
    print(f"      ✓ Total chunks generated: {total_chunks:,}")

    # -------------------------------------------------------------------------
    # 4. Report Pre-Insertion Chunk Statistics & Strict JSON Validation
    # -------------------------------------------------------------------------
    print("\n[4/6] Pre-Insertion Chunk Validation & Statistics:")
    print(f"      - Schemes processed:              {total_schemes:,}")
    print(f"      - Schemes with no chunks:          {len(schemes_with_zero_chunks)}")
    print(f"      - Total chunks generated:         {total_chunks:,}")
    print(f"      - Duplicate chunk IDs:            {len(duplicate_chunk_ids)}")

    empty_chunks = [ch for ch in all_chunks if not ch["chunk_text"].strip()]
    print(f"      - Empty chunk text entries:       {len(empty_chunks)}")

    # Strict metadata JSON compliance validation
    metadata_json_errors = 0
    nan_infinity_pattern = re.compile(r'\b(NaN|Infinity|-Infinity)\b')
    for chk in all_chunks:
        meta_str = chk["metadata"]
        # Check 1: No unquoted NaN / Infinity tokens
        if nan_infinity_pattern.search(meta_str):
            metadata_json_errors += 1
            if metadata_json_errors <= 3:
                print(f"[FAIL] Found invalid token (NaN/Infinity) in chunk {chk['chunk_id']}: {meta_str[:120]}")
            continue
        # Check 2: Parse strictly with json.loads
        try:
            parsed = json.loads(meta_str)
            # Check 3: 'state' must be string or None, never "nan"
            if parsed.get("state") in ("nan", "NaN", "NAN"):
                metadata_json_errors += 1
                if metadata_json_errors <= 3:
                    print(f"[FAIL] 'state' is string 'nan' in chunk {chk['chunk_id']}")
        except Exception as e:
            metadata_json_errors += 1
            if metadata_json_errors <= 3:
                print(f"[FAIL] JSON parsing failed for chunk {chk['chunk_id']}: {e}")

    meta_status = "PASSED (0 invalid tokens, strict JSON compliant)" if metadata_json_errors == 0 else f"FAILED ({metadata_json_errors} errors)"
    print(f"      - Metadata JSON validation:       {meta_status}")

    # Foreign key reference check against schemes
    valid_scheme_ids = set(df_schemes["scheme_id"])
    missing_scheme_refs = [ch["scheme_id"] for ch in all_chunks if ch["scheme_id"] not in valid_scheme_ids]
    print(f"      - Missing scheme references:      {len(missing_scheme_refs)}")

    print("\n      Chunks by field:")
    for f in TEXT_FIELDS:
        count = chunks_by_field.get(f, 0)
        pct = (count / total_chunks * 100) if total_chunks > 0 else 0
        print(f"        • {f:15s}: {count:6,d} ({pct:5.1f}%)")

    min_words = min(chunk_lengths_words) if chunk_lengths_words else 0
    max_words = max(chunk_lengths_words) if chunk_lengths_words else 0
    avg_words = (sum(chunk_lengths_words) / len(chunk_lengths_words)) if chunk_lengths_words else 0
    min_chars = min(chunk_lengths_chars) if chunk_lengths_chars else 0
    max_chars = max(chunk_lengths_chars) if chunk_lengths_chars else 0
    avg_chars = (sum(chunk_lengths_chars) / len(chunk_lengths_chars)) if chunk_lengths_chars else 0

    print("\n      Chunk length statistics:")
    print(f"        • Word count:  min={min_words}, avg={avg_words:.1f}, max={max_words}")
    print(f"        • Char count:  min={min_chars}, avg={avg_chars:.1f}, max={max_chars}")

    # Validation checks
    valid = True
    if len(duplicate_chunk_ids) > 0:
        print(f"[FAIL] Found {len(duplicate_chunk_ids)} duplicate chunk IDs!")
        valid = False
    if len(empty_chunks) > 0:
        print(f"[FAIL] Found {len(empty_chunks)} empty chunks!")
        valid = False
    if len(schemes_with_zero_chunks) > 0:
        print(f"[FAIL] Found {len(schemes_with_zero_chunks)} schemes with 0 chunks!")
        valid = False
    if metadata_json_errors > 0:
        print(f"[FAIL] Found {metadata_json_errors} metadata JSON compliance errors!")
        valid = False
    if len(missing_scheme_refs) > 0:
        print(f"[FAIL] Found {len(missing_scheme_refs)} missing scheme references!")
        valid = False

    if not valid:
        print("\n[ERROR] Chunk validation failed. Aborting.")
        sys.exit(1)

    print("      ✓ All pre-insertion validation checks passed.")

    # -------------------------------------------------------------------------
    # 5. Dry-Run Mode vs Database Ingestion
    # -------------------------------------------------------------------------
    if dry_run:
        print("\n[5/6] DRY RUN MODE: Database insertion skipped as requested by --dry-run.")

        with engine.connect() as conn:
            existing_chunks_count = conn.execute(text("SELECT count(id) FROM scheme_chunks;")).scalar()

        print("\n" + "=" * 76)
        print("STEP 5.1 CHUNKING REPORT")
        print("=" * 76)
        print(f"Schemes processed:          {total_schemes:,}")
        print(f"Total chunks:               {total_chunks:,}")
        print(f"Chunks by field:            " + ", ".join(f"{k}: {v:,}" for k, v in chunks_by_field.items()))
        print(f"Chunk length statistics:    words [min={min_words}, avg={avg_words:.1f}, max={max_words}] | chars [min={min_chars}, avg={avg_chars:.1f}, max={max_chars}]")
        print(f"Duplicate chunk IDs:        {len(duplicate_chunk_ids)}")
        print(f"Empty chunks:               {len(empty_chunks)}")
        print(f"Missing scheme references:  {len(missing_scheme_refs)}")
        print(f"Metadata JSON validation:   PASSED")
        print(f"Embeddings generated:       0")
        print(f"scheme_chunks rows:         {existing_chunks_count} (dry-run, no database modification)")
        print(f"Overall result:             PASS (DRY RUN)")
        print("=" * 76)
        return

    # Check existing count in scheme_chunks
    with engine.connect() as conn:
        existing_chunks_count = conn.execute(text("SELECT count(id) FROM scheme_chunks;")).scalar()

    if existing_chunks_count > 0 and not rechunk:
        print(f"\n[*] 'scheme_chunks' already contains {existing_chunks_count:,} rows.")
        print("    Running in safe mode: existing chunks will be preserved (ON CONFLICT DO NOTHING).")
    elif existing_chunks_count > 0 and rechunk:
        print(f"\n[*] Clearing {existing_chunks_count:,} existing rows in 'scheme_chunks' (--rechunk requested)...")
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM scheme_chunks;"))
        print("    ✓ Table 'scheme_chunks' cleared.")

    # -------------------------------------------------------------------------
    # Insertion Phase
    # -------------------------------------------------------------------------
    print(f"\n[5/6] Ingesting Chunks into 'scheme_chunks' (batch size: {batch_size})...")

    insert_sql = text("""
        INSERT INTO scheme_chunks (
            scheme_id,
            chunk_id,
            field_name,
            chunk_text,
            chunk_index,
            embedding,
            metadata
        ) VALUES (
            :scheme_id,
            :chunk_id,
            :field_name,
            :chunk_text,
            :chunk_index,
            NULL,
            :metadata
        )
        ON CONFLICT (chunk_id) DO NOTHING;
    """)

    num_batches = (total_chunks + batch_size - 1) // batch_size

    try:
        with engine.begin() as trans_conn:
            for b_idx in range(num_batches):
                start_i = b_idx * batch_size
                end_i = min((b_idx + 1) * batch_size, total_chunks)
                batch_data = all_chunks[start_i:end_i]

                trans_conn.execute(insert_sql, batch_data)
                print(f"      Batch {b_idx + 1}/{num_batches}: Processed chunks {start_i + 1:,} to {end_i:,} [OK]")

        print("      ✓ Database transaction committed successfully.")

    except Exception as e:
        print(f"\n[ERROR] Insertion failed during batch execution: {e}")
        print("Transaction rolled back cleanly.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 6. Post-Insertion Verification
    # -------------------------------------------------------------------------
    print("\n[6/6] Verifying Database State Post-Insertion...")
    with engine.connect() as conn:
        final_chunks_count = conn.execute(text("SELECT count(id) FROM scheme_chunks;")).scalar()
        unique_chunk_ids_count = conn.execute(text("SELECT count(DISTINCT chunk_id) FROM scheme_chunks;")).scalar()
        non_null_embeddings = conn.execute(text("SELECT count(id) FROM scheme_chunks WHERE embedding IS NOT NULL;")).scalar()
        empty_chunk_texts = conn.execute(text("SELECT count(id) FROM scheme_chunks WHERE TRIM(chunk_text) = '';")).scalar()

        # Check foreign key integrity: orphaned chunks
        orphaned_chunks = conn.execute(text("""
            SELECT count(c.id) 
            FROM scheme_chunks c 
            LEFT JOIN schemes s ON c.scheme_id = s.scheme_id 
            WHERE s.scheme_id IS NULL;
        """)).scalar()

        # Check untouched tables
        schemes_c = conn.execute(text("SELECT count(id) FROM schemes;")).scalar()
        sources_c = conn.execute(text("SELECT count(id) FROM sources;")).scalar()
        versions_c = conn.execute(text("SELECT count(id) FROM scheme_versions;")).scalar()

    print(f"      - Total scheme_chunks:       {final_chunks_count:,}")
    print(f"      - Unique chunk IDs:          {unique_chunk_ids_count:,}")
    print(f"      - Non-NULL embeddings:       {non_null_embeddings} (expected: 0)")
    print(f"      - Empty chunk_text rows:     {empty_chunk_texts} (expected: 0)")
    print(f"      - Missing scheme references: {orphaned_chunks} (expected: 0)")
    print(f"      - schemes table count:       {schemes_c:,} (untouched)")
    print(f"      - sources table count:       {sources_c} (untouched)")
    print(f"      - scheme_versions count:     {versions_c} (untouched)")

    post_valid = True
    if non_null_embeddings > 0:
        print(f"[FAIL] Embeddings must be NULL in Step 5.1! Found {non_null_embeddings}")
        post_valid = False
    if empty_chunk_texts > 0:
        print(f"[FAIL] Found {empty_chunk_texts} empty chunk_text rows!")
        post_valid = False
    if orphaned_chunks > 0:
        print(f"[FAIL] Found {orphaned_chunks} orphaned chunks not referencing schemes!")
        post_valid = False
    if final_chunks_count != unique_chunk_ids_count:
        print(f"[FAIL] final_chunks_count ({final_chunks_count}) != unique_chunk_ids_count ({unique_chunk_ids_count})")
        post_valid = False

    overall_result = "PASS" if post_valid else "FAIL"

    # Final Formatted Report
    print("\n" + "=" * 76)
    print("STEP 5.1 CHUNKING REPORT")
    print("=" * 76)
    print(f"Schemes processed:          {total_schemes:,}")
    print(f"Total chunks:               {final_chunks_count:,}")
    print(f"Chunks by field:            " + ", ".join(f"{k}: {v:,}" for k, v in chunks_by_field.items()))
    print(f"Chunk length statistics:    words [min={min_words}, avg={avg_words:.1f}, max={max_words}] | chars [min={min_chars}, avg={avg_chars:.1f}, max={max_chars}]")
    print(f"Duplicate chunk IDs:        0")
    print(f"Empty chunks:               0")
    print(f"Missing scheme references:  0")
    print(f"Metadata JSON validation:   PASSED")
    print(f"Embeddings generated:       0")
    print(f"scheme_chunks rows:         {final_chunks_count:,}")
    print(f"Overall result:             {overall_result}")
    print("=" * 76)

    if not post_valid:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Step 5.1: RAG Document Chunking Pipeline for SugamGov"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and validate chunks without inserting into PostgreSQL"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for PostgreSQL insertion (default: 1000)"
    )
    parser.add_argument(
        "--rechunk",
        action="store_true",
        help="Clear existing scheme_chunks rows before rechunking"
    )

    args = parser.parse_args()

    process_and_ingest_chunks(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        rechunk=args.rechunk
    )


if __name__ == "__main__":
    main()
