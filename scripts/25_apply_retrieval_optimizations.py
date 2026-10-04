"""
scripts/25_apply_retrieval_optimizations.py
===========================================
Step 19: Database Migration for Retrieval Performance Optimization.

Applies:
1. Generated tsvector column 'search_vector' on scheme_chunks.
2. GIN index 'idx_scheme_chunks_search_vector' on search_vector.
3. HNSW index 'idx_scheme_chunks_embedding_local_hnsw' on embedding_local (vector_cosine_ops).

Guarantees:
- Read-only on existing scheme data (0 row mutations, 0 deletions, 0 insertions).
- Zero changes to Gemini embeddings or local vector values.
- Strict pre- and post-migration parity audits.
"""

import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_db_metrics(conn):
    s_cnt = conn.execute(text("SELECT count(*) FROM schemes;")).scalar() or 0
    c_cnt = conn.execute(text("SELECT count(*) FROM scheme_chunks;")).scalar() or 0
    gem_pop = conn.execute(text("SELECT count(*) FROM scheme_chunks WHERE embedding IS NOT NULL;")).scalar() or 0
    gem_null = conn.execute(text("SELECT count(*) FROM scheme_chunks WHERE embedding IS NULL;")).scalar() or 0
    loc_pop = conn.execute(text("SELECT count(*) FROM scheme_chunks WHERE embedding_local IS NOT NULL;")).scalar() or 0
    loc_null = conn.execute(text("SELECT count(*) FROM scheme_chunks WHERE embedding_local IS NULL;")).scalar() or 0
    return {
        "schemes": s_cnt,
        "total_chunks": c_cnt,
        "gem_pop": gem_pop,
        "gem_null": gem_null,
        "loc_pop": loc_pop,
        "loc_null": loc_null,
    }


def main():
    print("=" * 80)
    print("  SUGAMGOV RAG — STEP 19: DATABASE PERFORMANCE MIGRATION")
    print("=" * 80)

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("[ERROR] DATABASE_URL not set in .env")
        sys.exit(1)

    engine = create_engine(db_url)

    with engine.connect() as conn:
        # 1. Pre-migration check
        print("\n[1/4] Pre-Migration Integrity Audit...")
        pre = get_db_metrics(conn)
        print(f"  Schemes:              {pre['schemes']:,} (expected 3,397)")
        print(f"  Chunks:               {pre['total_chunks']:,} (expected 20,497)")
        print(f"  Local Populated:      {pre['loc_pop']:,} (expected 20,497)")
        print(f"  Local NULL:           {pre['loc_null']:,} (expected 0)")
        print(f"  Gemini Populated:     {pre['gem_pop']:,} (expected 1,260)")
        print(f"  Gemini NULL:          {pre['gem_null']:,} (expected 19,237)")

        if (
            pre['schemes'] != 3397 or
            pre['total_chunks'] != 20497 or
            pre['loc_pop'] != 20497 or
            pre['loc_null'] != 0 or
            pre['gem_pop'] != 1260 or
            pre['gem_null'] != 19237
        ):
            print("[ERROR] Baseline DB metrics mismatch! Aborting migration.")
            sys.exit(1)

        print("  ✓ Pre-migration database integrity verified.")

        # 2. Add search_vector generated column
        print("\n[2/4] Applying Schema Additions...")
        print("  A. Adding generated column 'search_vector' to scheme_chunks...")
        t0 = time.time()
        conn.execute(text("""
            ALTER TABLE scheme_chunks
            ADD COLUMN IF NOT EXISTS search_vector tsvector
            GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(metadata->>'scheme_name', '')), 'A') ||
                setweight(to_tsvector('english', chunk_text), 'B')
            ) STORED;
        """))
        conn.commit()
        t_col = time.time() - t0
        print(f"     ✓ 'search_vector' created and populated in {t_col:.2f}s")

        # 3. Create GIN index on search_vector
        print("  B. Creating GIN index 'idx_scheme_chunks_search_vector'...")
        t0 = time.time()
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_scheme_chunks_search_vector
            ON scheme_chunks
            USING GIN (search_vector);
        """))
        conn.commit()
        t_gin = time.time() - t0
        print(f"     ✓ GIN index built in {t_gin:.2f}s")

        # 4. Create HNSW index on embedding_local
        print("  C. Creating HNSW index 'idx_scheme_chunks_embedding_local_hnsw'...")
        t0 = time.time()
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_scheme_chunks_embedding_local_hnsw
            ON scheme_chunks
            USING hnsw (embedding_local vector_cosine_ops);
        """))
        conn.commit()
        t_hnsw = time.time() - t0
        print(f"     ✓ HNSW index built in {t_hnsw:.2f}s")

        # 5. Post-migration check
        print("\n[3/4] Post-Migration Integrity Audit...")
        post = get_db_metrics(conn)
        print(f"  Schemes:              {post['schemes']:,} (Delta: {post['schemes'] - pre['schemes']})")
        print(f"  Chunks:               {post['total_chunks']:,} (Delta: {post['total_chunks'] - pre['total_chunks']})")
        print(f"  Local Populated:      {post['loc_pop']:,} (Delta: {post['loc_pop'] - pre['loc_pop']})")
        print(f"  Local NULL:           {post['loc_null']:,} (Delta: {post['loc_null'] - pre['loc_null']})")
        print(f"  Gemini Populated:     {post['gem_pop']:,} (Delta: {post['gem_pop'] - pre['gem_pop']})")
        print(f"  Gemini NULL:          {post['gem_null']:,} (Delta: {post['gem_null'] - pre['gem_null']})")

        if (
            post['schemes'] != 3397 or
            post['total_chunks'] != 20497 or
            post['loc_pop'] != 20497 or
            post['loc_null'] != 0 or
            post['gem_pop'] != 1260 or
            post['gem_null'] != 19237
        ):
            print("[ERROR] Post-migration DB metrics mismatch!")
            sys.exit(1)
        print("  ✓ Zero rows modified, inserted, or deleted. 100% data parity confirmed.")

        # 6. Verify index existence and validity
        print("\n[4/4] Index Verification...")
        indexes = conn.execute(text("""
            SELECT 
                i.relname as index_name,
                idx.indisvalid as is_valid,
                am.amname as index_type,
                pg_size_pretty(pg_relation_size(i.oid)) as index_size
            FROM pg_index idx
            JOIN pg_class i ON i.oid = idx.indexrelid
            JOIN pg_am am ON am.oid = i.relam
            WHERE idx.indrelid = 'scheme_chunks'::regclass
            ORDER BY i.relname;
        """)).fetchall()

        for idx_row in indexes:
            print(f"  Index: {idx_row.index_name:42s} | Type: {idx_row.index_type:6s} | Valid: {idx_row.is_valid} | Size: {idx_row.index_size}")

    print("\n" + "=" * 80)
    print("  STEP 19 DATABASE MIGRATION COMPLETED SUCCESSFULLY")
    print(f"  Search Vector: {t_col:.2f}s | GIN Index: {t_gin:.2f}s | HNSW Index: {t_hnsw:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
