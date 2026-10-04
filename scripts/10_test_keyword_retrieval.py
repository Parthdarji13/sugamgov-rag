"""
scripts/10_test_keyword_retrieval.py
====================================
Step 5.3.1: Verification & Test Suite for PostgreSQL Keyword Retrieval Foundation.

This test script:
  1. Verifies database connectivity and captures baseline integrity counts.
  2. Validates keyword full-text search against the 4 required benchmark queries:
     - 'student scholarship'
     - 'farmer financial assistance'
     - 'women pension'
     - 'housing scheme Gujarat'
  3. Validates metadata pre-filtering:
     - state='Gujarat'
     - level='Central'
     - category='Education & Learning'
  4. Runs automated structural & integrity checks:
     - Structured RetrievalResult objects returned
     - No NULL or empty chunk_id
     - Valid scheme references (every returned scheme_id exists in schemes)
     - Configurable top_k is strictly respected
     - Filters are strictly applied with 0% contamination
     - Deterministic descending relevance score order
  5. Strictly READ-ONLY: performs zero INSERT/UPDATE/DELETE operations.
  6. Verifies database integrity counts remain strictly unchanged:
     - schemes = 3,397
     - scheme_chunks = 20,497
     - embedded vectors = 1,260

Context & Architectural Note:
-----------------------------
- What keyword retrieval does:
  Performs weighted lexical full-text search (scheme_name=1.0, chunk_text=0.4) using
  PostgreSQL's native 'websearch_to_tsquery' and 'ts_rank_cd' (Cover Density Ranking).
- Why built before semantic retrieval:
  Dense vector embeddings currently exist for 1,260 chunks, while 19,237 chunks are pending
  Gemini free-tier quota reset. Keyword search provides immediate 100% search coverage
  over all 20,497 chunks with zero external API dependencies.
- How it will later be combined with vector retrieval:
  Once all 20,497 chunk embeddings are generated, keyword search and pgvector cosine similarity
  will be merged using Reciprocal Rank Fusion (RRF):
      RRF_Score = 1 / (60 + rank_keyword) + 1 / (60 + rank_vector)
"""

import os
import sys
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Add project root to sys.path to enable src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import KeywordRetriever, RetrievalResult

# Reconfigure stdout for UTF-8 on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_db_counts(engine) -> Tuple[int, int, int]:
    """Returns (schemes_count, chunks_count, embedded_count)."""
    with engine.connect() as conn:
        s_count = conn.execute(text("SELECT count(id) FROM schemes;")).scalar()
        c_count = conn.execute(text("SELECT count(id) FROM scheme_chunks;")).scalar()
        e_count = conn.execute(text("SELECT count(id) FROM scheme_chunks WHERE embedding IS NOT NULL;")).scalar()
    return s_count, c_count, e_count


def print_result_card(idx: int, res: RetrievalResult):
    """Prints a formatted card for a single retrieval result."""
    state_str = res.state or "All-India / Central"
    level_str = res.level or "N/A"
    cats_str = ", ".join(res.categories) if res.categories else "None"
    if len(cats_str) > 45:
        cats_str = cats_str[:42] + "..."

    print(f"    [{idx}] Scheme: [{res.scheme_id}] {res.scheme_name}")
    print(f"        Chunk ID:  {res.chunk_id}")
    print(f"        Field:     {res.field_name} | Score: {res.score:.4f}")
    print(f"        State:     {state_str} | Level: {level_str}")
    print(f"        Category:  {cats_str}")
    print(f"        Preview:   {res.preview(120)}")
    print("")


def run_tests():
    print("=" * 80)
    print("  sugamgov-rag | Step 5.3.1: PostgreSQL Keyword Retrieval Test Suite")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Environment & Database Connectivity
    # -------------------------------------------------------------------------
    print("\n[1/5] Verifying Database Connectivity & Baseline Counts...")
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print(f"[FAIL] .env file not found at {env_file}")
        sys.exit(1)

    load_dotenv(env_file)
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("[FAIL] DATABASE_URL is not set in .env")
        sys.exit(1)

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        print("      ✓ Database connection established successfully.")
    except Exception as e:
        print(f"[FAIL] Database connection failed: {e}")
        sys.exit(1)

    init_schemes, init_chunks, init_embedded = get_db_counts(engine)
    print(f"      - Baseline schemes:       {init_schemes:,} (expected: 3,397)")
    print(f"      - Baseline scheme_chunks: {init_chunks:,} (expected: 20,497)")
    print(f"      - Baseline embedded:      {init_embedded:,} (expected: 1,260)")

    if init_schemes != 3397 or init_chunks != 20497 or init_embedded != 1260:
        print("[FAIL] Baseline database counts do not match expected Step 5.3 state!")
        sys.exit(1)

    retriever = KeywordRetriever(engine)
    total_checks = 0
    passed_checks = 0

    # -------------------------------------------------------------------------
    # 2. Benchmark Queries (Unfiltered)
    # -------------------------------------------------------------------------
    print("\n[2/5] Testing Benchmark Keyword Queries (top_k=5)...")
    benchmark_queries = [
        "student scholarship",
        "farmer financial assistance",
        "women pension",
        "housing scheme Gujarat",
    ]

    for q in benchmark_queries:
        print(f"\n   >>> Query: \"{q}\"")
        results = retriever.retrieve(query=q, top_k=5)
        print(f"       Total Results Returned: {len(results)}")

        total_checks += 1
        if len(results) > 0:
            passed_checks += 1
            print("       ✓ Retrieved non-empty result set.")
        else:
            print("       ✗ No results returned!")

        for i, res in enumerate(results, 1):
            print_result_card(i, res)

    # -------------------------------------------------------------------------
    # 3. Filtered Retrieval Tests (Metadata Pre-filtering)
    # -------------------------------------------------------------------------
    print("\n[3/5] Testing Metadata Pre-Filtered Queries...")

    # Filter Test A: State = 'Gujarat'
    print("\n   --- Filter Test A: State='Gujarat' ---")
    q_guj = "housing scheme"
    guj_results = retriever.retrieve(query=q_guj, state="Gujarat", top_k=5)
    print(f"       Query: \"{q_guj}\" | Filter: state=\"Gujarat\" | Results: {len(guj_results)}")

    total_checks += 1
    guj_valid = True
    if guj_results:
        for r in guj_results:
            is_guj = (r.state and "gujarat" in r.state.lower()) or any(
                "gujarat" in str(s).lower() for s in r.metadata.get("states", [])
            )
            if not is_guj:
                guj_valid = False
                print(f"       ✗ Non-Gujarat result leaked: {r.scheme_id} (state={r.state})")
        if guj_valid:
            passed_checks += 1
            print("       ✓ 100% of returned results belong to Gujarat.")
        for i, res in enumerate(guj_results, 1):
            print_result_card(i, res)
    else:
        print("       ✗ No results returned for Gujarat housing query!")

    # Filter Test B: Level = 'Central'
    print("\n   --- Filter Test B: Level='Central' ---")
    q_central = "farmer financial assistance"
    central_results = retriever.retrieve(query=q_central, level="Central", top_k=5)
    print(f"       Query: \"{q_central}\" | Filter: level=\"Central\" | Results: {len(central_results)}")

    total_checks += 1
    central_valid = True
    if central_results:
        for r in central_results:
            if not r.level or r.level.lower() != "central":
                central_valid = False
                print(f"       ✗ Non-Central result leaked: {r.scheme_id} (level={r.level})")
        if central_valid:
            passed_checks += 1
            print("       ✓ 100% of returned results have level='Central'.")
        for i, res in enumerate(central_results, 1):
            print_result_card(i, res)
    else:
        print("       ✗ No results returned for Central farmer assistance query!")

    # Filter Test C: Category = 'Education & Learning'
    print("\n   --- Filter Test C: Category='Education & Learning' ---")
    q_edu = "student scholarship"
    edu_results = retriever.retrieve(query=q_edu, category="Education & Learning", top_k=5)
    print(f"       Query: \"{q_edu}\" | Filter: category=\"Education & Learning\" | Results: {len(edu_results)}")

    total_checks += 1
    edu_valid = True
    if edu_results:
        for r in edu_results:
            has_edu = any("education & learning" in c.lower() for c in r.categories)
            if not has_edu:
                edu_valid = False
                print(f"       ✗ Result missing Education & Learning: {r.scheme_id} (categories={r.categories})")
        if edu_valid:
            passed_checks += 1
            print("       ✓ 100% of returned results belong to 'Education & Learning'.")
        for i, res in enumerate(edu_results, 1):
            print_result_card(i, res)
    else:
        print("       ✗ No results returned for Education & Learning scholarship query!")

    # -------------------------------------------------------------------------
    # 4. Automated Structural & Quality Validations
    # -------------------------------------------------------------------------
    print("\n[4/5] Executing Automated Structural & Quality Validations...")

    # Check 1: Return type is RetrievalResult
    sample_set = retriever.retrieve("scholarship", top_k=3)
    total_checks += 1
    if sample_set and all(isinstance(r, RetrievalResult) for r in sample_set):
        passed_checks += 1
        print("      ✓ Validation 1: Results are typed RetrievalResult objects.")
    else:
        print("      ✗ Validation 1 failed: Result typing error.")

    # Check 2: No NULL chunk_id or empty chunk_id
    total_checks += 1
    has_null_chunks = any(not r.chunk_id or r.chunk_id.strip() == "" for r in sample_set)
    if not has_null_chunks:
        passed_checks += 1
        print("      ✓ Validation 2: No NULL or empty chunk_id values detected.")
    else:
        print("      ✗ Validation 2 failed: NULL chunk_id detected.")

    # Check 3: Valid scheme references
    total_checks += 1
    with engine.connect() as conn:
        all_scheme_ids = [r.scheme_id for r in sample_set]
        found_count = conn.execute(
            text("SELECT count(*) FROM schemes WHERE scheme_id = ANY(:ids);"),
            {"ids": all_scheme_ids}
        ).scalar()
    if found_count == len(all_scheme_ids):
        passed_checks += 1
        print(f"      ✓ Validation 3: All {found_count} returned scheme_id references exist in 'schemes'.")
    else:
        print(f"      ✗ Validation 3 failed: Missing scheme reference in database.")

    # Check 4: Configurable top_k is strictly respected
    total_checks += 1
    top_3 = retriever.retrieve("scholarship", top_k=3)
    top_7 = retriever.retrieve("scholarship", top_k=7)
    if len(top_3) == 3 and len(top_7) == 7:
        passed_checks += 1
        print(f"      ✓ Validation 4: top_k parameter strictly respected (top_3={len(top_3)}, top_7={len(top_7)}).")
    else:
        print(f"      ✗ Validation 4 failed: top_k mismatch (got {len(top_3)} for k=3, {len(top_7)} for k=7).")

    # Check 5: Deterministic descending relevance score order
    total_checks += 1
    scores = [r.score for r in top_7]
    is_sorted_desc = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    all_positive = all(s > 0 for s in scores)
    if is_sorted_desc and all_positive:
        passed_checks += 1
        print(f"      ✓ Validation 5: Scores are positive and strictly sorted descending ({scores[0]:.4f} -> {scores[-1]:.4f}).")
    else:
        print(f"      ✗ Validation 5 failed: Sorting or score validity issue (scores={scores}).")

    # Check 6: Empty query handling
    total_checks += 1
    empty_res = retriever.retrieve("   ")
    if empty_res == []:
        passed_checks += 1
        print("      ✓ Validation 6: Empty query safely returns empty list.")
    else:
        print("      ✗ Validation 6 failed: Empty query did not return empty list.")

    # -------------------------------------------------------------------------
    # 5. Database Integrity Verification (Read-Only Guarantee)
    # -------------------------------------------------------------------------
    print("\n[5/5] Verifying Post-Execution Database Integrity (Read-Only Check)...")
    final_schemes, final_chunks, final_embedded = get_db_counts(engine)

    schemes_match = (final_schemes == init_schemes == 3397)
    chunks_match = (final_chunks == init_chunks == 20497)
    embedded_match = (final_embedded == init_embedded == 1260)

    print(f"      - Final schemes:          {final_schemes:,} (diff: {final_schemes - init_schemes:+d})")
    print(f"      - Final scheme_chunks:    {final_chunks:,} (diff: {final_chunks - init_chunks:+d})")
    print(f"      - Final embedded vectors: {final_embedded:,} (diff: {final_embedded - init_embedded:+d})")

    total_checks += 1
    if schemes_match and chunks_match and embedded_match:
        passed_checks += 1
        print("      ✓ Read-only guarantee confirmed: 100% database parity preserved.")
    else:
        print("      ✗ Database integrity failure: Record counts modified!")

    # -------------------------------------------------------------------------
    # Final Report Summary
    # -------------------------------------------------------------------------
    overall_pass = (passed_checks == total_checks)
    print("\n" + "=" * 80)
    print("STEP 5.3.1 KEYWORD RETRIEVAL VERIFICATION REPORT")
    print("=" * 80)
    print(f"Total Validation Checks:     {total_checks}")
    print(f"Passed Checks:              {passed_checks}")
    print(f"Failed Checks:              {total_checks - passed_checks}")
    print(f"Schemes Table Count:        {final_schemes:,}")
    print(f"Scheme Chunks Count:        {final_chunks:,}")
    print(f"Populated Vector Count:     {final_embedded:,}")
    print(f"Database Modified:          NO (READ-ONLY)")
    print(f"Overall Result:             {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 80)

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
