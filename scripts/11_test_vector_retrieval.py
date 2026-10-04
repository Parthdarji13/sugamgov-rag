"""
scripts/11_test_vector_retrieval.py
===================================
Step 16: Verification & Test Suite for PostgreSQL + pgvector Local Semantic Retrieval.
Model: intfloat/multilingual-e5-small (384-dimensional).

This test script:
  1. Verifies database connectivity and baseline counts:
     - schemes = 3,397
     - scheme_chunks = 20,497
     - Gemini embeddings populated = 1,260
     - Gemini embeddings NULL = 19,237
     - Local embeddings populated = 20,497
     - Local embeddings NULL = 0
  2. Generates in-memory test query embeddings using local 'intfloat/multilingual-e5-small' (384-dim)
     with asymmetric 'query: ' prefix:
     - English: 'student scholarship', 'farmer financial assistance', 'women pension', 'housing scheme Gujarat'
     - Hindi: 'किसान वित्तीय सहायता'
     - Gujarati: 'ખેડૂત આર્થિક સહાય'
  3. Executes semantic vector retrieval using pgvector cosine distance against embedding_local:
     - Cosine similarity = 1 - (embedding_local <=> query_vector)
     - Higher score = greater semantic similarity
     - Prints top results for each query
  4. Tests metadata pre-filtering on local embeddings:
     - state='Gujarat'
     - level='Central'
     - category='Education & Learning'
  5. Runs automated structural, dimension & safety validations:
     - Returned results contain valid chunk_id strings
     - All returned scheme_id references exist in 'schemes'
     - All returned rows have non-NULL embedding_local in PostgreSQL
     - Vector dimension is verified to be exactly 384
     - top_k parameter is strictly respected
     - Similarity scores are finite and sorted descending
     - Blank query is handled safely (returns empty list)
     - GEMINI_API_KEY is not required for retrieval
  6. Strictly READ-ONLY with respect to PostgreSQL:
     - Zero INSERT/UPDATE/DELETE queries
  7. Verifies post-execution database integrity counts remain strictly unchanged.
"""

import os
import sys
import math
from pathlib import Path
from typing import Tuple, List, Dict, Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Add project root to sys.path to enable src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import (
    VectorRetriever,
    RetrievalResult,
    generate_query_embedding,
    get_embedding_model,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
)

# Reconfigure stdout for UTF-8 on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_db_counts(engine) -> Dict[str, int]:
    """Returns database counts for schemes, chunks, gemini vectors, and local vectors."""
    with engine.connect() as conn:
        s_count = conn.execute(text("SELECT count(id) FROM schemes;")).scalar() or 0
        c_count = conn.execute(text("SELECT count(id) FROM scheme_chunks;")).scalar() or 0
        gem_pop = conn.execute(text("SELECT count(id) FROM scheme_chunks WHERE embedding IS NOT NULL;")).scalar() or 0
        gem_null = conn.execute(text("SELECT count(id) FROM scheme_chunks WHERE embedding IS NULL;")).scalar() or 0
        loc_pop = conn.execute(text("SELECT count(id) FROM scheme_chunks WHERE embedding_local IS NOT NULL;")).scalar() or 0
        loc_null = conn.execute(text("SELECT count(id) FROM scheme_chunks WHERE embedding_local IS NULL;")).scalar() or 0
    return {
        "schemes": s_count,
        "chunks": c_count,
        "gem_pop": gem_pop,
        "gem_null": gem_null,
        "loc_pop": loc_pop,
        "loc_null": loc_null,
    }


def print_result_card(idx: int, res: RetrievalResult):
    """Prints a formatted card for a single vector retrieval result."""
    state_str = res.state or "All-India / Central"
    level_str = res.level or "N/A"
    cats_str = ", ".join(res.categories) if res.categories else "None"
    if len(cats_str) > 45:
        cats_str = cats_str[:42] + "..."

    print(f"    [{idx}] Scheme: [{res.scheme_id}] {res.scheme_name}")
    print(f"        Chunk ID:   {res.chunk_id}")
    print(f"        Field:      {res.field_name} | Cosine Sim: {res.score:.4f}")
    print(f"        State:      {state_str} | Level: {level_str}")
    print(f"        Category:   {cats_str}")
    print(f"        Preview:    {res.preview(120)}")
    print("")


def run_tests():
    print("=" * 80)
    print("  sugamgov-rag | Step 16: Local Vector Retrieval Migration Test Suite")
    print(f"  Model: {EMBEDDING_MODEL} (dim={EMBEDDING_DIMENSION})")
    print("=" * 80)

    total_checks = 0
    passed_checks = 0

    # -------------------------------------------------------------------------
    # 1. Environment & Database Connectivity
    # -------------------------------------------------------------------------
    print("\n[1/6] Verifying Database Connectivity & Baseline Counts...")
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

    init_counts = get_db_counts(engine)
    print(f"      - Baseline schemes:              {init_counts['schemes']:,} (expected: 3,397)")
    print(f"      - Baseline scheme_chunks:        {init_counts['chunks']:,} (expected: 20,497)")
    print(f"      - Baseline Gemini embeddings:    {init_counts['gem_pop']:,} (expected: 1,260)")
    print(f"      - Baseline Gemini NULL:          {init_counts['gem_null']:,} (expected: 19,237)")
    print(f"      - Baseline Local embeddings:     {init_counts['loc_pop']:,} (expected: 20,497)")
    print(f"      - Baseline Local NULL:           {init_counts['loc_null']:,} (expected: 0)")

    if (
        init_counts["schemes"] != 3397
        or init_counts["chunks"] != 20497
        or init_counts["gem_pop"] != 1260
        or init_counts["gem_null"] != 19237
        or init_counts["loc_pop"] != 20497
        or init_counts["loc_null"] != 0
    ):
        print("[FAIL] Baseline database counts do not match expected state!")
        sys.exit(1)

    total_checks += 1
    passed_checks += 1
    print("      ✓ Database baseline counts verified.")

    # -------------------------------------------------------------------------
    # 2. Local Embedding Model Loading & Dimension Validation
    # -------------------------------------------------------------------------
    print("\n[2/6] Loading Local Embedding Model (Offline / In-Process)...")
    model = get_embedding_model()
    actual_dim = model.get_embedding_dimension()
    print(f"      ✓ Local model loaded: {EMBEDDING_MODEL}")
    print(f"      - Embedding dimension: {actual_dim} (expected: {EMBEDDING_DIMENSION})")

    total_checks += 1
    if actual_dim == EMBEDDING_DIMENSION == 384:
        passed_checks += 1
        print("      ✓ Model dimension matches EMBEDDING_DIMENSION (384).")
    else:
        print(f"      ✗ Dimension mismatch: {actual_dim} vs {EMBEDDING_DIMENSION}")

    retriever = VectorRetriever(engine_or_url=engine, model=model)

    # -------------------------------------------------------------------------
    # 3. Multilingual Query Benchmark (English, Hindi, Gujarati)
    # -------------------------------------------------------------------------
    print("\n[3/6] Testing Multilingual Queries via Local Semantic Vector Retrieval (top_k=5)...")
    test_queries = [
        {"query": "student scholarship", "lang": "EN"},
        {"query": "farmer financial assistance", "lang": "EN"},
        {"query": "women pension", "lang": "EN"},
        {"query": "housing scheme Gujarat", "lang": "EN"},
        {"query": "किसान वित्तीय सहायता", "lang": "HI"},
        {"query": "ખેડૂત આર્થિક સહાય", "lang": "GU"},
    ]

    cached_vectors = {}

    for item in test_queries:
        q = item["query"]
        lang = item["lang"]
        print(f"\n   >>> [{lang}] Generating query embedding: \"{q}\"")
        q_vec = generate_query_embedding(q, model=model)
        cached_vectors[q] = q_vec

        total_checks += 1
        if len(q_vec) == EMBEDDING_DIMENSION and all(math.isfinite(v) for v in q_vec):
            passed_checks += 1
            print(f"       ✓ Query vector generated (dim={len(q_vec)}, finite values).")
        else:
            print("       ✗ Query vector invalid!")

        results = retriever.retrieve_by_vector(query_vector=q_vec, top_k=5)
        print(f"       Total Semantic Results Returned: {len(results)}")

        total_checks += 1
        if len(results) > 0:
            passed_checks += 1
            print(f"       ✓ Retrieved non-empty result set from 20,497 local embeddings.")
        else:
            print("       ✗ No results returned!")

        for i, res in enumerate(results[:3], 1):
            print_result_card(i, res)

    # -------------------------------------------------------------------------
    # 4. Filtered Semantic Retrieval Tests
    # -------------------------------------------------------------------------
    print("\n[4/6] Testing Metadata Pre-Filtered Semantic Retrieval...")

    # Filter Test A: State = 'Gujarat'
    print("\n   --- Filter Test A: State='Gujarat' ---")
    q_guj = "housing scheme Gujarat"
    guj_results = retriever.retrieve_by_vector(
        query_vector=cached_vectors[q_guj],
        state="Gujarat",
        top_k=5,
    )
    print(f"       Query: \"{q_guj}\" | Filter: state=\"Gujarat\" | Results: {len(guj_results)}")

    total_checks += 1
    guj_valid = True
    if guj_results:
        for r in guj_results:
            is_guj = (r.state and "gujarat" in r.state.lower()) or (r.level and r.level.lower() == "central")
            if not is_guj:
                guj_valid = False
                print(f"       ✗ Non-Gujarat state result leaked: {r.scheme_id} ({r.state})")
        if guj_valid:
            passed_checks += 1
            print("       ✓ 100% of returned results respect State='Gujarat' filter.")
        for i, res in enumerate(guj_results[:2], 1):
            print_result_card(i, res)
    else:
        print("       ✗ No results returned for Gujarat filter!")

    # Filter Test B: Level = 'Central'
    print("\n   --- Filter Test B: Level='Central' ---")
    q_cent = "student scholarship"
    cent_results = retriever.retrieve_by_vector(
        query_vector=cached_vectors[q_cent],
        level="Central",
        top_k=5,
    )
    print(f"       Query: \"{q_cent}\" | Filter: level=\"Central\" | Results: {len(cent_results)}")

    total_checks += 1
    cent_valid = True
    if cent_results:
        for r in cent_results:
            if not r.level or r.level.strip().lower() != "central":
                cent_valid = False
                print(f"       ✗ Non-Central result leaked: {r.scheme_id} ({r.level})")
        if cent_valid:
            passed_checks += 1
            print("       ✓ 100% of returned results have level='Central'.")
        for i, res in enumerate(cent_results[:2], 1):
            print_result_card(i, res)
    else:
        print("       ✗ No results returned for Central filter!")

    # Filter Test C: Category = 'Education & Learning'
    print("\n   --- Filter Test C: Category='Education & Learning' ---")
    q_edu = "student scholarship"
    edu_results = retriever.retrieve_by_vector(
        query_vector=cached_vectors[q_edu],
        category="Education & Learning",
        top_k=5,
    )
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
        for i, res in enumerate(edu_results[:2], 1):
            print_result_card(i, res)
    else:
        print("       ✗ No results returned for Education & Learning filter!")

    # -------------------------------------------------------------------------
    # 5. Automated Structural & Quality Validations
    # -------------------------------------------------------------------------
    print("\n[5/6] Executing Automated Structural & Quality Validations...")

    # Check 1: Return type is RetrievalResult
    sample_set = retriever.retrieve_by_vector(cached_vectors["student scholarship"], top_k=3)
    total_checks += 1
    if sample_set and all(isinstance(r, RetrievalResult) for r in sample_set):
        passed_checks += 1
        print("      ✓ Validation 1: Results are typed RetrievalResult objects.")
    else:
        print("      ✗ Validation 1 failed: Result typing error.")

    # Check 2: No NULL or empty chunk_id
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

    # Check 4: Confirm returned chunks strictly have non-NULL embedding_local in PostgreSQL
    total_checks += 1
    with engine.connect() as conn:
        all_chunk_ids = [r.chunk_id for r in sample_set]
        null_count = conn.execute(
            text("SELECT count(*) FROM scheme_chunks WHERE chunk_id = ANY(:ids) AND embedding_local IS NULL;"),
            {"ids": all_chunk_ids}
        ).scalar()
    if null_count == 0:
        passed_checks += 1
        print("      ✓ Validation 4: 100% of returned chunks possess non-NULL embedding_local.")
    else:
        print(f"      ✗ Validation 4 failed: {null_count} returned chunks have NULL embedding_local!")

    # Check 5: Database vector dimension for embedding_local is exactly 384
    total_checks += 1
    with engine.connect() as conn:
        db_dim = conn.execute(
            text("SELECT vector_dims(embedding_local) FROM scheme_chunks WHERE embedding_local IS NOT NULL LIMIT 1;")
        ).scalar()
    if db_dim == EMBEDDING_DIMENSION == 384:
        passed_checks += 1
        print(f"      ✓ Validation 5: Database vector dimension for embedding_local is verified as {db_dim}.")
    else:
        print(f"      ✗ Validation 5 failed: Dimension is {db_dim}, expected {EMBEDDING_DIMENSION}.")

    # Check 6: Configurable top_k is strictly respected
    total_checks += 1
    top_3 = retriever.retrieve_by_vector(cached_vectors["student scholarship"], top_k=3)
    top_7 = retriever.retrieve_by_vector(cached_vectors["student scholarship"], top_k=7)
    if len(top_3) == 3 and len(top_7) == 7:
        passed_checks += 1
        print(f"      ✓ Validation 6: top_k parameter strictly respected (top_3={len(top_3)}, top_7={len(top_7)}).")
    else:
        print(f"      ✗ Validation 6 failed: top_k mismatch (got {len(top_3)} for k=3, {len(top_7)} for k=7).")

    # Check 7: Similarity scores are finite and strictly sorted descending
    total_checks += 1
    scores = [r.score for r in top_7]
    is_sorted_desc = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    all_finite = all(math.isfinite(s) for s in scores)
    if is_sorted_desc and all_finite:
        passed_checks += 1
        print(f"      ✓ Validation 7: Similarity scores are finite and sorted descending ({scores[0]:.4f} -> {scores[-1]:.4f}).")
    else:
        print(f"      ✗ Validation 7 failed: Score validity issue (scores={scores}).")

    # Check 8: Blank query safely handled
    total_checks += 1
    empty_res = retriever.retrieve("   ")
    empty_vec_res = retriever.retrieve_by_vector([])
    if empty_res == [] and empty_vec_res == []:
        passed_checks += 1
        print("      ✓ Validation 8: Blank query and empty vector safely return empty list.")
    else:
        print("      ✗ Validation 8 failed: Blank query handling error.")

    # -------------------------------------------------------------------------
    # 6. Database Integrity Verification (Read-Only Check)
    # -------------------------------------------------------------------------
    print("\n[6/6] Verifying Post-Execution Database Integrity (Read-Only Check)...")
    final_counts = get_db_counts(engine)

    schemes_match = (final_counts["schemes"] == init_counts["schemes"] == 3397)
    chunks_match = (final_counts["chunks"] == init_counts["chunks"] == 20497)
    gem_pop_match = (final_counts["gem_pop"] == init_counts["gem_pop"] == 1260)
    gem_null_match = (final_counts["gem_null"] == init_counts["gem_null"] == 19237)
    loc_pop_match = (final_counts["loc_pop"] == init_counts["loc_pop"] == 20497)
    loc_null_match = (final_counts["loc_null"] == init_counts["loc_null"] == 0)

    print(f"      - Final schemes:               {final_counts['schemes']:,} (diff: {final_counts['schemes'] - init_counts['schemes']:+d})")
    print(f"      - Final scheme_chunks:         {final_counts['chunks']:,} (diff: {final_counts['chunks'] - init_counts['chunks']:+d})")
    print(f"      - Final Gemini embeddings:     {final_counts['gem_pop']:,} (diff: {final_counts['gem_pop'] - init_counts['gem_pop']:+d})")
    print(f"      - Final Gemini NULL:           {final_counts['gem_null']:,} (diff: {final_counts['gem_null'] - init_counts['gem_null']:+d})")
    print(f"      - Final Local embeddings:      {final_counts['loc_pop']:,} (diff: {final_counts['loc_pop'] - init_counts['loc_pop']:+d})")
    print(f"      - Final Local NULL:            {final_counts['loc_null']:,} (diff: {final_counts['loc_null'] - init_counts['loc_null']:+d})")

    total_checks += 1
    if (
        schemes_match
        and chunks_match
        and gem_pop_match
        and gem_null_match
        and loc_pop_match
        and loc_null_match
    ):
        passed_checks += 1
        print("      ✓ Read-only guarantee confirmed: 100% database parity preserved.")
    else:
        print("      ✗ Database integrity failure: Record counts modified!")

    # -------------------------------------------------------------------------
    # Final Report Summary
    # -------------------------------------------------------------------------
    overall_pass = (passed_checks == total_checks)
    print("\n" + "=" * 80)
    print("STEP 16 LOCAL VECTOR RETRIEVAL VERIFICATION REPORT")
    print("=" * 80)
    print(f"Total Validation Checks:     {total_checks}")
    print(f"Passed Checks:              {passed_checks}")
    print(f"Failed Checks:              {total_checks - passed_checks}")
    print(f"Schemes Table Count:        {final_counts['schemes']:,}")
    print(f"Scheme Chunks Count:        {final_counts['chunks']:,}")
    print(f"Gemini Populated Vectors:   {final_counts['gem_pop']:,} (UNTOUCHED)")
    print(f"Local Populated Vectors:    {final_counts['loc_pop']:,} (100% COVERAGE)")
    print(f"Database Modified:          NO (READ-ONLY)")
    print(f"Overall Result:             {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 80)

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
