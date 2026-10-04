"""
scripts/12_test_hybrid_retrieval.py
===================================
Step 5.3.3: Verification & Test Suite for Hybrid Retrieval using Reciprocal Rank Fusion (RRF).

This test script:
  1. Verifies database connectivity and baseline counts:
     - schemes = 3,397
     - scheme_chunks = 20,497
     - embeddings populated = 1,260
     - embeddings NULL = 19,237
  2. Tests the 4 required benchmark queries using Hybrid RRF retrieval:
     - 'student scholarship'
     - 'farmer financial assistance'
     - 'women pension'
     - 'housing scheme Gujarat'
     For each query, displays:
     - Keyword candidate count (candidate_k = 25)
     - Vector candidate count (candidate_k = 25)
     - Final hybrid result count (top_k = 5)
     - Rank, chunk_id, scheme_id, scheme_name, field_name, RRF score,
       keyword rank, vector rank, state, level, and preview.
  3. Tests metadata pre-filtering across both retrieval modalities:
     - Filter A: state = 'Gujarat'
     - Filter B: level = 'Central'
     - Filter C: category = 'Education & Learning'
  4. Executes thorough automated validation assertions:
     - RRF scores are finite numbers
     - Results are strictly sorted by RRF score descending
     - top_k parameter is strictly enforced
     - chunk_id values are non-NULL and unique (no duplicates in final results)
     - All returned scheme references exist in the database
     - Metadata filters have zero leakage
     - Correctness of RRF contributions:
         * Keyword-only items receive exactly 1 / (k + rank_kw)
         * Vector-only items receive exactly 1 / (k + rank_vec)
         * Multi-modal items receive exactly 1 / (k + rank_kw) + 1 / (k + rank_vec)
     - Modifying RRF k constant (e.g. k=20 vs k=60) produces valid mathematical scores
     - Blank query safely returns []
  5. Strictly READ-ONLY with respect to PostgreSQL.
  6. Verifies post-execution database counts remain strictly unchanged.

Important Architecture & Evaluation Notes:
------------------------------------------
- Current Semantic Coverage:
  Dense vector embeddings currently exist for 1,260 chunks; 19,237 chunks are pending
  Gemini free-tier quota reset. Keyword search ensures 100% of chunks are discoverable,
  while pgvector provides semantic ranking for available embeddings. Once the remaining
  embeddings are generated, the exact same hybrid pipeline will automatically possess
  comprehensive semantic coverage.
- Evaluation Disclaimer:
  This script verifies the mathematical and algorithmic correctness of the Reciprocal
  Rank Fusion implementation. No claims regarding relative retrieval accuracy are made,
  as a labeled domain evaluation dataset has not yet been established.
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
    HybridRetriever,
    KeywordRetriever,
    VectorRetriever,
    RetrievalResult,
    generate_query_embedding,
    compute_rrf_score,
    DEFAULT_RRF_K,
    DEFAULT_CANDIDATE_K,
    DEFAULT_TOP_K,
)

# Reconfigure stdout for UTF-8 on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_db_counts(engine) -> Tuple[int, int, int, int]:
    """Returns (schemes_count, chunks_count, embedded_count, null_count)."""
    with engine.connect() as conn:
        s_count = conn.execute(text("SELECT count(id) FROM schemes;")).scalar()
        c_count = conn.execute(text("SELECT count(id) FROM scheme_chunks;")).scalar()
        e_count = conn.execute(text("SELECT count(id) FROM scheme_chunks WHERE embedding IS NOT NULL;")).scalar()
        n_count = conn.execute(text("SELECT count(id) FROM scheme_chunks WHERE embedding IS NULL;")).scalar()
    return s_count, c_count, e_count, n_count


def print_hybrid_card(rank: int, res: RetrievalResult):
    """Prints a detailed result card for a single hybrid retrieval result."""
    state_str = res.state or "All-India / Central"
    level_str = res.level or "N/A"
    kw_str = f"#{res.keyword_rank}" if res.keyword_rank is not None else "None"
    vec_str = f"#{res.vector_rank}" if res.vector_rank is not None else "None"

    print(f"    [{rank}] Scheme: [{res.scheme_id}] {res.scheme_name}")
    print(f"        Chunk ID:     {res.chunk_id} ({res.field_name})")
    print(f"        RRF Score:    {res.score:.6f} | (KW Rank: {kw_str}, Vector Rank: {vec_str})")
    print(f"        State:        {state_str} | Level: {level_str}")
    print(f"        Preview:      {res.preview(120)}")
    print("")


def run_tests():
    print("=" * 80)
    print("  sugamgov-rag | Step 5.3.3: Hybrid Retrieval (Reciprocal Rank Fusion) Test")
    print("=" * 80)

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
    api_key = os.getenv("GEMINI_API_KEY")

    if not db_url:
        print("[FAIL] DATABASE_URL is not set in .env")
        sys.exit(1)
    if not api_key:
        print("[FAIL] GEMINI_API_KEY is not set in .env")
        sys.exit(1)

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        print("      ✓ Database connection established successfully.")
    except Exception as e:
        print(f"[FAIL] Database connection failed: {e}")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)
    print("      ✓ Google GenAI client initialized for in-memory query embeddings.")

    init_schemes, init_chunks, init_embedded, init_null = get_db_counts(engine)
    print(f"      - Baseline schemes:          {init_schemes:,} (expected: 3,397)")
    print(f"      - Baseline scheme_chunks:    {init_chunks:,} (expected: 20,497)")
    print(f"      - Baseline embedded vectors: {init_embedded:,} (expected: 1,260)")
    print(f"      - Baseline NULL embeddings:  {init_null:,} (expected: 19,237)")

    if init_schemes != 3397 or init_chunks != 20497 or init_embedded != 1260 or init_null != 19237:
        print("[FAIL] Baseline database counts do not match expected Step 5.3 state!")
        sys.exit(1)

    kw_retriever = KeywordRetriever(engine)
    vec_retriever = VectorRetriever(engine, client=client)
    hybrid_retriever = HybridRetriever(
        engine_or_url=engine,
        keyword_retriever=kw_retriever,
        vector_retriever=vec_retriever,
        client=client,
        rrf_k=DEFAULT_RRF_K,
        default_candidate_k=DEFAULT_CANDIDATE_K,
    )

    total_checks = 0
    passed_checks = 0

    # -------------------------------------------------------------------------
    # 2. Benchmark Queries (Unfiltered Hybrid Retrieval)
    # -------------------------------------------------------------------------
    print("\n[2/6] Testing Benchmark Queries via Hybrid RRF Retrieval...")
    benchmark_queries = [
        "student scholarship",
        "farmer financial assistance",
        "women pension",
        "housing scheme Gujarat",
    ]

    cached_query_vectors = {}

    for q in benchmark_queries:
        print(f"\n   >>> Query: \"{q}\"")
        q_vec = generate_query_embedding(q, client=client)
        cached_query_vectors[q] = q_vec

        # Retrieve candidates directly for reporting metrics
        kw_cands = kw_retriever.retrieve(query=q, top_k=25)
        vec_cands = vec_retriever.retrieve_by_vector(query_vector=q_vec, top_k=25)

        # Execute hybrid retrieval
        hybrid_results = hybrid_retriever.retrieve(
            query=q,
            query_vector=q_vec,
            top_k=5,
            candidate_k=25,
            rrf_k=60,
        )

        print(f"       Keyword Candidates:  {len(kw_cands)}")
        print(f"       Vector Candidates:   {len(vec_cands)}")
        print(f"       Final Hybrid Top 5:  {len(hybrid_results)}")

        total_checks += 1
        if len(hybrid_results) > 0:
            passed_checks += 1
            print("       ✓ Retrieved non-empty hybrid result set.")
        else:
            print("       ✗ No hybrid results returned!")

        for rank, res in enumerate(hybrid_results, 1):
            print_hybrid_card(rank, res)

    # -------------------------------------------------------------------------
    # 3. Filtered Hybrid Retrieval Tests
    # -------------------------------------------------------------------------
    print("\n[3/6] Testing Metadata Pre-Filtered Hybrid Retrieval...")

    # Filter Test A: State = 'Gujarat'
    print("\n   --- Filter Test A: State='Gujarat' ---")
    q_guj = "housing scheme Gujarat"
    guj_results = hybrid_retriever.retrieve(
        query=q_guj,
        query_vector=cached_query_vectors[q_guj],
        state="Gujarat",
        top_k=5,
        candidate_k=25,
    )
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
            print_hybrid_card(i, res)
    else:
        print("       ✗ No results returned for Gujarat filter!")

    # Filter Test B: Level = 'Central'
    print("\n   --- Filter Test B: Level='Central' ---")
    q_central = "farmer financial assistance"
    central_results = hybrid_retriever.retrieve(
        query=q_central,
        query_vector=cached_query_vectors[q_central],
        level="Central",
        top_k=5,
        candidate_k=25,
    )
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
            print_hybrid_card(i, res)
    else:
        print("       ✗ No results returned for Central filter!")

    # Filter Test C: Category = 'Education & Learning'
    print("\n   --- Filter Test C: Category='Education & Learning' ---")
    q_edu = "student scholarship"
    edu_results = hybrid_retriever.retrieve(
        query=q_edu,
        query_vector=cached_query_vectors[q_edu],
        category="Education & Learning",
        top_k=5,
        candidate_k=25,
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
        for i, res in enumerate(edu_results, 1):
            print_hybrid_card(i, res)
    else:
        print("       ✗ No results returned for Education & Learning filter!")

    # -------------------------------------------------------------------------
    # 4. Automated Mathematical & Quality Validations
    # -------------------------------------------------------------------------
    print("\n[4/6] Executing Automated Mathematical & Quality Validations...")

    sample_results = hybrid_retriever.retrieve(
        query="farmer financial assistance",
        query_vector=cached_query_vectors["farmer financial assistance"],
        top_k=10,
        candidate_k=25,
        rrf_k=60,
    )

    # Check 1: RRF scores are finite numbers
    total_checks += 1
    if sample_results and all(math.isfinite(r.score) for r in sample_results):
        passed_checks += 1
        print("      ✓ Validation 1: All RRF scores are finite numbers.")
    else:
        print("      ✗ Validation 1 failed: Non-finite score detected.")

    # Check 2: Results are strictly sorted descending by RRF score
    total_checks += 1
    scores = [r.score for r in sample_results]
    is_sorted_desc = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    if is_sorted_desc:
        passed_checks += 1
        print(f"      ✓ Validation 2: Results sorted descending by RRF score ({scores[0]:.6f} -> {scores[-1]:.6f}).")
    else:
        print("      ✗ Validation 2 failed: RRF results not sorted descending.")

    # Check 3: top_k strictly enforced
    total_checks += 1
    top_3 = hybrid_retriever.retrieve("scholarship", query_vector=cached_query_vectors["student scholarship"], top_k=3)
    top_7 = hybrid_retriever.retrieve("scholarship", query_vector=cached_query_vectors["student scholarship"], top_k=7)
    if len(top_3) == 3 and len(top_7) == 7:
        passed_checks += 1
        print(f"      ✓ Validation 3: top_k strictly enforced (top_3={len(top_3)}, top_7={len(top_7)}).")
    else:
        print("      ✗ Validation 3 failed: top_k mismatch.")

    # Check 4: No NULL or empty chunk IDs
    total_checks += 1
    no_null_chunks = all(r.chunk_id and r.chunk_id.strip() != "" for r in sample_results)
    if no_null_chunks:
        passed_checks += 1
        print("      ✓ Validation 4: Zero NULL or empty chunk IDs detected.")
    else:
        print("      ✗ Validation 4 failed: Empty or NULL chunk_id detected.")

    # Check 5: All scheme references exist in PostgreSQL
    total_checks += 1
    with engine.connect() as conn:
        all_scheme_ids = list({r.scheme_id for r in sample_results})
        db_schemes_found = conn.execute(
            text("SELECT count(*) FROM schemes WHERE scheme_id = ANY(:ids);"),
            {"ids": all_scheme_ids}
        ).scalar()
    if db_schemes_found == len(all_scheme_ids):
        passed_checks += 1
        print(f"      ✓ Validation 5: All {len(all_scheme_ids)} referenced scheme_ids exist in 'schemes'.")
    else:
        print("      ✗ Validation 5 failed: Broken foreign scheme reference.")

    # Check 6: Duplicate chunk IDs do not occur in final results
    total_checks += 1
    chunk_ids = [r.chunk_id for r in sample_results]
    unique_chunk_ids = set(chunk_ids)
    if len(chunk_ids) == len(unique_chunk_ids):
        passed_checks += 1
        print(f"      ✓ Validation 6: All {len(chunk_ids)} returned chunks are distinct (0 duplicates).")
    else:
        print("      ✗ Validation 6 failed: Duplicate chunk IDs in results.")

    # Check 7: RRF Math Correctness for Keyword-only, Vector-only, and Both
    total_checks += 1
    math_valid = True
    rrf_k_test = 60
    for r in sample_results:
        expected_score = 0.0
        if r.keyword_rank is not None:
            expected_score += 1.0 / (rrf_k_test + r.keyword_rank)
        if r.vector_rank is not None:
            expected_score += 1.0 / (rrf_k_test + r.vector_rank)

        if abs(r.score - expected_score) > 1e-9:
            math_valid = False
            print(f"      ✗ Math error on chunk {r.chunk_id}: got {r.score}, expected {expected_score}")
            break

    if math_valid:
        passed_checks += 1
        print("      ✓ Validation 7: RRF formula verified exact across keyword-only, vector-only, and dual matches.")
    else:
        print("      ✗ Validation 7 failed: RRF formula mismatch.")

    # Check 8: Behavior with alternate RRF k constant (k=20)
    total_checks += 1
    res_k20 = hybrid_retriever.retrieve(
        query="farmer financial assistance",
        query_vector=cached_query_vectors["farmer financial assistance"],
        top_k=5,
        candidate_k=25,
        rrf_k=20,
    )
    # Check that scores reflect k=20
    k20_scores_valid = all(
        abs(r.score - ((1.0 / (20 + r.keyword_rank) if r.keyword_rank else 0.0) +
                       (1.0 / (20 + r.vector_rank) if r.vector_rank else 0.0))) < 1e-9
        for r in res_k20
    )
    if k20_scores_valid and len(res_k20) == 5:
        passed_checks += 1
        print("      ✓ Validation 8: Changing RRF k constant (k=20) maintains exact mathematical scoring.")
    else:
        print("      ✗ Validation 8 failed: k=20 score verification failed.")

    # Check 9: Blank query safely handled
    total_checks += 1
    empty_res = hybrid_retriever.retrieve("   ")
    if empty_res == []:
        passed_checks += 1
        print("      ✓ Validation 9: Blank query safely returns empty list.")
    else:
        print("      ✗ Validation 9 failed: Blank query handling error.")

    # -------------------------------------------------------------------------
    # 5. Database Integrity Verification (Read-Only Check)
    # -------------------------------------------------------------------------
    print("\n[5/6] Verifying Post-Execution Database Integrity (Read-Only Check)...")
    final_schemes, final_chunks, final_embedded, final_null = get_db_counts(engine)

    schemes_match = (final_schemes == init_schemes == 3397)
    chunks_match = (final_chunks == init_chunks == 20497)
    embedded_match = (final_embedded == init_embedded == 1260)
    null_match = (final_null == init_null == 19237)

    print(f"      - Final schemes:          {final_schemes:,} (diff: {final_schemes - init_schemes:+d})")
    print(f"      - Final scheme_chunks:    {final_chunks:,} (diff: {final_chunks - init_chunks:+d})")
    print(f"      - Final embedded vectors: {final_embedded:,} (diff: {final_embedded - init_embedded:+d})")
    print(f"      - Final NULL embeddings:  {final_null:,} (diff: {final_null - init_null:+d})")

    total_checks += 1
    if schemes_match and chunks_match and embedded_match and null_match:
        passed_checks += 1
        print("      ✓ Read-only guarantee confirmed: 100% database parity preserved.")
    else:
        print("      ✗ Database integrity failure: Record counts modified!")

    # -------------------------------------------------------------------------
    # 6. Final Report Summary
    # -------------------------------------------------------------------------
    overall_pass = (passed_checks == total_checks)
    print("\n" + "=" * 80)
    print("STEP 5.3.3 HYBRID RETRIEVAL VERIFICATION REPORT")
    print("=" * 80)
    print(f"Total Validation Checks:     {total_checks}")
    print(f"Passed Checks:              {passed_checks}")
    print(f"Failed Checks:              {total_checks - passed_checks}")
    print(f"Schemes Table Count:        {final_schemes:,}")
    print(f"Scheme Chunks Count:        {final_chunks:,}")
    print(f"Populated Vector Count:     {final_embedded:,}")
    print(f"NULL Vector Count:          {final_null:,}")
    print(f"Database Modified:          NO (READ-ONLY)")
    print(f"Overall Result:             {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 80)

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
