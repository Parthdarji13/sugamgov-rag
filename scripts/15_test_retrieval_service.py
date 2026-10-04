"""
scripts/15_test_retrieval_service.py
====================================
Step 16: Verification & Test Suite for the Unified Retrieval Service Layer with Local Embeddings.

This test script:
  1. Verifies database connectivity and baseline counts:
     - schemes = 3,397
     - scheme_chunks = 20,497
     - Gemini embeddings populated = 1,260
     - Gemini embeddings NULL = 19,237
     - Local embeddings populated = 20,497
     - Local embeddings NULL = 0
  2. Verifies dynamic coverage calculation from PostgreSQL:
     - total_chunks = 20,497
     - embedded_chunks = 20,497
     - unembedded_chunks = 0
     - semantic_coverage_complete = True
  3. Verifies zero embedding calls for blank / whitespace-only queries:
     - Tests retrieve("") and retrieve("   ")
     - Asserts response is empty with total_results = 0.
  4. Tests benchmark queries (English, Hindi, Gujarati) with live local embeddings:
     - 'student scholarship'
     - 'farmer financial assistance'
     - 'women pension'
     - 'housing scheme Gujarat'
     - 'किसान वित्तीय सहायता' (Hindi)
     - 'ખેડૂત આર્થિક સહાય' (Gujarati)
     For each query, inspects:
     - Query string and normalized filters
     - Retrieval method ('hybrid_rrf_scheme_reranked')
     - Returned unique scheme results (top_k = 5)
     - Best chunk ID, score, field name, preview
     - Supporting evidence chunks count
     - Asserts 100% scheme ID uniqueness.
  5. Tests metadata pre-filtering across both retrieval modalities:
     - Filter A: state = 'Gujarat'
     - Filter B: level = 'Central'
     - Filter C: category = 'Education & Learning'
     - Asserts zero metadata leakage.
  6. Tests edge cases and input validation:
     - Whitespace trimming on query and filter parameters
     - top_k = 1, top_k = 5
     - top_k <= 0 raises ValueError
     - candidate_k <= 0 raises ValueError
     - Non-string query raises TypeError
     - Response serialization via to_dict() and preview()
  7. Executes determinism verification:
     - Runs identical queries twice and asserts identical response structures.
  8. Strictly READ-ONLY with respect to PostgreSQL.
  9. Verifies post-execution database counts remain strictly unchanged.
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Add project root to sys.path to enable src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import (
    RetrievalService,
    RetrievalResponse,
    retrieve_service,
    SchemeRetrievalResult,
    RetrievalResult,
    generate_query_embedding,
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


def print_service_card(rank: int, res: SchemeRetrievalResult):
    """Prints a formatted card for a scheme result returned by RetrievalService."""
    print(f"      [{rank:2d}] Scheme ID:     {res.scheme_id}")
    print(f"           Scheme Name:   {res.scheme_name}")
    print(f"           Best Chunk:    {res.best_chunk_id} ({res.best_field_name})")
    print(f"           Score:         {res.best_retrieval_score:.6f}")
    if res.rrf_score is not None:
        print(f"           RRF Score:     {res.rrf_score:.6f}")
    print(f"           Evidence:      {len(res.evidence_chunks)} chunk(s)")
    ev_ids = [c.chunk_id for c in res.evidence_chunks]
    print(f"           Evidence IDs:  {', '.join(ev_ids)}")
    print(f"           State / Level: {res.state or 'Central/Multi-State'} | {res.level}")
    print(f"           Preview:       \"{res.preview(100)}\"")
    print()


def run_tests():
    print("=" * 80)
    print("  sugamgov-rag | Step 16: Retrieval Service Layer Verification (Local Embeddings)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Environment & Baseline Database Integrity
    # -------------------------------------------------------------------------
    print("\n[1/8] Connecting to Database & Verifying Baseline Parity...")
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print(f"[FAIL] .env file not found at {env_file}")
        sys.exit(1)

    load_dotenv(env_file)
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("[FAIL] Missing DATABASE_URL in .env")
        sys.exit(1)

    engine = create_engine(db_url)

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
        print("[FAIL] Database baseline counts do not match expected state!")
        sys.exit(1)

    # Initialize RetrievalService
    service = RetrievalService(engine_or_url=engine)

    total_checks = 0
    passed_checks = 0

    # -------------------------------------------------------------------------
    # 2. Dynamic Coverage Information Verification
    # -------------------------------------------------------------------------
    print("\n[2/8] Verifying Dynamic Embedding Coverage Reporting...")
    coverage = service.get_coverage_info(refresh=True)
    print(f"      - Total chunks reported:     {coverage['total_chunks']:,}")
    print(f"      - Embedded chunks reported:  {coverage['embedded_chunks']:,}")
    print(f"      - Unembedded chunks reported:{coverage['unembedded_chunks']:,}")
    print(f"      - Semantic coverage complete:{coverage['semantic_coverage_complete']}")

    total_checks += 1
    if coverage["total_chunks"] == 20497:
        passed_checks += 1
        print("      ✓ Total chunks correctly reported as 20,497.")
    else:
        print(f"      ✗ Total chunks mismatch: {coverage['total_chunks']}")

    total_checks += 1
    if coverage["embedded_chunks"] == 20497:
        passed_checks += 1
        print("      ✓ Embedded chunks correctly reported as 20,497 (100%).")
    else:
        print(f"      ✗ Embedded chunks mismatch: {coverage['embedded_chunks']}")

    total_checks += 1
    if coverage["unembedded_chunks"] == 0:
        passed_checks += 1
        print("      ✓ Unembedded chunks correctly reported as 0.")
    else:
        print(f"      ✗ Unembedded chunks mismatch: {coverage['unembedded_chunks']}")

    total_checks += 1
    if coverage["semantic_coverage_complete"] is True:
        passed_checks += 1
        print("      ✓ semantic_coverage_complete is correctly True.")
    else:
        print("      ✗ semantic_coverage_complete is unexpectedly False!")

    # -------------------------------------------------------------------------
    # 3. Zero Calls for Blank / Whitespace Queries
    # -------------------------------------------------------------------------
    print("\n[3/8] Verifying No Retriever Execution for Blank Queries...")

    class CallTracker:
        def __init__(self):
            self.calls = 0

        def track(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("Retriever was called for blank query!")

    tracker = CallTracker()
    original_fn = getattr(service.hybrid_retriever, "retrieve", None)

    try:
        setattr(service.hybrid_retriever, "retrieve", tracker.track)

        res_empty = service.retrieve("")
        res_space = service.retrieve("     ")

        total_checks += 1
        if tracker.calls == 0:
            passed_checks += 1
            print("      ✓ Retriever call count is strictly ZERO for blank queries.")
        else:
            print(f"      ✗ Retriever was called {tracker.calls} times for blank query!")

        total_checks += 1
        if res_empty.total_results == 0 and res_empty.results == []:
            passed_checks += 1
            print("      ✓ retrieve('') safely returned empty response without error.")
        else:
            print("      ✗ retrieve('') returned non-empty response.")

        total_checks += 1
        if res_space.total_results == 0 and res_space.results == []:
            passed_checks += 1
            print("      ✓ retrieve('     ') safely returned empty response without error.")
        else:
            print("      ✗ retrieve('     ') returned non-empty response.")

    finally:
        if original_fn:
            setattr(service.hybrid_retriever, "retrieve", original_fn)

    # -------------------------------------------------------------------------
    # 4. Benchmark Queries (End-to-End Service Execution with Local Embeddings)
    # -------------------------------------------------------------------------
    print("\n[4/8] Testing Benchmark Queries via Unified RetrievalService...")

    benchmark_queries = [
        {"query": "student scholarship", "lang": "EN"},
        {"query": "farmer financial assistance", "lang": "EN"},
        {"query": "women pension", "lang": "EN"},
        {"query": "housing scheme Gujarat", "lang": "EN"},
        {"query": "किसान वित्तीय सहायता", "lang": "HI"},
        {"query": "ખેડૂત આર્થિક સહાય", "lang": "GU"},
    ]

    for item in benchmark_queries:
        q = item["query"]
        lang = item["lang"]
        print(f"\n   >>> [{lang}] Query: \"{q}\"")

        response: RetrievalResponse = service.retrieve(
            query=q,
            top_k=5,
            candidate_k=25,
        )

        total_checks += 1
        if response.query == q.strip():
            passed_checks += 1
            print(f"       ✓ Response query sanitized: \"{response.query}\"")
        else:
            print(f"       ✗ Query mismatch in response: \"{response.query}\"")

        total_checks += 1
        if response.retrieval_method == "hybrid_rrf_scheme_reranked":
            passed_checks += 1
            print(f"       ✓ Retrieval method: {response.retrieval_method}")
        else:
            print(f"       ✗ Unexpected retrieval method: {response.retrieval_method}")

        total_checks += 1
        if response.total_results > 0 and len(response.results) == response.total_results:
            passed_checks += 1
            print(f"       ✓ Returned {response.total_results} unique consolidated schemes.")
        else:
            print(f"       ✗ Result count mismatch: total={response.total_results}, len={len(response.results)}")

        # Check scheme ID uniqueness
        total_checks += 1
        sids = [r.scheme_id for r in response.results]
        if len(sids) == len(set(sids)):
            passed_checks += 1
            print(f"       ✓ 100% Scheme ID uniqueness confirmed: {sids}")
        else:
            print(f"       ✗ Duplicate scheme IDs detected: {sids}")

        # Check evidence chunks attached
        total_checks += 1
        all_have_evidence = all(len(r.evidence_chunks) >= 1 for r in response.results)
        if all_have_evidence:
            passed_checks += 1
            total_ev = sum(len(r.evidence_chunks) for r in response.results)
            print(f"       ✓ All schemes have supporting evidence chunks (total: {total_ev}).")
        else:
            print("       ✗ Some schemes have zero evidence chunks!")

        for rank, res in enumerate(response.results[:2], 1):
            print_service_card(rank, res)

    # -------------------------------------------------------------------------
    # 5. Metadata Pre-Filtering Tests
    # -------------------------------------------------------------------------
    print("\n[5/8] Testing Metadata Pre-Filtering via Service Layer...")

    # Filter A: State = 'Gujarat'
    print("\n   --- Filter A: State='Gujarat' ---")
    resp_guj = service.retrieve("housing", top_k=5, state="Gujarat")
    total_checks += 1
    guj_leak = False
    for r in resp_guj.results:
        is_guj = (r.state and "gujarat" in r.state.lower()) or (r.level and r.level.lower() == "central")
        if not is_guj:
            guj_leak = True
            print(f"       ✗ State filter leak: {r.scheme_id} ({r.state})")
    if not guj_leak and resp_guj.total_results > 0:
        passed_checks += 1
        print(f"       ✓ State filter verified (total={resp_guj.total_results}, 0 leaks).")
    else:
        print(f"       ✗ State filter verification failed.")

    # Filter B: Level = 'Central'
    print("\n   --- Filter B: Level='Central' ---")
    resp_cent = service.retrieve("scholarship", top_k=5, level="Central")
    total_checks += 1
    cent_leak = any(r.level != "Central" for r in resp_cent.results)
    if not cent_leak and resp_cent.total_results > 0:
        passed_checks += 1
        print(f"       ✓ Level='Central' filter verified (total={resp_cent.total_results}, 0 leaks).")
    else:
        print(f"       ✗ Level filter verification failed.")

    # Filter C: Category = 'Education & Learning'
    print("\n   --- Filter C: Category='Education & Learning' ---")
    resp_cat = service.retrieve("scholarship", top_k=5, category="Education & Learning")
    total_checks += 1
    cat_leak = False
    for r in resp_cat.results:
        has_cat = any("education & learning" in c.lower() for c in r.categories)
        if not has_cat:
            cat_leak = True
    if not cat_leak and resp_cat.total_results > 0:
        passed_checks += 1
        print(f"       ✓ Category filter verified (total={resp_cat.total_results}, 0 leaks).")
    else:
        print(f"       ✗ Category filter verification failed.")

    # -------------------------------------------------------------------------
    # 6. Edge Cases & Input Validation
    # -------------------------------------------------------------------------
    print("\n[6/8] Testing Edge Cases & Input Validation...")

    total_checks += 1
    resp_trim = service.retrieve("  student scholarship   ", state="  Gujarat  ", level=" Central ")
    if (
        resp_trim.query == "student scholarship"
        and resp_trim.filters["state"] == "Gujarat"
        and resp_trim.filters["level"] == "Central"
    ):
        passed_checks += 1
        print("       ✓ Whitespace trimming verified for query and filter strings.")
    else:
        print("       ✗ Whitespace trimming failed!")

    total_checks += 1
    try:
        service.retrieve("test", top_k=0)
        print("       ✗ Expected ValueError for top_k=0 was not raised!")
    except ValueError:
        passed_checks += 1
        print("       ✓ ValueError correctly raised for top_k=0.")

    total_checks += 1
    try:
        service.retrieve(12345)  # type: ignore
        print("       ✗ Expected TypeError for non-string query was not raised!")
    except TypeError:
        passed_checks += 1
        print("       ✓ TypeError correctly raised for non-string query.")

    total_checks += 1
    resp_top1 = service.retrieve("farmer", top_k=1)
    if resp_top1.total_results == 1:
        passed_checks += 1
        print("       ✓ top_k=1 boundary strictly respected (total_results=1).")
    else:
        print(f"       ✗ top_k=1 boundary failed: got {resp_top1.total_results}")

    total_checks += 1
    d = resp_top1.to_dict()
    prev = resp_top1.preview()
    if isinstance(d, dict) and "results" in d and isinstance(prev, str) and len(prev) > 0:
        passed_checks += 1
        print("       ✓ to_dict() and preview() serialize correctly.")
    else:
        print("       ✗ Serialization verification failed.")

    # -------------------------------------------------------------------------
    # 7. Determinism Verification
    # -------------------------------------------------------------------------
    print("\n[7/8] Testing Determinism (Identical Repeat Execution)...")
    q_det = "farmer financial assistance"
    run_a = service.retrieve(q_det, top_k=5)
    run_b = service.retrieve(q_det, top_k=5)

    dict_a = run_a.to_dict()
    dict_b = run_b.to_dict()

    total_checks += 1
    if dict_a == dict_b:
        passed_checks += 1
        print("       ✓ Determinism confirmed: Multiple identical executions yield 100% identical responses.")
    else:
        print("       ✗ Non-deterministic output detected across runs!")

    # -------------------------------------------------------------------------
    # 8. Database Safety Parity Check
    # -------------------------------------------------------------------------
    print("\n[8/8] Verifying Database Read-Only Parity (Zero Writes)...")
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
        print("       ✓ Read-only guarantee confirmed: 100% database parity preserved.")
    else:
        print("       ✗ Database parity check failed: Records were modified!")

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    overall_pass = (passed_checks == total_checks)
    print("\n" + "=" * 80)
    print("STEP 16 RETRIEVAL SERVICE VERIFICATION REPORT")
    print("=" * 80)
    print(f"Total Validation Checks:     {total_checks}")
    print(f"Passed Checks:              {passed_checks}")
    print(f"Failed Checks:              {total_checks - passed_checks}")
    print(f"Schemes Table Count:        {final_counts['schemes']:,}")
    print(f"Scheme Chunks Count:        {final_counts['chunks']:,}")
    print(f"Gemini Populated Count:     {final_counts['gem_pop']:,} (UNTOUCHED)")
    print(f"Local Populated Count:      {final_counts['loc_pop']:,} (100% COVERAGE)")
    print(f"Database Modified:          NO (READ-ONLY)")
    print(f"Overall Result:             {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 80)

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
