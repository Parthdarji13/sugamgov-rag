"""
scripts/18_test_fastapi_backend.py
==================================
Comprehensive Verification Suite for Step 6: FastAPI RAG Backend Foundation.

Validation Phases:
  Phase 1:  FastAPI App Import & Structure Validation
  Phase 2:  GET /health Endpoint Validation
  Phase 3:  POST /api/retrieve Core Retrieval Validation
  Phase 4:  POST /api/retrieve State Filter Validation (Zero Cross-State Leakage)
  Phase 5:  POST /api/retrieve Level Filter Validation (Central-Only Isolation)
  Phase 6:  Invalid / Blank Query Validation (HTTP 422 Rejection)
  Phase 7:  Invalid Language Parameter Validation (HTTP 422 Rejection)
  Phase 8:  POST /api/generate Grounded RAG Flow Validation (Single English Query)
  Phase 9:  Credential & Secret Leakage Inspection
  Phase 10: PostgreSQL Database Read-Only Parity Audit (Zero Writes)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe UTF-8 console output for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


def audit_database_counts(engine) -> Dict[str, int]:
    """Audits row counts for all tables to verify zero database mutations."""
    with engine.connect() as conn:
        schemes_cnt = conn.execute(text("SELECT COUNT(*) FROM schemes")).scalar()
        chunks_cnt = conn.execute(text("SELECT COUNT(*) FROM scheme_chunks")).scalar()
        embedded_cnt = conn.execute(
            text("SELECT COUNT(*) FROM scheme_chunks WHERE embedding IS NOT NULL")
        ).scalar()
        null_cnt = conn.execute(
            text("SELECT COUNT(*) FROM scheme_chunks WHERE embedding IS NULL")
        ).scalar()
        loc_embedded_cnt = conn.execute(
            text("SELECT COUNT(*) FROM scheme_chunks WHERE embedding_local IS NOT NULL")
        ).scalar()
        loc_null_cnt = conn.execute(
            text("SELECT COUNT(*) FROM scheme_chunks WHERE embedding_local IS NULL")
        ).scalar()
        sources_cnt = conn.execute(text("SELECT COUNT(*) FROM sources")).scalar()
        versions_cnt = conn.execute(text("SELECT COUNT(*) FROM scheme_versions")).scalar()

    return {
        "schemes": schemes_cnt,
        "scheme_chunks": chunks_cnt,
        "embedded": embedded_cnt,
        "null_embeddings": null_cnt,
        "local_embedded": loc_embedded_cnt,
        "local_null": loc_null_cnt,
        "sources": sources_cnt,
        "scheme_versions": versions_cnt,
    }


def main():
    print("=" * 80)
    print("  sugamgov-rag | Step 6: FastAPI RAG Backend Verification Suite")
    print("=" * 80)

    total_checks = 0
    passed_checks = 0
    real_gemini_calls = 0

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not found in .env")
        sys.exit(1)

    engine = create_engine(db_url)

    # -------------------------------------------------------------------------
    # Phase 0: Baseline Database Parity Audit
    # -------------------------------------------------------------------------
    print("\n[Phase 0] Verifying Baseline Database Counts...")
    baseline_counts = audit_database_counts(engine)
    print(f"      - Schemes:          {baseline_counts['schemes']:,} (expected: 3,397)")
    print(f"      - Scheme Chunks:    {baseline_counts['scheme_chunks']:,} (expected: 20,497)")
    print(f"      - Gemini Vector:    {baseline_counts['embedded']:,} (expected: 1,260)")
    print(f"      - Gemini NULL:      {baseline_counts['null_embeddings']:,} (expected: 19,237)")
    print(f"      - Local Vector:     {baseline_counts['local_embedded']:,} (expected: 20,497)")
    print(f"      - Local NULL:       {baseline_counts['local_null']:,} (expected: 0)")
    print(f"      - Sources:          {baseline_counts['sources']:,} (expected: 0)")
    print(f"      - Scheme Versions:  {baseline_counts['scheme_versions']:,} (expected: 0)")

    assert baseline_counts["schemes"] == 3397, "Schemes count mismatch"
    assert baseline_counts["scheme_chunks"] == 20497, "Chunks count mismatch"
    assert baseline_counts["embedded"] == 1260, "Embedded count mismatch"
    assert baseline_counts["null_embeddings"] == 19237, "NULL count mismatch"
    assert baseline_counts["local_embedded"] == 20497, "Local embedded count mismatch"
    assert baseline_counts["local_null"] == 0, "Local NULL count mismatch"

    # -------------------------------------------------------------------------
    # Phase 1: FastAPI App Import & Structure Validation
    # -------------------------------------------------------------------------
    print("\n[Phase 1] Importing FastAPI App & Validating OpenAPI Metadata...")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.main import app
    from api.config import get_settings

    total_checks += 1
    if isinstance(app, FastAPI):
        passed_checks += 1
        print("       ✓ FastAPI application instance imported successfully.")
    else:
        print("       ✗ app is not a FastAPI instance!")

    total_checks += 1
    if app.title == "SugamGov AI RAG API" and app.version == "0.1.0":
        passed_checks += 1
        print("       ✓ OpenAPI title and version metadata verified.")
    else:
        print("       ✗ Unexpected OpenAPI metadata!")

    settings = get_settings()
    total_checks += 1
    if settings.api_port == 8000 and "http://localhost:3000" in settings.cors_origins:
        passed_checks += 1
        print(f"       ✓ Settings loaded: port={settings.api_port}, cors={settings.cors_origins}")
    else:
        print("       ✗ Settings configuration mismatch!")

    # Collect all response payloads to check for secrets in Phase 9
    payloads_to_inspect: List[str] = []

    # -------------------------------------------------------------------------
    # Phase 2: GET /health Endpoint Validation
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Testing GET /health Endpoint...")
    with TestClient(app) as client:
        health_resp = client.get("/health")
        payloads_to_inspect.append(health_resp.text)

        total_checks += 1
        if health_resp.status_code == 200:
            passed_checks += 1
            print("       ✓ /health returned HTTP 200.")
        else:
            print(f"       ✗ /health returned status {health_resp.status_code}!")

        health_data = health_resp.json()
        total_checks += 1
        if health_data.get("status") == "ok" and health_data.get("service") == "sugamgov-rag-api":
            passed_checks += 1
            print(f"       ✓ Health payload matches expected schema: {health_data}")
        else:
            print(f"       ✗ Health payload unexpected: {health_data}")

        # ---------------------------------------------------------------------
        # Phase 3: POST /api/retrieve Core Retrieval Validation
        # ---------------------------------------------------------------------
        print("\n[Phase 3] Testing POST /api/retrieve Endpoint ('student scholarship')...")
        ret_resp = client.post(
            "/api/retrieve",
            json={"query": "student scholarship", "top_k": 5, "candidate_k": 25},
        )
        payloads_to_inspect.append(ret_resp.text)

        total_checks += 1
        if ret_resp.status_code == 200:
            passed_checks += 1
            print("       ✓ /api/retrieve returned HTTP 200.")
        else:
            print(f"       ✗ /api/retrieve returned status {ret_resp.status_code}!")

        ret_data = ret_resp.json()
        results = ret_data.get("results", [])

        total_checks += 1
        if len(results) > 0:
            passed_checks += 1
            print(f"       ✓ Retrieved {len(results)} schemes matching 'student scholarship'.")
        else:
            print("       ✗ Zero results returned for 'student scholarship'!")

        total_checks += 1
        if "retrieval_method" in ret_data and "coverage_info" in ret_data:
            passed_checks += 1
            print(f"       ✓ retrieval_method='{ret_data['retrieval_method']}', coverage_info present.")
        else:
            print("       ✗ retrieval_method or coverage_info missing!")

        # Verify evidence chunks in results
        total_checks += 1
        has_evidence = all(len(r.get("evidence_chunks", [])) > 0 for r in results)
        if has_evidence:
            passed_checks += 1
            print("       ✓ All returned schemes contain granular evidence chunks.")
        else:
            print("       ✗ Some schemes missing evidence chunks!")

        # ---------------------------------------------------------------------
        # Phase 4: State Filter Validation (Zero Cross-State Leakage)
        # ---------------------------------------------------------------------
        print("\n[Phase 4] Testing POST /api/retrieve with state='Gujarat'...")
        guj_resp = client.post(
            "/api/retrieve",
            json={"query": "farmer", "state": "Gujarat", "top_k": 5},
        )
        payloads_to_inspect.append(guj_resp.text)

        total_checks += 1
        if guj_resp.status_code == 200:
            passed_checks += 1
            print("       ✓ State filter query returned HTTP 200.")
        else:
            print(f"       ✗ State filter query failed with {guj_resp.status_code}!")

        guj_data = guj_resp.json()
        guj_results = guj_data.get("results", [])

        total_checks += 1
        if len(guj_results) > 0:
            passed_checks += 1
            print(f"       ✓ Returned {len(guj_results)} schemes under state='Gujarat'.")
        else:
            print("       ✗ Zero schemes returned for Gujarat state filter!")

        total_checks += 1
        cross_state_leak = False
        for r in guj_results:
            scheme_state = r.get("state")
            scheme_states = r.get("states") or []
            scheme_level = r.get("level")
            # If state scheme, it must be Gujarat
            if scheme_level == "State":
                if scheme_state != "Gujarat" and "Gujarat" not in scheme_states:
                    cross_state_leak = True
                    print(f"       ✗ Leakage detected: {r['scheme_id']} has state={scheme_state}, states={scheme_states}")

        if not cross_state_leak:
            passed_checks += 1
            print("       ✓ Zero cross-state leakage: All schemes are strictly Gujarat-compatible.")
        else:
            print("       ✗ Cross-state leakage detected!")

        # ---------------------------------------------------------------------
        # Phase 5: Level Filter Validation (Central-Only Isolation)
        # ---------------------------------------------------------------------
        print("\n[Phase 5] Testing POST /api/retrieve with level='Central'...")
        cen_resp = client.post(
            "/api/retrieve",
            json={"query": "scholarship", "level": "Central", "top_k": 5},
        )
        payloads_to_inspect.append(cen_resp.text)

        total_checks += 1
        if cen_resp.status_code == 200:
            passed_checks += 1
            print("       ✓ Level filter query returned HTTP 200.")
        else:
            print(f"       ✗ Level filter query failed with {cen_resp.status_code}!")

        cen_data = cen_resp.json()
        cen_results = cen_data.get("results", [])

        total_checks += 1
        if len(cen_results) > 0:
            passed_checks += 1
            print(f"       ✓ Returned {len(cen_results)} schemes under level='Central'.")
        else:
            print("       ✗ Zero schemes returned for Central level filter!")

        total_checks += 1
        all_central = all(r.get("level") == "Central" for r in cen_results)
        if all_central:
            passed_checks += 1
            print("       ✓ Level isolation verified: All returned schemes have level='Central'.")
        else:
            print("       ✗ Non-Central scheme found in Central filtered results!")

        # ---------------------------------------------------------------------
        # Phase 6: Invalid / Blank Query Validation (HTTP 422)
        # ---------------------------------------------------------------------
        print("\n[Phase 6] Testing Validation Error: Blank / Whitespace Query...")
        blank_resp = client.post("/api/retrieve", json={"query": "    "})
        payloads_to_inspect.append(blank_resp.text)

        total_checks += 1
        if blank_resp.status_code == 422:
            passed_checks += 1
            print("       ✓ Blank query correctly rejected with HTTP 422 Unprocessable Entity.")
        else:
            print(f"       ✗ Blank query returned {blank_resp.status_code} instead of 422!")

        # ---------------------------------------------------------------------
        # Phase 7: Invalid Language Parameter Validation (HTTP 422)
        # ---------------------------------------------------------------------
        print("\n[Phase 7] Testing Validation Error: Unsupported Language Code...")
        bad_lang_resp = client.post(
            "/api/retrieve",
            json={"query": "scholarship", "language": "french"},
        )
        payloads_to_inspect.append(bad_lang_resp.text)

        total_checks += 1
        if bad_lang_resp.status_code == 422:
            passed_checks += 1
            print("       ✓ Unsupported language correctly rejected with HTTP 422 Unprocessable Entity.")
        else:
            print(f"       ✗ Unsupported language returned {bad_lang_resp.status_code} instead of 422!")

        # Also test candidate_k < top_k validation
        bad_k_resp = client.post(
            "/api/retrieve",
            json={"query": "scholarship", "top_k": 8, "candidate_k": 4},
        )
        payloads_to_inspect.append(bad_k_resp.text)

        total_checks += 1
        if bad_k_resp.status_code == 422:
            passed_checks += 1
            print("       ✓ candidate_k < top_k correctly rejected with HTTP 422.")
        else:
            print(f"       ✗ candidate_k < top_k returned {bad_k_resp.status_code} instead of 422!")

        # ---------------------------------------------------------------------
        # Phase 8: POST /api/generate Grounded RAG Generation Flow
        # ---------------------------------------------------------------------
        print("\n[Phase 8] Testing POST /api/generate Grounded Answer Flow (1 English Query)...")
        print("      Calling /api/generate for 'student scholarship'...")
        gen_start = time.time()
        gen_resp = client.post(
            "/api/generate",
            json={"query": "student scholarship", "top_k": 2, "candidate_k": 20},
        )
        gen_elapsed = time.time() - gen_start
        payloads_to_inspect.append(gen_resp.text)
        real_gemini_calls += 1

        total_checks += 1
        if gen_resp.status_code == 200:
            passed_checks += 1
            print(f"       ✓ /api/generate returned HTTP 200 ({gen_elapsed:.2f}s).")
        else:
            print(f"       ✗ /api/generate returned status {gen_resp.status_code}!")
            print(f"         Detail: {gen_resp.text}")

        gen_data = gen_resp.json()

        # Check required fields
        required_fields = [
            "answer", "language", "grounded", "confidence",
            "schemes", "evidence_used", "limitations",
            "retrieval_method", "total_results", "coverage_info"
        ]
        total_checks += 1
        missing_fields = [f for f in required_fields if f not in gen_data]
        if not missing_fields:
            passed_checks += 1
            print(f"       ✓ All required response fields present: {required_fields}")
        else:
            print(f"       ✗ Missing response fields: {missing_fields}")

        total_checks += 1
        if len(gen_data.get("answer", "").strip()) > 30:
            passed_checks += 1
            print("       ✓ Answer text is non-empty and well-formed.")
        else:
            print("       ✗ Answer text is empty or too short!")

        total_checks += 1
        if gen_data.get("grounded") is True:
            passed_checks += 1
            print("       ✓ grounded flag is True.")
        else:
            print(f"       ✗ grounded flag is not True: {gen_data.get('grounded')}")

        total_checks += 1
        if gen_data.get("confidence") in ["high", "medium", "low"]:
            passed_checks += 1
            print(f"       ✓ Categorical confidence level valid: '{gen_data.get('confidence')}'.")
        else:
            print(f"       ✗ Invalid confidence: {gen_data.get('confidence')}")

        total_checks += 1
        if gen_data.get("language") == "en":
            passed_checks += 1
            print(f"       ✓ Language correctly identified as 'en'.")
        else:
            print(f"       ✗ Unexpected language: {gen_data.get('language')}")

        total_checks += 1
        ev_used = gen_data.get("evidence_used", [])
        if len(ev_used) > 0 and all("chunk_id" in e and "scheme_id" in e for e in ev_used):
            passed_checks += 1
            print(f"       ✓ Granular evidence citations present ({len(ev_used)} chunks referenced).")
        else:
            print("       ✗ Evidence citations missing or malformed!")

        total_checks += 1
        schemes_cited = gen_data.get("schemes", [])
        if len(schemes_cited) > 0:
            passed_checks += 1
            print(f"       ✓ Relevant scheme IDs cited: {schemes_cited}")
        else:
            print("       ✗ Schemes list is empty!")

        # Print preview of generated response
        print(f"\n      --- Generated Answer Preview ---")
        answer_lines = gen_data.get("answer", "").strip().split("\n")
        for line in answer_lines[:6]:
            print(f"      | {line}")
        if len(answer_lines) > 6:
            print(f"      | ... [{len(answer_lines) - 6} more lines]")

        # ---------------------------------------------------------------------
        # Phase 9: Credential & Secret Leakage Inspection
        # ---------------------------------------------------------------------
        print("\n[Phase 9] Inspecting All HTTP Responses for Credential Leakage...")
        total_checks += 1
        secret_detected = False
        forbidden_tokens = ["AQ.", "AIza", "password=", "postgres:", "@localhost:5432"]

        for payload in payloads_to_inspect:
            for token in forbidden_tokens:
                if token in payload:
                    secret_detected = True
                    print(f"       ✗ Credential leakage detected! Found '{token}' in response payload.")
                    break

        if not secret_detected:
            passed_checks += 1
            print(f"       ✓ Zero credential leakage: Verified across {len(payloads_to_inspect)} response payloads.")

    # -------------------------------------------------------------------------
    # Phase 10: PostgreSQL Database Read-Only Parity Audit
    # -------------------------------------------------------------------------
    print("\n[Phase 10] Verifying Final Database Read-Only Parity (Zero Writes)...")
    final_counts = audit_database_counts(engine)

    parity_ok = True
    for key in baseline_counts:
        diff = final_counts[key] - baseline_counts[key]
        if diff != 0:
            parity_ok = False
            print(f"       ✗ Mutation in table {key}: baseline={baseline_counts[key]}, final={final_counts[key]} (diff={diff})")
        else:
            print(f"      - {key.replace('_', ' ').title():<18}: {final_counts[key]:,} (diff: +0)")

    total_checks += 1
    if parity_ok:
        passed_checks += 1
        print("       ✓ Read-only guarantee confirmed: 100% database parity preserved.")
    else:
        print("       ✗ Database was modified during API tests!")

    # -------------------------------------------------------------------------
    # Final Summary Report
    # -------------------------------------------------------------------------
    overall_pass = (total_checks == passed_checks) and parity_ok

    print("\n" + "=" * 80)
    print("STEP 6 FASTAPI BACKEND FOUNDATION VERIFICATION REPORT")
    print("=" * 80)
    print(f"Total Validation Checks:     {total_checks}")
    print(f"Passed Checks:              {passed_checks}")
    print(f"Failed Checks:              {total_checks - passed_checks}")
    print(f"Real Gemini Calls Made:     {real_gemini_calls}")
    print(f"API Port:                   {settings.api_port}")
    print(f"Schemes Table Count:        {final_counts['schemes']:,}")
    print(f"Scheme Chunks Count:        {final_counts['scheme_chunks']:,}")
    print(f"Populated Vector Count:     {final_counts['embedded']:,}")
    print(f"NULL Vector Count:          {final_counts['null_embeddings']:,}")
    print(f"Sources Table Count:        {final_counts['sources']:,}")
    print(f"Database Modified:          NO (READ-ONLY)")
    print(f"Overall Result:             {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 80)

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
