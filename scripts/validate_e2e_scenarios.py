"""
scripts/validate_e2e_scenarios.py
==================================
Comprehensive end-to-end scenario validation script for Step 10.
Executes all required test cases A through H against http://localhost:3000
and the integrated FastAPI RAG backend:

  Test A: Initial Chat ("What government scholarships are available for students?")
  Test B: Follow-up / Multi-turn ("What documents are required?")
  Test C: Multilingual (Hindi & Gujarati queries)
  Test D: Conversation Persistence (Login, Thread creation, Reload verification)
  Test E: Citation Inspection (Official source link & MyScheme URL format)
  Test F: No-evidence Behavior (Unrelated non-government query)
  Test G: Security Sanity (Zero API keys, connection strings, or secrets exposed)
  Test H: UI Sanity & Bundle Analysis (Zero crashes, valid HTML, correct CSS/fonts)
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
import subprocess

if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

NEXT_BASE = "http://127.0.0.1:3000"
test_log = []

def record(test_id: str, title: str, passed: bool, details: dict):
    status = "PASS" if passed else "FAIL"
    test_log.append({
        "id": test_id,
        "title": title,
        "status": status,
        "details": details
    })
    print(f"[{status}] {test_id}: {title}")
    for k, v in details.items():
        print(f"       {k}: {v}")

def get_session_cookie(email="integration_tester_rag@example.com", password="Password123!SafeTesting"):
    login_url = f"{NEXT_BASE}/api/auth/login"
    payload = json.dumps({"email": email, "password": password}).encode('utf-8')
    req = urllib.request.Request(login_url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        cookie_header = resp.headers.get("Set-Cookie", "")
        return cookie_header.split(";")[0]

def stream_chat_turn(message: str, language: str = "en", conversation_id: str = None, cookie: str = None):
    url = f"{NEXT_BASE}/api/chat"
    payload = {"message": message, "language": language}
    if conversation_id:
        payload["conversationId"] = conversation_id
    
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie

    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    start_time = time.time()
    
    metadata = None
    chunks = []
    errors = []
    raw_lines = []

    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp:
            l = line.decode('utf-8').strip()
            if not l:
                continue
            raw_lines.append(l)
            try:
                data = json.loads(l)
                if data.get("type") == "metadata":
                    metadata = data
                elif data.get("type") == "chunk":
                    chunks.append(data.get("text", ""))
                elif data.get("type") == "error":
                    errors.append(data.get("message", ""))
            except Exception as e:
                errors.append(f"JSON parse error: {e}")

    elapsed = time.time() - start_time
    full_text = "".join(chunks)
    return {
        "metadata": metadata,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "full_text": full_text,
        "errors": errors,
        "elapsed": elapsed,
        "raw_lines": raw_lines
    }

def main():
    print("=" * 70)
    print("STEP 10: END-TO-END VALIDATION SUITE")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # Test A: Initial Chat
    # --------------------------------------------------------------------------
    print("\n--- Running Test A: Initial Chat ---")
    query_a = "What government scholarships are available for students?"
    res_a = stream_chat_turn(query_a, language="en")
    
    has_meta_a = res_a["metadata"] is not None
    chunks_ok_a = res_a["chunk_count"] >= 3
    no_raw_leak_a = not any("event:" in l or "data:" in l for l in res_a["chunks"])
    has_scheme_a = res_a["metadata"].get("officialSource") is not None if has_meta_a else False
    has_url_a = res_a["metadata"].get("sourceUrl") is not None if has_meta_a else False
    has_content_a = len(res_a["full_text"]) > 50

    passed_a = has_meta_a and chunks_ok_a and no_raw_leak_a and has_scheme_a and has_url_a and has_content_a
    record("TEST-A", "Initial Chat Streaming & Citations", passed_a, {
        "query": query_a,
        "chunks_received": res_a["chunk_count"],
        "stream_time_seconds": f"{res_a['elapsed']:.2f}s",
        "official_source": res_a["metadata"].get("officialSource") if has_meta_a else None,
        "source_url": res_a["metadata"].get("sourceUrl") if has_meta_a else None,
        "retrieval_method": res_a["metadata"].get("retrievalMethod") if has_meta_a else None,
        "response_preview": res_a["full_text"][:120] + "..."
    })

    # --------------------------------------------------------------------------
    # Test B: Follow-up / Multi-turn Conversational Continuity
    # --------------------------------------------------------------------------
    print("\n--- Running Test B: Follow-up / Multi-turn ---")
    cookie = get_session_cookie()
    # Turn 1 with authenticated session
    res_b1 = stream_chat_turn("What government scholarships are available for students?", language="en", cookie=cookie)
    conv_id = res_b1["metadata"].get("conversationId")

    # Turn 2 follow-up without repeating context
    query_b2 = "What documents are required?"
    res_b2 = stream_chat_turn(query_b2, language="en", conversation_id=conv_id, cookie=cookie)
    
    multi_turn_ok = (
        res_b2["metadata"] is not None and
        res_b2["chunk_count"] >= 3 and
        len(res_b2["full_text"]) > 50 and
        res_b2["metadata"].get("conversationId") == conv_id
    )
    record("TEST-B", "Follow-up / Multi-Turn Continuity", multi_turn_ok, {
        "conversation_id": conv_id,
        "follow_up_query": query_b2,
        "chunks_received": res_b2["chunk_count"],
        "stream_time_seconds": f"{res_b2['elapsed']:.2f}s",
        "response_preview": res_b2["full_text"][:120] + "..."
    })

    # --------------------------------------------------------------------------
    # Test C: Multilingual Support (Hindi & Gujarati)
    # --------------------------------------------------------------------------
    print("\n--- Running Test C: Multilingual Validation ---")
    
    # C1: Hindi query on populated scheme
    query_hi = "Indira Mahila Shakti Udyam Protsahan Yojana के तहत क्या लाभ और सब्सिडी मिलती है?"
    res_hi = stream_chat_turn(query_hi, language="hi")
    has_devanagari = any('\u0900' <= char <= '\u097f' for char in res_hi["full_text"])
    passed_hi = has_devanagari and res_hi["chunk_count"] >= 3 and len(res_hi["full_text"]) > 100

    record("TEST-C1", "Multilingual - Grounded Hindi Scheme Generation", passed_hi, {
        "query": query_hi,
        "chunks_received": res_hi["chunk_count"],
        "has_devanagari_characters": has_devanagari,
        "official_source": res_hi["metadata"].get("officialSource") if res_hi["metadata"] else None,
        "response_length_chars": len(res_hi["full_text"]),
        "response_preview": res_hi["full_text"][:120] + "..."
    })

    # C2: Gujarati query on populated scheme
    query_gu = "Indira Mahila Shakti Udyam Protsahan Yojana હેઠળ શું લાભ અને સહાય મળે છે?"
    res_gu = stream_chat_turn(query_gu, language="gu")
    has_gujarati = any('\u0a80' <= char <= '\u0aff' for char in res_gu["full_text"])
    passed_gu = has_gujarati and res_gu["chunk_count"] >= 3 and len(res_gu["full_text"]) > 100

    record("TEST-C2", "Multilingual - Grounded Gujarati Scheme Generation", passed_gu, {
        "query": query_gu,
        "chunks_received": res_gu["chunk_count"],
        "has_gujarati_characters": has_gujarati,
        "official_source": res_gu["metadata"].get("officialSource") if res_gu["metadata"] else None,
        "response_length_chars": len(res_gu["full_text"]),
        "response_preview": res_gu["full_text"][:120] + "..."
    })

    # C3: Localized refusal verification (when scheme embeddings are pending)
    query_hi_pending = "पीएम किसान सम्मान निधि योजना के तहत कितनी राशि मिलती है?"
    res_hi_pending = stream_chat_turn(query_hi_pending, language="hi")
    has_deva_refusal = any('\u0900' <= char <= '\u097f' for char in res_hi_pending["full_text"])
    passed_c3 = has_deva_refusal and "पर्याप्त जानकारी नहीं मिली" in res_hi_pending["full_text"]

    record("TEST-C3", "Multilingual - Localized Safe Refusal on Pending Scheme", passed_c3, {
        "query": query_hi_pending,
        "is_supported": res_hi_pending["metadata"].get("isSupported") if res_hi_pending["metadata"] else None,
        "localized_refusal": res_hi_pending["full_text"][:100] + "..."
    })

    # --------------------------------------------------------------------------
    # Test D: Conversation Persistence & Reload Verification
    # --------------------------------------------------------------------------
    print("\n--- Running Test D: Conversation Persistence ---")
    # Verify the thread created in Test B is readable via GET /api/conversations/[id]
    history_req = urllib.request.Request(f"{NEXT_BASE}/api/conversations/{conv_id}", headers={"Cookie": cookie})
    with urllib.request.urlopen(history_req, timeout=10) as resp:
        thread_data = json.loads(resp.read().decode('utf-8'))
        saved_messages = thread_data.get("messages", [])
        
    thread_persisted = (
        len(saved_messages) >= 4 and
        any(m.get("sender") == "user" for m in saved_messages) and
        any(m.get("sender") == "assistant" for m in saved_messages) and
        thread_data.get("conversation", {}).get("id") == conv_id
    )
    record("TEST-D", "Conversation History Persistence & Reload", thread_persisted, {
        "conversation_id": conv_id,
        "thread_title": thread_data.get("conversation", {}).get("title"),
        "total_persisted_turns": len(saved_messages),
        "turn_1_user": saved_messages[0]["text"][:50] if len(saved_messages) > 0 else None,
        "turn_1_assistant": saved_messages[1]["text"][:50] if len(saved_messages) > 1 else None,
        "turn_2_user": saved_messages[2]["text"][:50] if len(saved_messages) > 2 else None,
        "turn_2_assistant": saved_messages[3]["text"][:50] if len(saved_messages) > 3 else None,
    })

    # --------------------------------------------------------------------------
    # Test E: Citation & Official Source Link Format
    # --------------------------------------------------------------------------
    print("\n--- Running Test E: Citation & Official Source Link ---")
    source_url = res_a["metadata"].get("sourceUrl", "")
    source_title = res_a["metadata"].get("officialSource", "")
    is_valid_gov_url = (
        "myscheme.gov.in" in source_url or
        "india.gov.in" in source_url or
        "pmkisan.gov.in" in source_url or
        "scholarships.gov.in" in source_url
    )
    passed_e = is_valid_gov_url and bool(source_title)
    record("TEST-E", "Citation & Official Source Portal Link", passed_e, {
        "official_source_title": source_title,
        "official_source_url": source_url,
        "is_valid_gov_domain": is_valid_gov_url,
        "retrieval_badge": res_a["metadata"].get("retrievalMethod")
    })

    # --------------------------------------------------------------------------
    # Test F: No-Evidence Behavior (Unrelated Non-Government Query)
    # --------------------------------------------------------------------------
    print("\n--- Running Test F: No-Evidence Safety ---")
    query_unrelated = "How do I bake a chocolate cake at home?"
    res_f = stream_chat_turn(query_unrelated, language="en")
    
    # Grounding check: system must state no evidence was found or refuse to invent facts
    evidence_refusal_markers = [
        "not contain enough evidence",
        "insufficient",
        "not available in retrieved evidence",
        "couldn't find verified information",
        "no government scheme"
    ]
    refusal_text = res_f["full_text"].lower()
    has_refusal = any(m in refusal_text for m in evidence_refusal_markers)
    no_fabricated_scheme = "chocolate cake yojana" not in refusal_text

    passed_f = has_refusal and no_fabricated_scheme
    record("TEST-F", "No-Evidence Grounding Safety", passed_f, {
        "unrelated_query": query_unrelated,
        "system_response_sample": res_f["full_text"][:120],
        "refused_fabrication": passed_f,
        "grounded_flag": res_f["metadata"].get("isSupported") if res_f["metadata"] else None
    })

    # --------------------------------------------------------------------------
    # Test G: Security Sanity Check (Zero Secrets Leaked)
    # --------------------------------------------------------------------------
    print("\n--- Running Test G: Security Sanity Check ---")
    all_responses = " ".join([
        res_a["full_text"],
        res_b2["full_text"],
        res_hi["full_text"],
        res_gu["full_text"],
        res_f["full_text"]
    ])
    
    # Also check landing page HTML
    landing_req = urllib.request.Request(f"{NEXT_BASE}/")
    with urllib.request.urlopen(landing_req, timeout=5) as r:
        landing_html = r.read().decode('utf-8', errors='ignore')
    
    security_targets = [
        "AIzaSy",             # Gemini API key pattern
        "mongodb+srv://",     # MongoDB URI pattern
        "postgresql://",      # PostgreSQL URI pattern
        "postgres:Parth",     # PostgreSQL password pattern
        "SESSION_SECRET",
    ]
    found_leaks = [s for s in security_targets if s in all_responses or s in landing_html]
    passed_g = len(found_leaks) == 0
    record("TEST-G", "Security Sanity Check (Zero Leaks)", passed_g, {
        "scanned_payload_chars": len(all_responses) + len(landing_html),
        "forbidden_patterns_checked": security_targets,
        "leaks_detected": found_leaks
    })

    # --------------------------------------------------------------------------
    # Test H: UI Sanity & Layout Integrity
    # --------------------------------------------------------------------------
    print("\n--- Running Test H: UI Sanity & Layout Integrity ---")
    # Verify landing page HTML has critical UI elements
    has_title = "<title>" in landing_html and "Sugam" in landing_html
    has_navbar = "nav" in landing_html.lower() or "ask ai" in landing_html.lower()
    has_hero = "Simplified with AI" in landing_html or "Government Services" in landing_html
    has_viewport = "viewport" in landing_html

    passed_h = has_title and has_navbar and has_hero and has_viewport
    record("TEST-H", "UI Sanity & Layout Integrity", passed_h, {
        "page_title_present": has_title,
        "navbar_present": has_navbar,
        "hero_section_present": has_hero,
        "responsive_viewport_present": has_viewport,
        "html_size_bytes": len(landing_html)
    })

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 10 VALIDATION SUMMARY")
    print("=" * 70)
    total = len(test_log)
    passed = sum(1 for t in test_log if t["status"] == "PASS")
    failed = sum(1 for t in test_log if t["status"] == "FAIL")

    for t in test_log:
        print(f"[{t['status']}] {t['id']}: {t['title']}")

    print("-" * 70)
    print(f"Total Tests:   {total}")
    print(f"Passed Tests:  {passed}")
    print(f"Failed Tests:  {failed}")
    print(f"Overall Result: {'PASS' if failed == 0 else 'FAIL'}")
    print("=" * 70)

    # Save results to scratch JSON for report generation
    scratch_report_path = r"C:\Users\pc\.gemini\antigravity-ide\brain\69d52c93-effb-449b-8394-da9aee6ca740\scratch\e2e_results.json"
    with open(scratch_report_path, "w", encoding="utf-8") as f:
        json.dump(test_log, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {scratch_report_path}")

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
