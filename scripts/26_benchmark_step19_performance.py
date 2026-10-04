"""
scripts/26_benchmark_step19_performance.py
==========================================
Step 19: Post-Optimization Retrieval Performance Benchmark.

Benchmarks the optimized production retrieval path after:
1. Persisted search_vector tsvector column with GIN index on scheme_chunks.
2. HNSW cosine index on scheme_chunks.embedding_local.
3. Concurrent keyword and vector execution in HybridRetriever.

Evaluates the exact same 20 representative queries from Step 18:
- 10 English queries
- 5 Hindi queries
- 5 Gujarati queries

Outputs:
- Detailed latency profile (avg, median, p95, min, max)
- Comparison against Step 18 baseline
- Percentage improvements across all stages
- Saved to reports/retrieval_perf_step19.json
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.retrieval import (
    VectorRetriever,
    KeywordRetriever,
    HybridRetriever,
    rerank_schemes,
    generate_query_embedding,
    get_embedding_model,
)
from src.retrieval.retrieval_service import RetrievalService
from src.retrieval.context_builder import EvidenceContextBuilder


SAMPLE_QUERIES = [
    ("EN_01", "EN", "What scholarships are available for disabled students?"),
    ("EN_02", "EN", "PM Kisan Samman Nidhi farmer subsidy"),
    ("EN_03", "EN", "Ayushman Bharat health insurance eligibility"),
    ("EN_04", "EN", "Housing assistance for poor rural families PMAY"),
    ("EN_05", "EN", "Mudra loan scheme for women entrepreneurs"),
    ("EN_06", "EN", "Old age pension scheme eligibility criteria"),
    ("EN_07", "EN", "Sanitation workers children scholarship Chhattisgarh"),
    ("EN_08", "EN", "AICTE Pragati scholarship for girls degree"),
    ("EN_09", "EN", "Solar water pump subsidy for farmers PM Kusum"),
    ("EN_10", "EN", "Self employment tool kit grant for artisans"),
    ("HI_01", "HI", "आयुष्मान भारत योजना के तहत मुफ्त इलाज पात्रता क्या है?"),
    ("HI_02", "HI", "किसान क्रेडिट कार्ड कैसे बनवाएं और इसके क्या फायदे हैं?"),
    ("HI_03", "HI", "प्रधानमंत्री मातृ वंदना योजना में गर्भवती महिलाओं को कितनी राशि मिलती है?"),
    ("HI_04", "HI", "पीएम आवास योजना शहरी ऑनलाइन आवेदन प्रक्रिया और पात्रता"),
    ("HI_05", "HI", "किसानों के लिए सोलर पंप कुसुम योजना सब्सिडी"),
    ("GU_01", "GU", "મુખ્યમંત્રી અમૃતમ મા યોજના હેઠળ કેશલેસ સારવારની પાત્રતા"),
    ("GU_06", "GU", "ગુજરાતના ખેડૂતો માટે સનેડો કૃષિ સાધન ખરીદી પર સહાય"),
    ("GU_07", "GU", "ગુજરાત અનુસૂચિત જાતિ ખેડૂતો માટે જમીન ખરીદી સહાય યોજના"),
    ("GU_08", "GU", "ગુજરાત સરદાર પટેલ આવાસ યોજના ગ્રામીણ મકાન સહાય"),
    ("GU_09", "GU", "બાંધકામ શ્રમિકો માટે શ્રમિક અન્નપૂર્ણા યોજના 5 રૂપિયામાં ભોજન"),
]


def main():
    print("=" * 80)
    print("  SUGAMGOV RAG — STEP 19: PERFORMANCE BENCHMARK (AFTER OPTIMIZATIONS)")
    print("=" * 80)

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("[ERROR] DATABASE_URL not set in .env")
        sys.exit(1)

    engine = create_engine(db_url)

    # Pre-warm model and connections
    print("\n[1/3] Loading Local Embedding Model & Initializing Components...")
    t0 = time.time()
    model = get_embedding_model()
    print(f"      Model loaded in {time.time() - t0:.2f}s")

    kw_retriever = KeywordRetriever(engine)
    vec_retriever = VectorRetriever(engine, model=model)
    hybrid_retriever = HybridRetriever(
        engine_or_url=engine,
        keyword_retriever=kw_retriever,
        vector_retriever=vec_retriever,
        rrf_k=60,
        default_candidate_k=40,
    )
    service = RetrievalService(
        engine_or_url=engine,
        hybrid_retriever=hybrid_retriever,
        default_candidate_k=40,
    )

    # Warmup query
    print("      Executing warm-up query...")
    _ = service.retrieve("warmup query", top_k=5)
    print("      ✓ Warm-up complete.")

    # 2. Benchmark Loop
    print("\n[2/3] Benchmarking 20 Representative Queries...")
    timers: Dict[str, List[float]] = {
        "t_query_embed": [],
        "t_vec_sql": [],
        "t_vec_total": [],
        "t_kw_sql": [],
        "t_kw_total": [],
        "t_hybrid_concurrent": [],
        "t_scheme_rerank": [],
        "t_prod_retrieve_e2e": [],
    }

    for idx, (qid, lang, qtext) in enumerate(SAMPLE_QUERIES, 1):
        clean_q = qtext.strip()
        top_k = 10
        candidate_k = 40

        # A. Query Embedding
        t0 = time.perf_counter()
        q_vec = generate_query_embedding(clean_q, model=model)
        t_embed = (time.perf_counter() - t0) * 1000

        # B. Vector SQL & Total Vector Leg
        vec_sql = text("""
            SELECT 
                c.chunk_id,
                c.scheme_id,
                s.scheme_name,
                c.field_name,
                c.chunk_text,
                c.metadata,
                s.state,
                s.level,
                s.categories,
                1 - (c.embedding_local <=> CAST(:qv AS vector)) AS similarity_score
            FROM scheme_chunks c
            JOIN schemes s ON c.scheme_id = s.scheme_id
            WHERE c.embedding_local IS NOT NULL
            ORDER BY c.embedding_local <=> CAST(:qv AS vector) ASC
            LIMIT :top_k;
        """)
        vec_str = "[" + ",".join(f"{v:.8f}" for v in q_vec) + "]"
        t0 = time.perf_counter()
        with engine.connect() as conn:
            _ = conn.execute(vec_sql, {"qv": vec_str, "top_k": candidate_k}).fetchall()
        t_vec_sql = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        vec_chunks = vec_retriever.retrieve_by_vector(query_vector=q_vec, top_k=candidate_k)
        t_vec_total = (time.perf_counter() - t0) * 1000

        # C. Keyword SQL & Total Keyword Leg
        kw_sql = text("""
            WITH filtered_chunks AS (
                SELECT 
                    c.id, c.chunk_id, c.scheme_id, s.scheme_name, c.field_name,
                    c.chunk_text, c.metadata, s.state, s.level, s.categories,
                    c.search_vector
                FROM scheme_chunks c
                JOIN schemes s ON c.scheme_id = s.scheme_id
            )
            SELECT 
                f.chunk_id, f.scheme_id, f.scheme_name, f.field_name,
                f.chunk_text, f.metadata, f.state, f.level, f.categories,
                ts_rank_cd(f.search_vector, q.query) AS score
            FROM filtered_chunks f,
            websearch_to_tsquery('english', :query) q(query)
            WHERE f.search_vector @@ q.query
            ORDER BY score DESC, f.id ASC
            LIMIT :top_k;
        """)
        t0 = time.perf_counter()
        with engine.connect() as conn:
            _ = conn.execute(kw_sql, {"query": clean_q, "top_k": candidate_k}).fetchall()
        t_kw_sql = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        kw_chunks = kw_retriever.retrieve(query=clean_q, top_k=candidate_k)
        t_kw_total = (time.perf_counter() - t0) * 1000

        # D. Hybrid Retrieval (Concurrent)
        t0 = time.perf_counter()
        hyb_chunks = hybrid_retriever.retrieve(
            query=clean_q,
            query_vector=q_vec,
            top_k=candidate_k,
            candidate_k=candidate_k,
            rrf_k=60,
        )
        t_hybrid = (time.perf_counter() - t0) * 1000

        # E. Scheme Reranking
        t0 = time.perf_counter()
        reranked = rerank_schemes(hyb_chunks, top_k=top_k)
        t_rerank = (time.perf_counter() - t0) * 1000

        # F. Complete RetrievalService.retrieve()
        t0 = time.perf_counter()
        resp = service.retrieve(query=clean_q, top_k=top_k, candidate_k=candidate_k)
        t_service_e2e = (time.perf_counter() - t0) * 1000

        timers["t_query_embed"].append(t_embed)
        timers["t_vec_sql"].append(t_vec_sql)
        timers["t_vec_total"].append(t_vec_total)
        timers["t_kw_sql"].append(t_kw_sql)
        timers["t_kw_total"].append(t_kw_total)
        timers["t_hybrid_concurrent"].append(t_hybrid)
        timers["t_scheme_rerank"].append(t_rerank)
        timers["t_prod_retrieve_e2e"].append(t_service_e2e)

        print(f"  [{idx:2d}/20] {qid} ({lang}): KW={t_kw_total:5.1f}ms | Vec={t_vec_total:5.1f}ms | Hyb={t_hybrid:5.1f}ms | E2E={t_service_e2e:5.1f}ms ({len(resp.results)} schemes)")

    # 3. Compute Summary Statistics
    stats: Dict[str, Dict[str, float]] = {}
    for k, v in timers.items():
        arr = np.array(v)
        stats[k] = {
            "avg": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    # Load Step 18 Baseline for Comparison
    step18_json_path = PROJECT_ROOT / "reports" / "retrieval_diagnostic.json"
    step18_stats = {}
    if step18_json_path.exists():
        with open(step18_json_path, "r", encoding="utf-8") as f:
            step18_data = json.load(f)
            step18_stats = step18_data.get("timing_stats_ms", {})

    print("\n" + "=" * 80)
    print("  PERFORMANCE COMPARISON: STEP 18 BASELINE vs STEP 19 OPTIMIZED")
    print("=" * 80)
    print(f"{'Retrieval Stage':<28} | {'Step 18 Avg':<12} | {'Step 19 Avg':<12} | {'Improvement':<12}")
    print("-" * 72)

    comparisons = [
        ("Query Embedding", step18_stats.get("t_query_embed", {}).get("avg", 22.21), stats["t_query_embed"]["avg"]),
        ("Vector SQL (pgvector)", step18_stats.get("t_vec_sql", {}).get("avg", 130.02), stats["t_vec_sql"]["avg"]),
        ("Keyword SQL (Postgres FTS)", step18_stats.get("t_kw_sql", {}).get("avg", 1904.83), stats["t_kw_sql"]["avg"]),
        ("Hybrid Retrieval Leg", step18_stats.get("t_kw_sql", {}).get("avg", 1904.83) + 130.0, stats["t_hybrid_concurrent"]["avg"]),
        ("Scheme Reranking", step18_stats.get("t_scheme_rerank", {}).get("avg", 0.20), stats["t_scheme_rerank"]["avg"]),
        ("RetrievalService E2E", step18_stats.get("t_total_prod_retrieve", {}).get("avg", 2713.52), stats["t_prod_retrieve_e2e"]["avg"]),
    ]

    comp_results = []
    for stage, b_avg, o_avg in comparisons:
        pct_imp = ((b_avg - o_avg) / b_avg) * 100 if b_avg > 0 else 0.0
        speedup = b_avg / o_avg if o_avg > 0 else 1.0
        print(f"{stage:<28} | {b_avg:9.2f} ms | {o_avg:9.2f} ms | +{pct_imp:5.1f}% ({speedup:.1f}x)")
        comp_results.append({
            "stage": stage,
            "baseline_avg_ms": b_avg,
            "optimized_avg_ms": o_avg,
            "percentage_improvement": pct_imp,
            "speedup_factor": speedup,
        })

    # Save to JSON
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "sample_queries_count": len(SAMPLE_QUERIES),
        "timing_stats_ms": stats,
        "comparisons": comp_results,
    }
    out_path = PROJECT_ROOT / "reports" / "retrieval_perf_step19.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  ✓ Saved benchmark results to: {out_path}")


if __name__ == "__main__":
    main()
