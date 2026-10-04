"""
scripts/27_evaluate_step20_final.py
===================================
Step 20: Final Post-Optimization Retrieval Evaluation.

Evaluates 86 queries:
  - 60 English benchmark queries (data/evaluation/retrieval_eval.json)
  - 12 Hindi benchmark queries
  - 14 Gujarati benchmark queries (with documented corrected ground-truth targets)

Evaluates 4 retrieval modes:
  1. Keyword FTS (GIN indexed search_vector)
  2. Local Vector Retrieval (HNSW indexed embedding_local, 384d)
  3. Hybrid RRF (Concurrent keyword + vector, k=60)
  4. Scheme-Level Reranking & Chunk Consolidation

Computes:
  - Recall@5, Recall@10, MRR@5, MRR@10
  - Rank #1 hits, total hits, misses
  - Detailed component & E2E latencies
  - Query-by-query Gujarati diagnostic breakdown
  - JSON artifact output: reports/final_retrieval_evaluation.json

Zero database modifications (strict read-only guarantee).
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.vector_retriever import VectorRetriever, generate_query_embedding
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.scheme_reranker import rerank_schemes
from src.retrieval.retrieval_service import RetrievalService

load_dotenv(PROJECT_ROOT / ".env")

CANDIDATE_CHUNKS = 40
RRF_K = 60
TOP_K = 10


def get_db_counts(conn) -> Dict[str, int]:
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


def collapse_to_schemes(chunks: List[Any]) -> List[str]:
    seen = set()
    schemes = []
    for c in chunks:
        sid = getattr(c, "scheme_id", None) or (c.get("scheme_id") if isinstance(c, dict) else None)
        if sid and sid not in seen:
            seen.add(sid)
            schemes.append(sid)
    return schemes


def evaluate_query_set(
    eval_queries: List[Dict[str, Any]],
    kw_retriever: KeywordRetriever,
    vec_retriever: VectorRetriever,
    hybrid_retriever: HybridRetriever,
    retrieval_service: RetrievalService,
    model: SentenceTransformer,
    set_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, List[float]]]:
    modalities = ["keyword", "vector", "hybrid", "scheme_reranked"]
    metrics_tracker = {
        m: {
            "recall@5": [], "recall@10": [],
            "mrr@5": [], "mrr@10": [],
            "hits": 0, "rank1_hits": 0,
            "outside_top5": 0, "outside_top10": 0, "outside_top40": 0
        }
        for m in modalities
    }

    latency_tracker = {
        "t_embed": [],
        "t_vec_sql": [],
        "t_vec_total": [],
        "t_kw_sql": [],
        "t_hybrid_concurrent": [],
        "t_scheme_rerank": [],
        "t_prod_e2e": [],
    }

    per_query_results = []
    n = len(eval_queries)

    print(f"\n--- Evaluating {set_name} ({n} queries) ---")

    for idx, item in enumerate(eval_queries, start=1):
        qid = item["query_id"]
        qtext = item["query"]
        lang = item.get("lang") or "EN"
        targets = set(item["relevant_scheme_ids"])

        # 1. Query Embedding
        t0 = time.perf_counter()
        q_vec = generate_query_embedding(qtext, model=model)
        t_embed = (time.perf_counter() - t0) * 1000.0

        # 2. Local Vector pgvector search
        t1 = time.perf_counter()
        vec_chunks = vec_retriever.retrieve_by_vector(query_vector=q_vec, top_k=CANDIDATE_CHUNKS)
        t_vec_sql = (time.perf_counter() - t1) * 1000.0
        t_vec_total = t_embed + t_vec_sql
        vec_schemes = collapse_to_schemes(vec_chunks)

        # 3. Keyword FTS Search
        t2 = time.perf_counter()
        kw_chunks = kw_retriever.retrieve(query=qtext, top_k=CANDIDATE_CHUNKS)
        t_kw_sql = (time.perf_counter() - t2) * 1000.0
        kw_schemes = collapse_to_schemes(kw_chunks)

        # 4. Concurrent Hybrid RRF Search
        t3 = time.perf_counter()
        hyb_chunks = hybrid_retriever.retrieve(
            query=qtext,
            query_vector=q_vec,
            top_k=CANDIDATE_CHUNKS,
            candidate_k=CANDIDATE_CHUNKS,
            rrf_k=RRF_K,
        )
        t_hyb_concurrent = (time.perf_counter() - t3) * 1000.0
        hyb_schemes = collapse_to_schemes(hyb_chunks)

        # 5. Scheme Consolidation & Reranking
        t4 = time.perf_counter()
        reranked_results = rerank_schemes(hyb_chunks, top_k=CANDIDATE_CHUNKS)
        t_scheme_rerank = (time.perf_counter() - t4) * 1000.0
        reranked_schemes = [s.scheme_id for s in reranked_results]

        # 6. Full Production Service Call (True End-to-End)
        t5 = time.perf_counter()
        prod_res = retrieval_service.retrieve(query=qtext, top_k=TOP_K, candidate_k=CANDIDATE_CHUNKS)
        t_prod_e2e = (time.perf_counter() - t5) * 1000.0

        latency_tracker["t_embed"].append(t_embed)
        latency_tracker["t_vec_sql"].append(t_vec_sql)
        latency_tracker["t_vec_total"].append(t_vec_total)
        latency_tracker["t_kw_sql"].append(t_kw_sql)
        latency_tracker["t_hybrid_concurrent"].append(t_hyb_concurrent)
        latency_tracker["t_scheme_rerank"].append(t_scheme_rerank)
        latency_tracker["t_prod_e2e"].append(t_prod_e2e)

        query_entry = {
            "query_id": qid,
            "query": qtext,
            "lang": lang,
            "relevant_scheme_ids": list(targets),
            "latencies_ms": {
                "t_embed": t_embed,
                "t_vec_sql": t_vec_sql,
                "t_vec_total": t_vec_total,
                "t_kw_sql": t_kw_sql,
                "t_hybrid_concurrent": t_hyb_concurrent,
                "t_scheme_rerank": t_scheme_rerank,
                "t_prod_e2e": t_prod_e2e,
            },
        }

        mode_pairs = [
            ("keyword", kw_schemes),
            ("vector", vec_schemes),
            ("hybrid", hyb_schemes),
            ("scheme_reranked", reranked_schemes),
        ]

        for mod_name, scheme_list in mode_pairs:
            first_rank = None
            for r_idx, sid in enumerate(scheme_list, start=1):
                if sid in targets:
                    first_rank = r_idx
                    break

            r5 = 1.0 if (first_rank is not None and first_rank <= 5) else 0.0
            r10 = 1.0 if (first_rank is not None and first_rank <= 10) else 0.0
            mrr5 = (1.0 / first_rank) if (first_rank is not None and first_rank <= 5) else 0.0
            mrr10 = (1.0 / first_rank) if (first_rank is not None and first_rank <= 10) else 0.0

            metrics_tracker[mod_name]["recall@5"].append(r5)
            metrics_tracker[mod_name]["recall@10"].append(r10)
            metrics_tracker[mod_name]["mrr@5"].append(mrr5)
            metrics_tracker[mod_name]["mrr@10"].append(mrr10)

            if first_rank is not None:
                metrics_tracker[mod_name]["hits"] += 1
                if first_rank == 1:
                    metrics_tracker[mod_name]["rank1_hits"] += 1
                if first_rank > 5:
                    metrics_tracker[mod_name]["outside_top5"] += 1
                if first_rank > 10:
                    metrics_tracker[mod_name]["outside_top10"] += 1
            else:
                metrics_tracker[mod_name]["outside_top5"] += 1
                metrics_tracker[mod_name]["outside_top10"] += 1
                metrics_tracker[mod_name]["outside_top40"] += 1

            query_entry[mod_name] = {
                "top_5": scheme_list[:5],
                "top_10": scheme_list[:10],
                "first_rank": first_rank,
            }

        per_query_results.append(query_entry)

        status_str = f"Rerank Rank: #{query_entry['scheme_reranked']['first_rank']}" if query_entry['scheme_reranked']['first_rank'] else "MISS"
        print(f"  [{idx:2d}/{n:2d}] {qid} ({lang}): {status_str:<16} | Prod E2E: {t_prod_e2e:5.1f}ms")

    summary = {}
    for mod in metrics_tracker:
        q_count = len(metrics_tracker[mod]["recall@5"])
        summary[mod] = {
            "query_count": q_count,
            "Recall@5": sum(metrics_tracker[mod]["recall@5"]) / q_count if q_count > 0 else 0.0,
            "Recall@10": sum(metrics_tracker[mod]["recall@10"]) / q_count if q_count > 0 else 0.0,
            "MRR@5": sum(metrics_tracker[mod]["mrr@5"]) / q_count if q_count > 0 else 0.0,
            "MRR@10": sum(metrics_tracker[mod]["mrr@10"]) / q_count if q_count > 0 else 0.0,
            "Hits": metrics_tracker[mod]["hits"],
            "Rank1_Hits": metrics_tracker[mod]["rank1_hits"],
            "Outside_Top5": metrics_tracker[mod]["outside_top5"],
            "Outside_Top10": metrics_tracker[mod]["outside_top10"],
            "Outside_Top40": metrics_tracker[mod]["outside_top40"],
        }

    return summary, per_query_results, latency_tracker


def compute_timing_stats(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"avg": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(latencies)
    n = len(s)
    avg = sum(s) / n
    med = s[n // 2] if n % 2 != 0 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    p95_idx = int(n * 0.95)
    p95 = s[min(p95_idx, n - 1)]
    return {
        "avg": avg,
        "median": med,
        "p95": p95,
        "min": s[0],
        "max": s[-1],
    }


def main():
    print("=" * 80)
    print("  SUGAMGOV RAG — STEP 20: FINAL RETRIEVAL EVALUATION")
    print("=" * 80)

    engine = create_engine(os.getenv("DATABASE_URL"))

    with engine.connect() as conn:
        print("\n[1/5] Pre-Evaluation Database Integrity Audit...")
        pre = get_db_counts(conn)
        print(f"  Schemes:              {pre['schemes']:,} (expected 3,397)")
        print(f"  Chunks:               {pre['total_chunks']:,} (expected 20,497)")
        print(f"  Local Populated:      {pre['loc_pop']:,} (expected 20,497)")
        print(f"  Local NULL:           {pre['loc_null']:,} (expected 0)")
        print(f"  Gemini Populated:     {pre['gem_pop']:,} (expected 1,260)")
        print(f"  Gemini NULL:          {pre['gem_null']:,} (expected 19,237)")

    print("\n[2/5] Initializing Retrieval Subsystem & Models...")
    t_load = time.perf_counter()
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    print(f"  Model loaded in {(time.perf_counter() - t_load):.2f}s")

    kw_retriever = KeywordRetriever(engine)
    vec_retriever = VectorRetriever(engine, model=model)
    hybrid_retriever = HybridRetriever(
        engine,
        keyword_retriever=kw_retriever,
        vector_retriever=vec_retriever,
        rrf_k=RRF_K,
        default_candidate_k=CANDIDATE_CHUNKS,
    )
    retrieval_service = RetrievalService(hybrid_retriever=hybrid_retriever)

    # Load datasets
    with open(PROJECT_ROOT / "data" / "evaluation" / "retrieval_eval.json", "r", encoding="utf-8") as f:
        en_queries = json.load(f)

    with open(PROJECT_ROOT / "data" / "evaluation" / "multilingual_eval_corrected.json", "r", encoding="utf-8") as f:
        multi_queries_corrected = json.load(f)

    hi_queries_corrected = [q for q in multi_queries_corrected if q["lang"] == "HI"]
    gu_queries_corrected = [q for q in multi_queries_corrected if q["lang"] == "GU"]

    print(f"\n[3/5] Datasets Loaded:")
    print(f"  English Queries:    {len(en_queries)}")
    print(f"  Hindi Queries:      {len(hi_queries_corrected)}")
    print(f"  Gujarati Queries:   {len(gu_queries_corrected)}")
    print(f"  Total Queries:      {len(en_queries) + len(multi_queries_corrected)}")

    # Execute English evaluation
    sum_en, res_en, lat_en = evaluate_query_set(
        en_queries, kw_retriever, vec_retriever, hybrid_retriever, retrieval_service, model, "English Benchmark"
    )

    # Execute Hindi evaluation
    sum_hi, res_hi, lat_hi = evaluate_query_set(
        hi_queries_corrected, kw_retriever, vec_retriever, hybrid_retriever, retrieval_service, model, "Hindi Benchmark (Corrected)"
    )

    # Execute Gujarati evaluation
    sum_gu, res_gu, lat_gu = evaluate_query_set(
        gu_queries_corrected, kw_retriever, vec_retriever, hybrid_retriever, retrieval_service, model, "Gujarati Benchmark (Corrected)"
    )

    # Overall aggregation
    all_queries_res = res_en + res_hi + res_gu
    all_latencies = {
        k: lat_en[k] + lat_hi[k] + lat_gu[k] for k in lat_en
    }

    # Overall metrics calculation
    overall_summary = {}
    for mod in ["keyword", "vector", "hybrid", "scheme_reranked"]:
        r5_all = []
        r10_all = []
        mrr5_all = []
        mrr10_all = []
        hits_all = 0
        rank1_all = 0
        for entry in all_queries_res:
            rank = entry[mod]["first_rank"]
            r5_all.append(1.0 if (rank is not None and rank <= 5) else 0.0)
            r10_all.append(1.0 if (rank is not None and rank <= 10) else 0.0)
            mrr5_all.append((1.0 / rank) if (rank is not None and rank <= 5) else 0.0)
            mrr10_all.append((1.0 / rank) if (rank is not None and rank <= 10) else 0.0)
            if rank is not None:
                hits_all += 1
                if rank == 1:
                    rank1_all += 1
        n_all = len(all_queries_res)
        overall_summary[mod] = {
            "query_count": n_all,
            "Recall@5": sum(r5_all) / n_all,
            "Recall@10": sum(r10_all) / n_all,
            "MRR@5": sum(mrr5_all) / n_all,
            "MRR@10": sum(mrr10_all) / n_all,
            "Hits": hits_all,
            "Rank1_Hits": rank1_all,
            "Outside_Top10": n_all - sum(r10_all),
        }

    timing_summary = {
        k: compute_timing_stats(v) for k, v in all_latencies.items()
    }

    print("\n" + "=" * 80)
    print("  OVERALL RETRIEVAL EVALUATION RESULTS (N = 86 QUERIES)")
    print("=" * 80)
    print(f"{'Retrieval Modality':<20} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR@5':<8} | {'MRR@10':<8} | {'Hits':<8} | {'Rank #1'}")
    print("-" * 80)
    for mod in ["keyword", "vector", "hybrid", "scheme_reranked"]:
        s = overall_summary[mod]
        print(f"{mod:<20} | {s['Recall@5']:<10.4f} | {s['Recall@10']:<10.4f} | {s['MRR@5']:<8.4f} | {s['MRR@10']:<8.4f} | {s['Hits']:<2d}/86    | {s['Rank1_Hits']:<2d}/86")

    print("\n" + "=" * 80)
    print("  LANGUAGE BREAKDOWN (SCHEME-RERANKED PRODUCTION MODE)")
    print("=" * 80)
    print(f"{'Language':<12} | {'Queries':<8} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR@5':<8} | {'MRR@10':<8} | {'Hits':<8} | {'Rank #1'}")
    print("-" * 80)
    for lang_name, s_dict in [("English", sum_en["scheme_reranked"]), ("Hindi", sum_hi["scheme_reranked"]), ("Gujarati", sum_gu["scheme_reranked"])]:
        print(f"{lang_name:<12} | {s_dict['query_count']:<8d} | {s_dict['Recall@5']:<10.4f} | {s_dict['Recall@10']:<10.4f} | {s_dict['MRR@5']:<8.4f} | {s_dict['MRR@10']:<8.4f} | {s_dict['Hits']:<2d}/{s_dict['query_count']}   | {s_dict['Rank1_Hits']:<2d}/{s_dict['query_count']}")

    print("\n" + "=" * 80)
    print("  FINAL PRODUCTION LATENCY SUMMARY (MS) OVER 86 QUERIES")
    print("=" * 80)
    print(f"{'Operation':<24} | {'Avg':<8} | {'Median':<8} | {'p95':<8} | {'Min':<8} | {'Max'}")
    print("-" * 80)
    for k, name in [
        ("t_embed", "Query Embedding"),
        ("t_vec_sql", "Vector SQL (HNSW)"),
        ("t_vec_total", "Vector Total Leg"),
        ("t_kw_sql", "Keyword SQL (GIN)"),
        ("t_hybrid_concurrent", "Hybrid Leg (Parallel)"),
        ("t_scheme_rerank", "Scheme Reranking"),
        ("t_prod_e2e", "Full Prod Retrieval E2E"),
    ]:
        ts = timing_summary[k]
        print(f"{name:<24} | {ts['avg']:<8.2f} | {ts['median']:<8.2f} | {ts['p95']:<8.2f} | {ts['min']:<8.2f} | {ts['max']:.2f}")

    # Build final structured JSON
    final_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "scope": {
            "total_queries": len(all_queries_res),
            "english_queries": len(en_queries),
            "hindi_queries": len(hi_queries_corrected),
            "gujarati_queries": len(gu_queries_corrected),
        },
        "overall_metrics": overall_summary,
        "language_metrics": {
            "english": sum_en,
            "hindi": sum_hi,
            "gujarati": sum_gu,
        },
        "latency_stats_ms": timing_summary,
        "per_query_results": all_queries_res,
    }

    out_json_path = PROJECT_ROOT / "reports" / "final_retrieval_evaluation.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Saved structured evaluation results to: {out_json_path}")

    # Final DB integrity audit
    with engine.connect() as conn:
        print("\n[5/5] Post-Evaluation Database Integrity Audit...")
        post = get_db_counts(conn)
        assert pre == post, "Database integrity mismatch detected!"
        print("  ✓ Read-only guarantee confirmed: 100% database parity preserved.")


if __name__ == "__main__":
    main()
