"""
src/retrieval/hybrid_retriever.py
=================================
Hybrid Retrieval Module using Reciprocal Rank Fusion (RRF) for SugamGov AI RAG.

Architecture & Role in SugamGov:
--------------------------------
1. What Hybrid Retrieval Does:
   Combines two complementary search modalities into a unified ranked list:
     - Lexical Search: PostgreSQL Full-Text Search (KeywordRetriever)
     - Dense Semantic Search: pgvector Cosine Similarity (VectorRetriever)
   
   The ranking of each candidate document is fused using Cormack et al.'s standard
   Reciprocal Rank Fusion (RRF) algorithm:
       RRF_score(d) = sum_{m in M} 1 / (k_rrf + rank_m(d))
   where M = {keyword, vector} and rank_m(d) is the 1-based rank position (1, 2, 3...).

2. Candidate Retrieval Strategy:
   To ensure high-quality fusion, the retriever queries candidate_k items (default: 25)
   from each modality before fusing, which is greater than the final top_k (default: 10).
   Candidates appearing in both modalities receive combined reciprocal rank boosts.
   Candidates appearing in only one modality receive their single reciprocal rank contribution.

3. Coverage & Semantic Embedding State:
   - Current State: Dense vector embeddings currently exist for 1,260 chunks, while
     19,237 chunks await completion due to Gemini API free-tier quotas.
   - Resiliency: For chunks with embeddings, the hybrid retriever combines both lexical
     and semantic signals. For chunks without embeddings, lexical signals ensure they
     remain retrievable. Once all 20,497 chunk embeddings are generated, this same
     hybrid retriever will automatically provide comprehensive semantic coverage.

4. Evaluation Note:
   This module implements and verifies the structural correctness of the RRF fusion algorithm.
   Empirical claims regarding whether hybrid retrieval outperforms keyword or vector
   retrieval on SugamGov require an end-to-end labeled evaluation dataset (Step 6+).
"""

import os
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.retrieval.models import RetrievalResult
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.vector_retriever import VectorRetriever, generate_query_embedding

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_K = 25
DEFAULT_TOP_K = 10


def compute_rrf_score(rank: Optional[int], rrf_k: int = DEFAULT_RRF_K) -> float:
    """Computes the Reciprocal Rank Fusion component for a 1-based rank."""
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / (rrf_k + rank)


class HybridRetriever:
    """
    Hybrid retriever combining PostgreSQL keyword search and pgvector semantic search
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(
        self,
        engine_or_url: Optional[Union[Engine, str]] = None,
        keyword_retriever: Optional[KeywordRetriever] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        client: Optional[Any] = None,
        rrf_k: int = DEFAULT_RRF_K,
        default_candidate_k: int = DEFAULT_CANDIDATE_K,
    ):
        """
        Initializes the HybridRetriever with database connection and component retrievers.
        """
        if isinstance(engine_or_url, Engine):
            self.engine = engine_or_url
        elif isinstance(engine_or_url, str):
            self.engine = create_engine(engine_or_url)
        else:
            if ENV_PATH.exists():
                load_dotenv(ENV_PATH)
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL is required to initialize HybridRetriever.")
            self.engine = create_engine(db_url)

        self.client = client
        self.rrf_k = rrf_k
        self.default_candidate_k = default_candidate_k

        self.keyword_retriever = keyword_retriever or KeywordRetriever(engine_or_url=self.engine)
        self.vector_retriever = vector_retriever or VectorRetriever(engine_or_url=self.engine, client=self.client)

    def retrieve(
        self,
        query: str,
        state: Optional[str] = None,
        level: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: Optional[int] = None,
        keyword_candidate_k: Optional[int] = None,
        vector_candidate_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        query_vector: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        """
        Performs hybrid retrieval using Reciprocal Rank Fusion.

        Args:
            query: User search text string
            state: Optional state filter (e.g. 'Gujarat')
            level: Optional level filter ('Central' or 'State')
            category: Optional category filter (e.g. 'Education & Learning')
            top_k: Number of final fused results to return (default: 10)
            candidate_k: Number of candidates to retrieve per modality (default: 25)
            keyword_candidate_k: Specific candidate count for keyword retrieval (overrides candidate_k)
            vector_candidate_k: Specific candidate count for vector retrieval (overrides candidate_k)
            rrf_k: RRF smoothing constant (default: 60)
            query_vector: Optional pre-computed 768-dim query embedding vector

        Returns:
            List of RetrievalResult objects sorted by RRF score descending.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        top_k = max(1, top_k)
        effective_rrf_k = rrf_k if rrf_k is not None else self.rrf_k
        base_candidate_k = candidate_k or self.default_candidate_k
        k_kw = max(top_k, keyword_candidate_k or base_candidate_k)
        k_vec = max(top_k, vector_candidate_k or base_candidate_k)

        # Execute Keyword and Vector Retrieval concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_kw = executor.submit(
                self.keyword_retriever.retrieve,
                query=clean_query,
                state=state,
                level=level,
                category=category,
                top_k=k_kw,
            )
            if query_vector is not None:
                future_vec = executor.submit(
                    self.vector_retriever.retrieve_by_vector,
                    query_vector=query_vector,
                    state=state,
                    level=level,
                    category=category,
                    top_k=k_vec,
                )
            else:
                future_vec = executor.submit(
                    self.vector_retriever.retrieve,
                    query=clean_query,
                    state=state,
                    level=level,
                    category=category,
                    top_k=k_vec,
                )

            keyword_candidates = future_kw.result()
            vector_candidates = future_vec.result()

        # 3. Reciprocal Rank Fusion (RRF) using chunk_id as unique identifier
        # fusion_map: chunk_id -> dict with metadata & rank info
        fusion_map: Dict[str, Dict[str, Any]] = {}

        # Add keyword candidates with 1-based ranks
        for rank_kw, item in enumerate(keyword_candidates, start=1):
            cid = item.chunk_id
            fusion_map[cid] = {
                "base_item": item,
                "keyword_rank": rank_kw,
                "vector_rank": None,
                "keyword_score": item.score,
                "vector_score": None,
                "rrf_score": compute_rrf_score(rank_kw, rrf_k=effective_rrf_k),
            }

        # Add or update vector candidates with 1-based ranks
        for rank_vec, item in enumerate(vector_candidates, start=1):
            cid = item.chunk_id
            vec_rrf_contrib = compute_rrf_score(rank_vec, rrf_k=effective_rrf_k)

            if cid in fusion_map:
                fusion_map[cid]["vector_rank"] = rank_vec
                fusion_map[cid]["vector_score"] = item.score
                fusion_map[cid]["rrf_score"] += vec_rrf_contrib
            else:
                fusion_map[cid] = {
                    "base_item": item,
                    "keyword_rank": None,
                    "vector_rank": rank_vec,
                    "keyword_score": None,
                    "vector_score": item.score,
                    "rrf_score": vec_rrf_contrib,
                }

        # 4. Sort deterministically:
        # Primary: rrf_score descending
        # Secondary: chunk_id ascending (tie-breaker)
        sorted_entries = sorted(
            fusion_map.values(),
            key=lambda e: (-e["rrf_score"], e["base_item"].chunk_id),
        )

        # 5. Build final RetrievalResult instances
        final_results: List[RetrievalResult] = []
        for entry in sorted_entries[:top_k]:
            b = entry["base_item"]
            final_results.append(
                RetrievalResult(
                    chunk_id=b.chunk_id,
                    scheme_id=b.scheme_id,
                    scheme_name=b.scheme_name,
                    field_name=b.field_name,
                    chunk_text=b.chunk_text,
                    score=float(entry["rrf_score"]),
                    metadata=b.metadata,
                    state=b.state,
                    level=b.level,
                    categories=b.categories,
                    keyword_rank=entry["keyword_rank"],
                    vector_rank=entry["vector_rank"],
                    keyword_score=entry["keyword_score"],
                    vector_score=entry["vector_score"],
                )
            )

        return final_results


def retrieve_hybrid(
    query: str,
    state: Optional[str] = None,
    level: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rrf_k: int = DEFAULT_RRF_K,
    engine: Optional[Engine] = None,
    client: Optional[Any] = None,
    query_vector: Optional[List[float]] = None,
) -> List[RetrievalResult]:
    """
    Convenience function for hybrid retrieval using Reciprocal Rank Fusion.
    """
    retriever = HybridRetriever(engine_or_url=engine, client=client, rrf_k=rrf_k)
    return retriever.retrieve(
        query=query,
        state=state,
        level=level,
        category=category,
        top_k=top_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        query_vector=query_vector,
    )
