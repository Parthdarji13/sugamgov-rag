"""
src/retrieval/keyword_retriever.py
==================================
PostgreSQL Full-Text Search (FTS) Keyword Retriever for SugamGov AI RAG.

Architecture & Role in SugamGov:
--------------------------------
1. What Keyword Retrieval Does:
   - Performs lexical search across scheme chunks stored in PostgreSQL ('scheme_chunks' table).
   - Generates weighted tsvectors combining:
       * Scheme title ('schemes.scheme_name') at Weight 'A' (1.0 - highest priority)
       * Chunk body ('scheme_chunks.chunk_text') at Weight 'B' (0.4)
   - Uses 'websearch_to_tsquery' to naturally parse user queries, supporting boolean operators,
     quoted phrases, and word stems.
   - Computes deterministic relevance scores using PostgreSQL's native 'ts_rank_cd'
     (Cover Density Ranking), which rewards query term proximity and frequency.
   - Applies strict metadata pre-filtering (state, administrative level, category)
     BEFORE text matching and ranking.

2. Why Built Before Semantic Retrieval:
   - Resiliency & Full Coverage: Dense vector embeddings currently exist for 1,260 chunks,
     while 19,237 chunks await completion due to Gemini API free-tier quotas. Keyword
     retrieval provides immediate, production-grade search over 100% of the 20,497 chunks
     with zero external API dependencies or latency.
   - Exact Nomenclature Matching: Users frequently search for official scheme names, specific
     acts, legal terms, or regional abbreviations (e.g. 'GBOCWWB', 'Kisan', 'Awas').
     Lexical full-text search excels at these exact tokens where embedding models can be fuzzy.
   - Decoupled System Design: Establishing the retrieval interfaces, filters, and data models
     now allows backend components (API routes, caching, evaluation) to proceed without
     waiting for embedding generation.

3. How It Will Later Be Combined With Vector Retrieval:
   - Once all 20,497 chunk embeddings are generated and an HNSW vector index is created in pgvector,
     this keyword retriever will form the lexical leg of a Hybrid Retrieval pipeline.
   - Results from both keyword search and vector cosine similarity search will be merged using
     Reciprocal Rank Fusion (RRF):
         RRF_Score(d) = 1 / (60 + rank_keyword(d)) + 1 / (60 + rank_vector(d))
   - Hybrid search guarantees the best of both worlds: exact keyword recall for specific schemes
     combined with deep semantic understanding for exploratory, conversational queries.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.retrieval.models import RetrievalResult

# Default location for environment configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


class KeywordRetriever:
    """
    PostgreSQL-based keyword retriever for scheme_chunks.
    Executes read-only queries with full-text search and metadata filtering.
    """
    def __init__(self, engine_or_url: Optional[Union[Engine, str]] = None):
        """
        Initializes the retriever with a SQLAlchemy Engine or database URL.
        If None, loads DATABASE_URL from .env.
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
                raise ValueError("DATABASE_URL environment variable is required to initialize KeywordRetriever.")
            self.engine = create_engine(db_url)

    def retrieve(
        self,
        query: str,
        state: Optional[str] = None,
        level: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Retrieves top_k relevant scheme_chunks for a given query with optional metadata filters.

        Args:
            query: User search text (e.g., "farmer financial assistance")
            state: Optional state filter (e.g., "Gujarat")
            level: Optional level filter ('Central' or 'State')
            category: Optional category filter (e.g., "Education & Learning")
            top_k: Maximum number of results to return (default: 10)

        Returns:
            List of RetrievalResult objects sorted by relevance score descending.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        top_k = max(1, top_k)

        # 1. Build metadata filter clauses (applied BEFORE ranking)
        filter_clauses = []
        params: Dict[str, Any] = {
            "query": clean_query,
            "top_k": top_k,
        }

        if state:
            clean_state = state.strip()
            # Match schemes.state column OR elements inside schemes.states JSONB array
            filter_clauses.append("""(
                s.state ILIKE :state 
                OR EXISTS (
                    SELECT 1 FROM jsonb_array_elements_text(s.states) st 
                    WHERE st ILIKE :state
                )
            )""")
            params["state"] = clean_state

        if level:
            clean_level = level.strip()
            filter_clauses.append("s.level ILIKE :level")
            params["level"] = clean_level

        if category:
            clean_cat = category.strip()
            filter_clauses.append("""EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(s.categories) cat 
                WHERE cat ILIKE :category
            )""")
            params["category"] = f"%{clean_cat}%"

        where_filters = ""
        if filter_clauses:
            where_filters = "AND " + " AND ".join(filter_clauses)

        # 2. Primary Query: Natural websearch full-text search with Cover Density Ranking using persisted search_vector GIN index
        sql_primary = text(f"""
            WITH filtered_chunks AS (
                SELECT 
                    c.id,
                    c.chunk_id,
                    c.scheme_id,
                    s.scheme_name,
                    c.field_name,
                    c.chunk_text,
                    c.metadata,
                    s.state,
                    s.level,
                    s.categories,
                    c.search_vector
                FROM scheme_chunks c
                JOIN schemes s ON c.scheme_id = s.scheme_id
                WHERE 1=1
                  {where_filters}
            )
            SELECT 
                f.chunk_id,
                f.scheme_id,
                f.scheme_name,
                f.field_name,
                f.chunk_text,
                f.metadata,
                f.state,
                f.level,
                f.categories,
                ts_rank_cd(f.search_vector, q.query) AS score
            FROM filtered_chunks f,
            websearch_to_tsquery('english', :query) q(query)
            WHERE f.search_vector @@ q.query
            ORDER BY score DESC, f.id ASC
            LIMIT :top_k;
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(sql_primary, params).fetchall()

            # 3. Fallback: If strict AND search returned 0 results on a multi-word query,
            # execute a relaxed OR search to recover partial keyword matches.
            if not rows:
                words = [w for w in re.findall(r'\b[A-Za-z0-9_]+\b', clean_query) if len(w) > 1]
                if len(words) > 1:
                    or_tsquery = " | ".join(words)
                    params["or_query"] = or_tsquery
                    sql_fallback = text(f"""
                        WITH filtered_chunks AS (
                            SELECT 
                                c.id,
                                c.chunk_id,
                                c.scheme_id,
                                s.scheme_name,
                                c.field_name,
                                c.chunk_text,
                                c.metadata,
                                s.state,
                                s.level,
                                s.categories,
                                c.search_vector
                            FROM scheme_chunks c
                            JOIN schemes s ON c.scheme_id = s.scheme_id
                            WHERE 1=1
                              {where_filters}
                        )
                        SELECT 
                            f.chunk_id,
                            f.scheme_id,
                            f.scheme_name,
                            f.field_name,
                            f.chunk_text,
                            f.metadata,
                            f.state,
                            f.level,
                            f.categories,
                            ts_rank_cd(f.search_vector, q.query) AS score
                        FROM filtered_chunks f,
                        to_tsquery('english', :or_query) q(query)
                        WHERE f.search_vector @@ q.query
                        ORDER BY score DESC, f.id ASC
                        LIMIT :top_k;
                    """)
                    rows = conn.execute(sql_fallback, params).fetchall()

        # 4. Map SQL rows to strongly-typed RetrievalResult objects
        results: List[RetrievalResult] = []
        for row in rows:
            cats = row.categories if isinstance(row.categories, list) else []
            meta = row.metadata if isinstance(row.metadata, dict) else {}

            results.append(
                RetrievalResult(
                    chunk_id=str(row.chunk_id),
                    scheme_id=str(row.scheme_id),
                    scheme_name=str(row.scheme_name),
                    field_name=str(row.field_name),
                    chunk_text=str(row.chunk_text),
                    score=float(row.score),
                    metadata=meta,
                    state=row.state,
                    level=row.level,
                    categories=cats,
                )
            )

        return results

    def retrieve_by_scheme_name(
        self,
        scheme_name_query: str,
        state: Optional[str] = None,
        level: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Retrieves scheme chunks by directly matching scheme_name using ILIKE.
        This bypasses FTS entirely and is critical for short/acronym queries
        like 'pm kisan' where FTS may fail.

        Strategy:
          1. Try exact ILIKE match with the full query string
          2. If <3 results, also try matching each significant word (>= 4 chars)
             individually against scheme_name (OR logic)

        Args:
            scheme_name_query: Scheme name or partial name to search (e.g. 'Pradhan Mantri Kisan Samman Nidhi')
            state: Optional state filter
            level: Optional level filter
            category: Optional category filter
            top_k: Maximum number of results to return

        Returns:
            List of RetrievalResult objects from matching schemes, scored by a fixed high relevance.
        """
        clean_query = scheme_name_query.strip()
        if not clean_query:
            return []

        top_k = max(1, top_k)

        # Build metadata filter clauses
        filter_clauses = []
        params: Dict[str, Any] = {"top_k": top_k}

        if state:
            filter_clauses.append("""(
                s.state ILIKE :state 
                OR EXISTS (
                    SELECT 1 FROM jsonb_array_elements_text(s.states) st 
                    WHERE st ILIKE :state
                )
            )""")
            params["state"] = state.strip()

        if level:
            filter_clauses.append("s.level ILIKE :level")
            params["level"] = level.strip()

        if category:
            filter_clauses.append("""EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(s.categories) cat 
                WHERE cat ILIKE :category
            )""")
            params["category"] = f"%{category.strip()}%"

        where_filters = ""
        if filter_clauses:
            where_filters = "AND " + " AND ".join(filter_clauses)

        # Strategy 1: Full phrase ILIKE match
        params["name_pattern"] = f"%{clean_query}%"
        sql_name = text(f"""
            SELECT 
                c.chunk_id,
                c.scheme_id,
                s.scheme_name,
                c.field_name,
                c.chunk_text,
                c.metadata,
                s.state,
                s.level,
                s.categories
            FROM scheme_chunks c
            JOIN schemes s ON c.scheme_id = s.scheme_id
            WHERE s.scheme_name ILIKE :name_pattern
              {where_filters}
            ORDER BY
                CASE WHEN c.field_name = 'scheme_name' THEN 0
                     WHEN c.field_name = 'benefits' THEN 1
                     WHEN c.field_name = 'eligibility_criteria' THEN 2
                     ELSE 3
                END,
                c.id ASC
            LIMIT :top_k;
        """)

        all_rows = []
        with self.engine.connect() as conn:
            rows = conn.execute(sql_name, params).fetchall()
            all_rows.extend(rows)

            # Strategy 2: If not enough results, try individual significant words (OR)
            if len(all_rows) < 3:
                words = [w for w in re.findall(r'\b[A-Za-z0-9]+\b', clean_query) if len(w) >= 4]
                if words:
                    # Build OR ILIKE conditions for each word
                    word_conditions = []
                    for i, word in enumerate(words[:6]):  # Limit to 6 words
                        param_key = f"word_{i}"
                        word_conditions.append(f"s.scheme_name ILIKE :{param_key}")
                        params[param_key] = f"%{word}%"

                    if word_conditions:
                        existing_ids = {r.chunk_id for r in all_rows}
                        word_where = " OR ".join(word_conditions)
                        sql_words = text(f"""
                            SELECT 
                                c.chunk_id,
                                c.scheme_id,
                                s.scheme_name,
                                c.field_name,
                                c.chunk_text,
                                c.metadata,
                                s.state,
                                s.level,
                                s.categories
                            FROM scheme_chunks c
                            JOIN schemes s ON c.scheme_id = s.scheme_id
                            WHERE ({word_where})
                              {where_filters}
                            ORDER BY
                                CASE WHEN c.field_name = 'scheme_name' THEN 0
                                     WHEN c.field_name = 'benefits' THEN 1
                                     WHEN c.field_name = 'eligibility_criteria' THEN 2
                                     ELSE 3
                                END,
                                c.id ASC
                            LIMIT :top_k;
                        """)
                        word_rows = conn.execute(sql_words, params).fetchall()
                        for wr in word_rows:
                            if str(wr.chunk_id) not in existing_ids:
                                all_rows.append(wr)

        # Map to RetrievalResult with a fixed high relevance score
        # (these are direct name matches, so they're highly relevant)
        results: List[RetrievalResult] = []
        for row in all_rows[:top_k]:
            cats = row.categories if isinstance(row.categories, list) else []
            meta = row.metadata if isinstance(row.metadata, dict) else {}

            results.append(
                RetrievalResult(
                    chunk_id=str(row.chunk_id),
                    scheme_id=str(row.scheme_id),
                    scheme_name=str(row.scheme_name),
                    field_name=str(row.field_name),
                    chunk_text=str(row.chunk_text),
                    score=0.5,  # High fixed score — direct name match is very relevant
                    metadata=meta,
                    state=row.state,
                    level=row.level,
                    categories=cats,
                )
            )

        return results


def retrieve_keywords(
    query: str,
    state: Optional[str] = None,
    level: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = 10,
    engine: Optional[Engine] = None,
) -> List[RetrievalResult]:
    """
    Convenience function to perform keyword retrieval using KeywordRetriever.
    """
    retriever = KeywordRetriever(engine_or_url=engine)
    return retriever.retrieve(
        query=query,
        state=state,
        level=level,
        category=category,
        top_k=top_k,
    )
