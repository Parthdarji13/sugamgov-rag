"""
src/retrieval/vector_retriever.py
=================================
PostgreSQL + pgvector Semantic / Vector Retriever for SugamGov AI RAG.

Architecture & Role in SugamGov:
--------------------------------
1. What Vector Retrieval Does:
   - Computes dense semantic similarity between a user query vector and chunk embeddings
     stored in the 'scheme_chunks.embedding_local' pgvector column (384-dimensional).
   - Evaluates pgvector cosine distance:
         cosine_distance = (embedding_local <=> query_vector)
     and derives deterministic cosine similarity:
         cosine_similarity = 1 - cosine_distance
     Score interpretation: Higher score indicates greater semantic relevance (1.0 = identical).
   - Operates over 100% of scheme chunks (20,497 / 20,497 populated in embedding_local).
   - Joins 'schemes' to incorporate parent scheme metadata (state, level, categories).
   - Applies metadata pre-filtering (state, level, category) before final ranking and limit.
   - Returns structured 'RetrievalResult' instances matching the schema used by KeywordRetriever.

2. Query Embedding Generation:
   - Uses local multilingual embedding model 'intfloat/multilingual-e5-small' (384-dim).
   - Automatically prefixes search queries with 'query: ' as required by the model architecture.
   - Uses L2 normalization (normalize_embeddings=True).
   - Cached lazy singleton loader ensures model weights are loaded once in-memory.
   - Zero external API dependencies or network calls required for query embedding.

3. Hybrid / RRF Integration:
   - Supplies the dense vector search leg for Hybrid Search using Reciprocal Rank Fusion (RRF):
         RRF_Score(d) = 1 / (60 + rank_keyword(d)) + 1 / (60 + rank_vector(d))
"""

import os
import math
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.retrieval.models import RetrievalResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSION = 384

_model_lock = threading.Lock()
_cached_model = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> Any:
    """
    Returns a lazy singleton SentenceTransformer instance.
    The model is loaded once and cached in memory across all requests.
    """
    global _cached_model
    if _cached_model is None:
        with _model_lock:
            if _cached_model is None:
                from sentence_transformers import SentenceTransformer
                _cached_model = SentenceTransformer(model_name)
    return _cached_model


def generate_query_embedding(
    query: str,
    model: Optional[Any] = None,
    client: Optional[Any] = None,
) -> List[float]:
    """
    Generates a 384-dimensional dense embedding for a search query using
    local multilingual-e5-small model.
    
    Requires asymmetric prefix 'query: ' for queries as specified by
    the intfloat/multilingual-e5-small architecture.
    
    Args:
        query: Query text string
        model: Optional pre-loaded SentenceTransformer model instance
        client: Deprecated; retained for backwards compatibility
        
    Returns:
        384-element list of float values.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    embedder = model or get_embedding_model()

    # Prepend query: task prefix for multilingual-e5
    prefixed_query = f"query: {clean_query}"
    embedding = embedder.encode(prefixed_query, normalize_embeddings=True)

    if hasattr(embedding, "tolist"):
        values = embedding.tolist()
    else:
        values = list(embedding)

    if len(values) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Returned embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(values)}"
        )

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Generated query embedding contains non-finite values (NaN / Inf).")

    return [float(v) for v in values]


class VectorRetriever:
    """
    PostgreSQL + pgvector semantic vector retriever for scheme_chunks.
    Executes read-only cosine similarity queries against populated embedding_local vectors.
    """
    def __init__(
        self,
        engine_or_url: Optional[Union[Engine, str]] = None,
        client: Optional[Any] = None,
        model: Optional[Any] = None,
    ):
        """
        Initializes the VectorRetriever with a SQLAlchemy Engine, optional GenAI Client,
        and optional local SentenceTransformer model.
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
                raise ValueError("DATABASE_URL environment variable is required to initialize VectorRetriever.")
            self.engine = create_engine(db_url)

        self.client = client
        self.model = model

    def retrieve_by_vector(
        self,
        query_vector: List[float],
        state: Optional[str] = None,
        level: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Performs vector similarity search against populated scheme_chunks embedding_local vectors.

        Args:
            query_vector: 384-dimensional float embedding vector.
            state: Optional state filter (e.g., 'Gujarat').
            level: Optional level filter ('Central' or 'State').
            category: Optional category filter (e.g., 'Education & Learning').
            top_k: Maximum number of results to return (default: 10).

        Returns:
            List of RetrievalResult objects sorted by cosine similarity descending.
        """
        if not query_vector:
            return []

        if len(query_vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Query vector dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(query_vector)}"
            )

        top_k = max(1, top_k)
        vec_str = str(query_vector)

        # 1. Build metadata filter clauses (applied BEFORE ranking)
        filter_clauses = []
        params: Dict[str, Any] = {
            "query_vector": vec_str,
            "top_k": top_k,
        }

        if state:
            clean_state = state.strip()
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

        # 2. SQL Vector Query:
        # - Strictly filter c.embedding_local IS NOT NULL
        # - Apply metadata filters before limit
        # - Compute cosine similarity: 1 - (embedding_local <=> query_vector)
        # - Order by similarity_score DESC, f.id ASC for deterministic results
        sql = text(f"""
            WITH filtered_chunks AS (
                SELECT 
                    c.id,
                    c.chunk_id,
                    c.scheme_id,
                    s.scheme_name,
                    c.field_name,
                    c.chunk_text,
                    c.embedding_local,
                    c.metadata,
                    s.state,
                    s.level,
                    s.categories
                FROM scheme_chunks c
                JOIN schemes s ON c.scheme_id = s.scheme_id
                WHERE c.embedding_local IS NOT NULL
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
                1 - (f.embedding_local <=> CAST(:query_vector AS vector)) AS similarity_score
            FROM filtered_chunks f
            ORDER BY similarity_score DESC, f.id ASC
            LIMIT :top_k;
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        # 3. Map SQL rows to strongly-typed RetrievalResult objects
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
                    score=float(row.similarity_score),
                    metadata=meta,
                    state=row.state,
                    level=row.level,
                    categories=cats,
                )
            )

        return results

    def retrieve(
        self,
        query: str,
        state: Optional[str] = None,
        level: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Generates query embedding on-the-fly and retrieves top_k semantically similar chunks.

        Args:
            query: Query text string (e.g., 'farmer financial assistance')
            state: Optional state filter (e.g., 'Gujarat')
            level: Optional level filter ('Central' or 'State')
            category: Optional category filter (e.g., 'Education & Learning')
            top_k: Maximum number of results to return (default: 10)

        Returns:
            List of RetrievalResult objects sorted by cosine similarity descending.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        query_vector = generate_query_embedding(
            clean_query,
            model=self.model,
            client=self.client,
        )
        return self.retrieve_by_vector(
            query_vector=query_vector,
            state=state,
            level=level,
            category=category,
            top_k=top_k,
        )


def retrieve_vectors(
    query_or_vector: Union[str, List[float]],
    state: Optional[str] = None,
    level: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = 10,
    engine: Optional[Engine] = None,
    client: Optional[Any] = None,
    model: Optional[Any] = None,
) -> List[RetrievalResult]:
    """
    Convenience function for semantic vector retrieval.
    Accepts either a query text string or a pre-computed 384-dim float vector.
    """
    retriever = VectorRetriever(engine_or_url=engine, client=client, model=model)
    if isinstance(query_or_vector, str):
        return retriever.retrieve(
            query=query_or_vector,
            state=state,
            level=level,
            category=category,
            top_k=top_k,
        )
    return retriever.retrieve_by_vector(
        query_vector=query_or_vector,
        state=state,
        level=level,
        category=category,
        top_k=top_k,
    )
