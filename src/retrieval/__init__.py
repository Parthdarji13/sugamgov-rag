"""
src/retrieval
=============
Retrieval package for SugamGov AI RAG.
Provides keyword/lexical search, data models, and future hybrid search integrations.
"""

from src.retrieval.models import RetrievalResult
from src.retrieval.keyword_retriever import KeywordRetriever, retrieve_keywords
from src.retrieval.vector_retriever import (
    VectorRetriever,
    retrieve_vectors,
    generate_query_embedding,
    get_embedding_model,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    retrieve_hybrid,
    compute_rrf_score,
    DEFAULT_RRF_K,
    DEFAULT_CANDIDATE_K,
    DEFAULT_TOP_K,
)
from src.retrieval.scheme_reranker import (
    SchemeRetrievalResult,
    SchemeReranker,
    rerank_schemes,
)
from src.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalResponse,
    retrieve_service,
)
from src.retrieval.context_builder import (
    EvidenceContextBuilder,
    EvidenceContext,
    SchemeContextItem,
    EvidenceChunkItem,
    build_evidence_context,
)

__all__ = [
    "RetrievalResult",
    "KeywordRetriever",
    "retrieve_keywords",
    "VectorRetriever",
    "retrieve_vectors",
    "generate_query_embedding",
    "get_embedding_model",
    "HybridRetriever",
    "retrieve_hybrid",
    "compute_rrf_score",
    "DEFAULT_RRF_K",
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_TOP_K",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "SchemeRetrievalResult",
    "SchemeReranker",
    "rerank_schemes",
    "RetrievalService",
    "RetrievalResponse",
    "retrieve_service",
    "EvidenceContextBuilder",
    "EvidenceContext",
    "SchemeContextItem",
    "EvidenceChunkItem",
    "build_evidence_context",
]
