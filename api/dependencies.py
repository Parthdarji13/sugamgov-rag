"""
api/dependencies.py
===================
Dependency injection providers for SugamGov AI RAG FastAPI Backend.

Maintains thread-safe singleton instances for:
- Settings
- RetrievalService (reusing database connection pool)
- EvidenceContextBuilder
- RAGGenerator (reusing Gemini API client)
"""

from functools import lru_cache
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from api.config import get_settings, Settings
from src.retrieval.retrieval_service import RetrievalService
from src.retrieval.context_builder import EvidenceContextBuilder
from src.generation.rag_generator import RAGGenerator
from src.retrieval.vector_retriever import get_embedding_model, VectorRetriever
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

_engine: Optional[Engine] = None
_retrieval_service: Optional[RetrievalService] = None
_context_builder: Optional[EvidenceContextBuilder] = None
_rag_generator: Optional[RAGGenerator] = None


def get_db_engine() -> Engine:
    """Returns or creates the shared SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        if not settings.database_url:
            raise ValueError("DATABASE_URL environment variable is required.")
        _engine = create_engine(
            settings.database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_local_embedding_model():
    """Provides cached singleton SentenceTransformer embedding model."""
    return get_embedding_model()


def get_retrieval_service() -> RetrievalService:
    """Provides singleton RetrievalService instance with pooled DB engine and cached model."""
    global _retrieval_service
    if _retrieval_service is None:
        engine = get_db_engine()
        model = get_local_embedding_model()
        kw_retriever = KeywordRetriever(engine_or_url=engine)
        vec_retriever = VectorRetriever(engine_or_url=engine, model=model)
        hybrid_retriever = HybridRetriever(
            engine_or_url=engine,
            keyword_retriever=kw_retriever,
            vector_retriever=vec_retriever,
        )
        _retrieval_service = RetrievalService(
            engine_or_url=engine,
            hybrid_retriever=hybrid_retriever,
        )
    return _retrieval_service


def get_context_builder() -> EvidenceContextBuilder:
    """Provides singleton EvidenceContextBuilder instance."""
    global _context_builder
    if _context_builder is None:
        _context_builder = EvidenceContextBuilder()
    return _context_builder


def get_rag_generator() -> RAGGenerator:
    """Provides singleton RAGGenerator instance configured with settings."""
    global _rag_generator
    if _rag_generator is None:
        settings = get_settings()
        _rag_generator = RAGGenerator(
            model_name=settings.gemini_generation_model,
        )
    return _rag_generator


def reset_dependencies() -> None:
    """Helper for testing: resets singletons."""
    global _engine, _retrieval_service, _context_builder, _rag_generator
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception:
            pass
    _engine = None
    _retrieval_service = None
    _context_builder = None
    _rag_generator = None

    try:
        from api.chat.session_store import get_session_store
        get_session_store().clear_all()
    except Exception:
        pass
