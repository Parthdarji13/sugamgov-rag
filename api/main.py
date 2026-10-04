"""
api/main.py
===========
FastAPI Backend Application for SugamGov AI RAG.

Endpoints:
  GET  /health        - Safe health check
  POST /api/retrieve  - Hybrid keyword + vector scheme retrieval (no LLM generation)
  POST /api/generate  - Strictly grounded answer generation from retrieved evidence

Architecture:
  User Query -> RetrievalService -> SchemeReranker -> EvidenceContextBuilder -> RAGGenerator -> Structured Response

Start Server:
  uvicorn api.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.schemas import (
    RetrievalChatRequest,
    HealthResponse,
    RetrieveResponse,
    ChatResponse,
    SchemeResultSchema,
    EvidenceChunkSchema,
    EvidenceCitationSchema,
)
from api.dependencies import (
    get_retrieval_service,
    get_context_builder,
    get_rag_generator,
    reset_dependencies,
)
from api.chat.routes import router as chat_router
from src.retrieval.retrieval_service import RetrievalService
from src.retrieval.context_builder import EvidenceContextBuilder
from src.generation.rag_generator import RAGGenerator

# Setup sanitized logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sugamgov_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context for startup and shutdown resource handling."""
    logger.info("SugamGov AI RAG API starting up...")
    try:
        from src.retrieval.vector_retriever import get_embedding_model
        logger.info("Pre-warming local multilingual embedding model (intfloat/multilingual-e5-small)...")
        get_embedding_model()
        logger.info("Local multilingual embedding model pre-warmed and ready.")
    except Exception as e:
        logger.warning("Could not pre-warm embedding model at startup: %s", e)
    yield
    logger.info("SugamGov AI RAG API shutting down...")
    reset_dependencies()


app = FastAPI(
    title="SugamGov AI RAG API",
    description="""
Grounded AI Retrieval-Augmented Generation (RAG) API for Indian Government Schemes.

### Architecture & Capabilities:
- **Vector Coverage**: 20,497 of 20,497 scheme chunks (100% complete coverage) have populated local embeddings using `intfloat/multilingual-e5-small` (384-dimensional pgvector).
- **Hybrid Retrieval**: Combines PostgreSQL Full-Text Search (lexical) with pgvector semantic similarity via Reciprocal Rank Fusion (RRF).
- **Scheme Consolidation**: Deduplicates chunks by scheme to return unique, highly relevant schemes with their supporting evidence chunks.
- **Strict Grounding**: The answer generation layer strictly uses retrieved evidence. Facts or eligibility conditions not present in the evidence are explicitly flagged as 'Not available in retrieved evidence'.
- **Conversational Chat**: Multi-turn chat (`POST /api/chat`) with bounded in-memory history and deterministic query reformulation for follow-up turns. Sessions can be cleared via `DELETE /api/chat/{session_id}`.
- **Multilingual Support**: Supports English (`en`), Hindi (`hi`), Gujarati (`gu`), and automatic query language detection (`auto`). Official scheme names remain preserved.
- **Security & Privacy**: Zero credentials, database connection strings, or Gemini API keys are exposed. Untrusted database evidence is isolated against prompt-injection.

### Current Limitations:
- **Conversational Persistence**: Sessions are currently held in thread-safe application memory (bounded to 10 messages). Full persistence to MongoDB is planned for a future step.
- **Sources**: The `sources` table currently contains 0 records; responses do not claim live official website verification.
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS Middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global Exception Handlers (Credential & Secret Isolation)
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """
    Safely catches unhandled exceptions to guarantee zero credentials,
    passwords, or API keys are ever leaked in error responses.
    """
    err_str = str(exc)
    logger.error("Unhandled exception processing request: %s", type(exc).__name__)

    # Specific handling for Google GenAI Quota (429)
    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": "Gemini API rate limit or quota exceeded. Please retry after a brief pause."
            },
        )

    # Specific handling for Google GenAI Service Unavailable (503 / 500)
    if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "Gemini generation service is temporarily unavailable due to high demand. Please try again shortly."
            },
        )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred while processing the request."},
    )


# ============================================================================
# Endpoints
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns operational status of the SugamGov AI RAG API without revealing credentials.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Safe health check endpoint."""
    return HealthResponse(status="ok", service="sugamgov-rag-api")


@app.post(
    "/api/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve Government Schemes",
    description="""
Executes hybrid keyword + vector retrieval and scheme-level consolidation.
Does NOT invoke LLM answer generation (zero Gemini generation quota used).
    """,
    tags=["Retrieval"],
)
async def retrieve_schemes(
    request: RetrievalChatRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> RetrieveResponse:
    """
    Retrieves unique government schemes matching user search query and filters.
    """
    try:
        retrieval_response = retrieval_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            state=request.state,
            level=request.level,
            category=request.category,
        )

        results_schema: list[SchemeResultSchema] = []
        for r in retrieval_response.results:
            evidence_schema = [
                EvidenceChunkSchema(
                    chunk_id=c.chunk_id,
                    field_name=c.field_name,
                    chunk_text=c.chunk_text,
                    score=c.score,
                    keyword_rank=c.keyword_rank,
                    vector_rank=c.vector_rank,
                    keyword_score=c.keyword_score,
                    vector_score=c.vector_score,
                )
                for c in r.evidence_chunks
            ]
            results_schema.append(
                SchemeResultSchema(
                    scheme_id=r.scheme_id,
                    scheme_name=r.scheme_name,
                    level=r.level,
                    state=r.state,
                    states=r.states,
                    categories=r.categories,
                    score=r.best_retrieval_score,
                    best_chunk_id=r.best_chunk_id,
                    best_field=r.best_field_name,
                    evidence_chunks=evidence_schema,
                )
            )

        return RetrieveResponse(
            query=retrieval_response.query,
            results=results_schema,
            total_results=retrieval_response.total_results,
            retrieval_method=retrieval_response.retrieval_method,
            filters=retrieval_response.filters,
            coverage_info=retrieval_response.coverage_info,
        )
    except Exception as e:
        logger.error("Error during scheme retrieval: %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval service failed to execute query.",
        )


@app.post(
    "/api/generate",
    response_model=ChatResponse,
    summary="Grounded Scheme Answer Generation",
    description="""
Executes full grounded RAG pipeline:
1. Hybrid retrieval and scheme consolidation via `RetrievalService`.
2. Deterministic context assembly and character budget enforcement via `EvidenceContextBuilder`.
3. Strict grounded LLM answer generation with citations via `RAGGenerator`.
If no evidence is found, returns a deterministic factual response without calling Gemini.
    """,
    tags=["Generation"],
)
async def generate_grounded_answer(
    request: RetrievalChatRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    context_builder: EvidenceContextBuilder = Depends(get_context_builder),
    generator: RAGGenerator = Depends(get_rag_generator),
) -> ChatResponse:
    """
    Generates a strictly grounded, factual answer to the user's question using retrieved evidence.
    """
    try:
        # 1. Retrieve relevant schemes
        retrieval_response = retrieval_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            state=request.state,
            level=request.level,
            category=request.category,
        )

        # 2. Build structured, bounded evidence context
        evidence_context = context_builder.build(retrieval_response=retrieval_response)

        # 3. Generate grounded answer (short-circuits automatically if evidence is empty)
        rag_answer = generator.generate(
            query=request.query,
            evidence_context=evidence_context,
            language=request.language or "auto",
        )

        evidence_used_schema: list[EvidenceCitationSchema] = []
        for c in rag_answer.evidence_used:
            if isinstance(c, dict):
                evidence_used_schema.append(
                    EvidenceCitationSchema(
                        scheme_id=str(c.get("scheme_id", "")),
                        scheme_name=str(c.get("scheme_name", "")),
                        chunk_id=str(c.get("chunk_id", "")),
                        field_name=str(c.get("field_name", "")),
                    )
                )
            else:
                evidence_used_schema.append(
                    EvidenceCitationSchema(
                        scheme_id=getattr(c, "scheme_id", ""),
                        scheme_name=getattr(c, "scheme_name", ""),
                        chunk_id=getattr(c, "chunk_id", ""),
                        field_name=getattr(c, "field_name", ""),
                    )
                )

        schemes_list = [
            s.get("scheme_id", str(s)) if isinstance(s, dict) else getattr(s, "scheme_id", str(s))
            for s in rag_answer.schemes
        ]

        return ChatResponse(
            query=rag_answer.query,
            answer=rag_answer.answer,
            language=rag_answer.language,
            grounded=rag_answer.grounded,
            confidence=rag_answer.confidence,
            schemes=schemes_list,
            evidence_used=evidence_used_schema,
            limitations=rag_answer.limitations,
            retrieval_method=retrieval_response.retrieval_method,
            total_results=retrieval_response.total_results,
            coverage_info=retrieval_response.coverage_info,
        )
    except HTTPException:
        raise
    except Exception as e:
        err_str = str(e)
        logger.error("Error during grounded generation: %s", type(e).__name__)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Gemini API rate limit or quota exceeded. Please retry after a brief pause.",
            )
        if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Gemini generation service is temporarily unavailable due to high demand.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grounded answer generation failed to complete.",
        )


# ============================================================================
# Mount Sub-Routers
# ============================================================================

app.include_router(chat_router)
