"""
src/retrieval/retrieval_service.py
==================================
Unified Retrieval Service Layer for SugamGov AI RAG.

Architecture & Role in SugamGov:
--------------------------------
The RetrievalService provides the single, production-facing entry point for all
upstream retrieval operations. It sits between user-facing endpoints (FastAPI chat/search
APIs) and the internal retrieval subsystems:

User Search Request / Chat Query
               ↓
    ┌─────────────────────────┐
    │    RetrievalService     │
    │  (Request Validation)   │
    └──────────┬──────────────┘
               ↓
    ┌─────────────────────────┐
    │     HybridRetriever     │
    │ (PostgreSQL FTS + RRF)  │
    └──────────┬──────────────┘
               ↓
    ┌─────────────────────────┐
    │     SchemeReranker      │
    │  (Scheme Consolidation) │
    └──────────┬──────────────┘
               ↓
    ┌─────────────────────────┐
    │    RetrievalResponse    │
    │ (Clean Scheme Delivery) │
    └─────────────────────────┘

Key Capabilities:
  1. Input Validation & Sanitization:
     - Trims whitespace, handles whitespace-only and empty queries safely.
     - Validates and normalizes top_k and candidate_k parameters.
     - Sanitizes metadata filters (state, level, category).
     - Short-circuits blank queries without making Gemini API calls (quota preservation).
  2. Subsystem Coordination:
     - Calls HybridRetriever for fused keyword + dense vector search.
     - Automatically passes candidate pool to SchemeReranker for scheme deduplication.
  3. Structured Output & Evidence Preservation:
     - Returns RetrievalResponse dataclass with metadata, coverage info, and
       unique SchemeRetrievalResult items preserving all matching evidence chunks.
  4. Robust Error Handling:
     - Gracefully degrades to full-corpus lexical search if vector embedding API
       is quota-limited or unavailable.
     - Re-raises clean, sanitized exceptions on serious database connectivity failures
       without leaking passwords or database URLs.
  5. Dynamic Coverage Reporting:
     - Directly measures and reports populated vs NULL embedding coverage from PostgreSQL.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.retrieval.models import RetrievalResult
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    DEFAULT_RRF_K,
    DEFAULT_CANDIDATE_K,
    DEFAULT_TOP_K,
)
from src.retrieval.scheme_reranker import (
    SchemeRetrievalResult,
    SchemeReranker,
    rerank_schemes,
)
from src.retrieval.scheme_resolver import get_scheme_resolver, SchemeResolver

# Fuzzy matching for typo correction (optional — degrades gracefully if not installed)
try:
    from rapidfuzz import process as rfprocess, fuzz as rffuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

ACRONYM_MAP: Dict[str, str] = {
    # PM-KISAN variants
    "pm kisan": "Pradhan Mantri Kisan Samman Nidhi",
    "pm-kisan": "Pradhan Mantri Kisan Samman Nidhi",
    "pmkisan": "Pradhan Mantri Kisan Samman Nidhi",
    "kisan samman": "Pradhan Mantri Kisan Samman Nidhi",
    "kisan nidhi": "Pradhan Mantri Kisan Samman Nidhi",
    # PM Awas
    "pm awas": "Pradhan Mantri Awas Yojana",
    "pmay": "Pradhan Mantri Awas Yojana",
    "pm-awas": "Pradhan Mantri Awas Yojana",
    "awas yojana": "Pradhan Mantri Awas Yojana",
    # Ayushman Bharat / PM-JAY
    "ayushman bharat": "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana",
    "pmjay": "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana",
    "pm-jay": "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana",
    "pm jay": "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana",
    "jan arogya": "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana",
    # Insurance / Pension
    "pmjjby": "Pradhan Mantri Jeevan Jyoti Bima Yojana",
    "pmsby": "Pradhan Mantri Suraksha Bima Yojana",
    "apy": "Atal Pension Yojana",
    "atal pension": "Atal Pension Yojana",
    # Ujjwala
    "pmuy": "Pradhan Mantri Ujjwala Yojana",
    "ujjwala": "Pradhan Mantri Ujjwala Yojana",
    "pm ujjwala": "Pradhan Mantri Ujjwala Yojana",
    # Employment / MSME
    "pmegp": "Prime Minister Employment Generation Programme",
    "mudra": "Pradhan Mantri MUDRA Yojana",
    "pm mudra": "Pradhan Mantri MUDRA Yojana",
    "startup india": "Startup India Scheme",
    "standup india": "Stand Up India Scheme",
    # Credit / Farm
    "kcc": "Kisan Credit Card",
    "kisan credit": "Kisan Credit Card",
    # Education
    "pm vidya": "PM eVIDYA",
    "sukanya": "Sukanya Samriddhi Yojana",
    # Swachh
    "swachh bharat": "Swachh Bharat Mission",
    "sbm": "Swachh Bharat Mission",
    # Digital India
    "digital india": "Digital India Programme",
    # Fasal Bima
    "pmfby": "Pradhan Mantri Fasal Bima Yojana",
    "fasal bima": "Pradhan Mantri Fasal Bima Yojana",
    # Jan Dhan
    "jan dhan": "Pradhan Mantri Jan Dhan Yojana",
    "pmjdy": "Pradhan Mantri Jan Dhan Yojana",
    # Gramin / Rural
    "mgnrega": "Mahatma Gandhi National Rural Employment Guarantee Act",
    "nrega": "Mahatma Gandhi National Rural Employment Guarantee Act",
    "mnrega": "Mahatma Gandhi National Rural Employment Guarantee Act",
}

# Vocabulary for fuzzy typo correction — known correct spellings for common scheme-related words.
# Words from ACRONYM_MAP + a curated list of frequently mistyped scheme/govt terms.
_EXTRA_SCHEME_WORDS: List[str] = [
    # Scheme name words users often mistype
    "kisan", "pradhan", "mantri", "samman", "nidhi", "yojana", "bharat", "ayushman",
    "arogya", "jeevan", "jyoti", "bima", "suraksha", "pension", "ujjwala", "awas",
    "scholarship", "education", "farmer", "health", "pension", "insurance", "credit",
    "employment", "generation", "programme", "housing", "sanitation", "digital",
    "mahila", "balika", "kanya", "kishori", "vikas", "grameen", "rural", "urban",
    "swachh", "rashtriya", "national", "central", "state", "subsidy", "scheme",
    "welfare", "social", "financial", "assistance", "support", "benefit", "income",
    "certificate", "ration", "card", "passport", "disability", "widows", "elderly",
    "minority", "backward", "tribal", "fishermen", "artisan", "weaver", "skill",
    "training", "startup", "enterprise", "mudra", "standup", "startup", "msme",
]
_FUZZY_VOCAB: List[str] = sorted(set(
    [
        word
        for phrase in list(ACRONYM_MAP.keys()) + list(ACRONYM_MAP.values())
        for word in phrase.lower().split()
        if len(word) >= 4
    ] + [w for w in _EXTRA_SCHEME_WORDS if len(w) >= 4]
))


def _fuzzy_correct_query(query: str, threshold: int = 82) -> str:
    """
    Corrects common typos in the user query by fuzzy-matching each token
    against a vocabulary of known scheme-related terms.

    Examples:
      "pm kishan" -> "pm kisan"
      "ayushman bahart" -> "ayushman bharat"
      "pradhan mantry" -> "pradhan mantri"

    Args:
        query: Raw user query string.
        threshold: Minimum similarity score (0-100) to apply a correction. Default 82.

    Returns:
        Corrected query string (or original if no corrections needed or rapidfuzz not available).
    """
    if not _RAPIDFUZZ_AVAILABLE or not _FUZZY_VOCAB:
        return query

    tokens = query.split()
    corrected_tokens = []
    changed = False

    for token in tokens:
        # Skip short tokens, numbers, and non-latin scripts (Hindi/Gujarati)
        if len(token) < 4 or not token.isascii():
            corrected_tokens.append(token)
            continue

        lower_tok = token.lower()
        # Already an exact match — no correction needed
        if lower_tok in _FUZZY_VOCAB:
            corrected_tokens.append(token)
            continue

        result = rfprocess.extractOne(
            lower_tok,
            _FUZZY_VOCAB,
            scorer=rffuzz.WRatio,
            score_cutoff=threshold,
        )
        if result:
            best_match, score, _ = result
            # Preserve original casing style
            corrected = best_match if token.islower() else best_match.capitalize()
            corrected_tokens.append(corrected)
            changed = True
            import logging as _log
            _log.getLogger("sugamgov_api").debug(
                "[TYPO FIX] '%s' -> '%s' (score=%d)", token, corrected, score
            )
        else:
            corrected_tokens.append(token)

    return " ".join(corrected_tokens) if changed else query


@dataclass
class RetrievalResponse:
    """
    Standardized service-level retrieval response.

    Fields:
      query: The sanitized input search query
      results: List of consolidated, unique SchemeRetrievalResult items
      total_results: Number of unique schemes returned
      retrieval_method: Identification of the retrieval strategy used
      filters: Dictionary of applied metadata filters (state, level, category)
      coverage_info: Dynamic embedding coverage metrics from PostgreSQL
    """
    query: str
    results: List[SchemeRetrievalResult]
    total_results: int
    retrieval_method: str
    filters: Dict[str, Optional[str]]
    coverage_info: Dict[str, Any]

    def to_dict(self, include_evidence: bool = True) -> Dict[str, Any]:
        """Converts response to a JSON-serializable dictionary."""
        return {
            "query": self.query,
            "total_results": self.total_results,
            "retrieval_method": self.retrieval_method,
            "filters": self.filters,
            "coverage_info": self.coverage_info,
            "results": [r.to_dict(include_evidence=include_evidence) for r in self.results],
        }

    def preview(self) -> str:
        """Returns a human-readable one-line summary of the response."""
        sids = [r.scheme_id for r in self.results]
        return (
            f"RetrievalResponse(query='{self.query}', total={self.total_results}, "
            f"schemes={sids}, method='{self.retrieval_method}')"
        )


class RetrievalService:
    """
    Production-facing retrieval service orchestrating query validation,
    hybrid retrieval, scheme consolidation, and coverage reporting.
    """
    def __init__(
        self,
        engine_or_url: Optional[Union[Engine, str]] = None,
        client: Optional[Any] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        scheme_reranker: Optional[SchemeReranker] = None,
        field_priority: Optional[Dict[str, int]] = None,
        rrf_k: int = DEFAULT_RRF_K,
        default_candidate_k: int = DEFAULT_CANDIDATE_K,
    ):
        """
        Initializes the RetrievalService with database connection and sub-components.

        Args:
            engine_or_url: Optional SQLAlchemy Engine or PostgreSQL connection string.
            client: Optional Google GenAI client instance.
            hybrid_retriever: Optional pre-configured HybridRetriever instance.
            scheme_reranker: Optional pre-configured SchemeReranker instance.
            field_priority: Optional field priority dict for tie-breaking.
            rrf_k: Reciprocal Rank Fusion smoothing constant (default: 60).
            default_candidate_k: Default candidate count per modality (default: 25).
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
                raise ValueError("DATABASE_URL environment variable is required to initialize RetrievalService.")
            self.engine = create_engine(db_url)

        self.client = client
        self.rrf_k = rrf_k
        self.default_candidate_k = default_candidate_k

        # Initialize or reuse sub-components
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(
            engine_or_url=self.engine,
            client=self.client,
            rrf_k=self.rrf_k,
            default_candidate_k=self.default_candidate_k,
        )
        self.scheme_reranker = scheme_reranker or SchemeReranker(field_priority=field_priority)
        self.scheme_resolver = get_scheme_resolver(self.engine)

        # Internal cache for dynamic coverage metrics
        self._coverage_info: Optional[Dict[str, Any]] = None

    def get_coverage_info(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Queries and returns real-time embedding coverage metrics from PostgreSQL.
        Caches metrics in memory unless refresh=True is requested.

        Returns:
            Dict containing total_chunks, embedded_chunks, unembedded_chunks,
            and semantic_coverage_complete boolean flag.
        """
        if self._coverage_info is not None and not refresh:
            return self._coverage_info

        try:
            with self.engine.connect() as conn:
                total_chunks = conn.execute(text("SELECT count(id) FROM scheme_chunks;")).scalar() or 0
                embedded_chunks = conn.execute(
                    text("SELECT count(id) FROM scheme_chunks WHERE embedding_local IS NOT NULL;")
                ).scalar() or 0
                unembedded_chunks = conn.execute(
                    text("SELECT count(id) FROM scheme_chunks WHERE embedding_local IS NULL;")
                ).scalar() or 0

            self._coverage_info = {
                "total_chunks": int(total_chunks),
                "embedded_chunks": int(embedded_chunks),
                "unembedded_chunks": int(unembedded_chunks),
                "semantic_coverage_complete": bool(unembedded_chunks == 0 and total_chunks > 0),
            }
            return self._coverage_info
        except Exception:
            raise RuntimeError("Database connection failure while computing coverage metrics.") from None

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: Optional[int] = None,
        state: Optional[str] = None,
        level: Optional[str] = None,
        category: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
    ) -> RetrievalResponse:
        """
        Executes end-to-end hybrid retrieval and scheme-level consolidation.

        Args:
            query: User search text string.
            top_k: Maximum number of unique schemes to return (must be >= 1).
            candidate_k: Number of chunk candidates to retrieve per modality.
            state: Optional state filter (e.g. 'Gujarat').
            level: Optional level filter ('Central' or 'State').
            category: Optional category filter (e.g. 'Education & Learning').
            query_vector: Optional pre-computed 384-dim query embedding vector.

        Returns:
            RetrievalResponse containing unique SchemeRetrievalResult items,
            metadata filters, total count, and coverage metrics.
        """
        # 1. Query validation & sanitization
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")

        clean_query = query.strip()

        # Sanitize metadata filters
        clean_state = state.strip() if isinstance(state, str) and state.strip() else None
        clean_level = level.strip() if isinstance(level, str) and level.strip() else None
        clean_category = category.strip() if isinstance(category, str) and category.strip() else None

        active_filters: Dict[str, Optional[str]] = {
            "state": clean_state,
            "level": clean_level,
            "category": clean_category,
        }

        # IMPORTANT: Short-circuit blank or whitespace-only queries immediately
        if not clean_query:
            return RetrievalResponse(
                query=clean_query,
                results=[],
                total_results=0,
                retrieval_method="hybrid_rrf_scheme_reranked",
                filters=active_filters,
                coverage_info=self.get_coverage_info(),
            )

        # 2. Parameter validation
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")

        base_candidate_k = candidate_k if candidate_k is not None else self.default_candidate_k
        if base_candidate_k <= 0:
            raise ValueError(f"candidate_k must be greater than 0, got {base_candidate_k}")

        # Ensure candidate pool is large enough to satisfy requested top_k after consolidation
        effective_candidate_k = max(base_candidate_k, top_k * 2)

        # 3. Dynamic Scheme Resolution (handles acronyms, typos, and short names across all 5,954 schemes)
        resolved_chunks: List[RetrievalResult] = []
        hybrid_search_query = clean_query

        try:
            scheme_matches = self.scheme_resolver.resolve(
                query=clean_query,
                state=clean_state,
                top_k=top_k,
                min_confidence=75.0,
            )
            if scheme_matches:
                top_match = scheme_matches[0]
                import logging as _log
                _log.getLogger("sugamgov_api").info(
                    "[SCHEME RESOLVED] Query '%s' -> '%s' (ID=%s, score=%.1f, type=%s)",
                    clean_query, top_match.scheme_name, top_match.scheme_id, top_match.score, top_match.match_type
                )
                # Fetch authoritative chunks for all confident matches
                for sm in scheme_matches:
                    s_chunks = self.scheme_resolver.get_chunks_for_scheme(
                        scheme_id=sm.scheme_id,
                        query=clean_query,
                        limit=6,
                    )
                    resolved_chunks.extend(s_chunks)

                # Rewrite hybrid query to canonical scheme name for optimal FTS / vector search
                hybrid_search_query = top_match.scheme_name
        except Exception as e:
            import logging as _log
            _log.getLogger("sugamgov_api").warning("SchemeResolver error: %s", e)

        # 4. Candidate Retrieval via HybridRetriever
        try:
            candidate_chunks = self.hybrid_retriever.retrieve(
                query=hybrid_search_query,
                state=clean_state,
                level=clean_level,
                category=clean_category,
                top_k=effective_candidate_k,
                candidate_k=effective_candidate_k,
                rrf_k=self.rrf_k,
                query_vector=query_vector,
            )
        except Exception as e:
            # Differentiate serious database infrastructure errors from API rate-limit errors
            err_str = str(e).lower()
            if any(term in err_str for term in ["psycopg", "connection", "operationalerror", "timeout", "refused"]):
                raise RuntimeError("Database connection failure during retrieval execution.") from None

            # If Gemini API quota (429) or embedding generation failed, gracefully degrade to keyword retrieval
            if any(term in err_str for term in ["quota", "429", "resource_exhausted", "embed", "clienterror"]):
                candidate_chunks = self.hybrid_retriever.keyword_retriever.retrieve(
                    query=hybrid_search_query,
                    state=clean_state,
                    level=clean_level,
                    category=clean_category,
                    top_k=effective_candidate_k,
                )
            else:
                # Re-raise unexpected internal errors
                raise

        # 5. Prioritize Authoritative Resolved Chunks at the front of candidate set
        if resolved_chunks:
            existing_chunk_ids = {c.chunk_id for c in resolved_chunks}
            for c in candidate_chunks:
                if c.chunk_id not in existing_chunk_ids:
                    resolved_chunks.append(c)
            candidate_chunks = resolved_chunks

        # 6. Scheme-Level Consolidation and Deterministic Reranking
        scheme_results = self.scheme_reranker.rerank(
            results=candidate_chunks,
            top_k=top_k,
        )

        # 5. Build and return structured RetrievalResponse
        return RetrievalResponse(
            query=clean_query,
            results=scheme_results,
            total_results=len(scheme_results),
            retrieval_method="hybrid_rrf_scheme_reranked",
            filters=active_filters,
            coverage_info=self.get_coverage_info(),
        )


def retrieve_service(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: Optional[int] = None,
    state: Optional[str] = None,
    level: Optional[str] = None,
    category: Optional[str] = None,
    query_vector: Optional[List[float]] = None,
    engine: Optional[Engine] = None,
    client: Optional[Any] = None,
) -> RetrievalResponse:
    """
    Convenience function to execute retrieval via RetrievalService.
    """
    service = RetrievalService(engine_or_url=engine, client=client)
    return service.retrieve(
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        state=state,
        level=level,
        category=category,
        query_vector=query_vector,
    )
