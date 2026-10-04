"""
api/schemas.py
==============
Pydantic Request and Response Schemas for SugamGov AI RAG API.

Enforces strict input validation:
- Non-empty, whitespace-trimmed queries.
- Supported languages: 'auto', 'en', 'hi', 'gu'.
- Numerical bounds: top_k in [1, 10], candidate_k in [top_k, 100].
- Reusable, structured output models for retrieval and generation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

SUPPORTED_LANGUAGES = {"auto", "en", "hi", "gu"}


# ============================================================================
# Request Schemas
# ============================================================================

class RetrievalChatRequest(BaseModel):
    """
    Standard request schema for both retrieval and grounded generation endpoints.
    """
    query: str = Field(
        ...,
        description="Search query or question regarding government schemes (cannot be empty).",
        min_length=1,
    )
    language: Optional[str] = Field(
        default="auto",
        description="Language code: 'auto' (auto-detect), 'en' (English), 'hi' (Hindi), or 'gu' (Gujarati).",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of unique schemes to return (1 to 10).",
    )
    candidate_k: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Candidate chunks retrieved per modality before deduplication (must be >= top_k).",
    )
    state: Optional[str] = Field(
        default=None,
        description="Optional state filter (e.g. 'Gujarat', 'Madhya Pradesh').",
    )
    level: Optional[str] = Field(
        default=None,
        description="Optional administrative level filter: 'Central' or 'State'.",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional sector category filter (e.g. 'Agriculture', 'Education & Learning').",
    )

    @field_validator("query")
    @classmethod
    def validate_and_trim_query(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("query cannot be empty or whitespace-only")
        return trimmed

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> str:
        if v is None:
            return "auto"
        clean = v.strip().lower()
        if clean not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{clean}'. Must be one of: {sorted(list(SUPPORTED_LANGUAGES))}"
            )
        return clean

    @field_validator("state", "level", "category")
    @classmethod
    def sanitize_filter_string(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        trimmed = v.strip()
        return trimmed if trimmed else None

    @model_validator(mode="after")
    def validate_k_relationship(self) -> "RetrievalChatRequest":
        if self.candidate_k < self.top_k:
            raise ValueError(
                f"candidate_k ({self.candidate_k}) must be greater than or equal to top_k ({self.top_k})"
            )
        return self


# ============================================================================
# Response Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Safe health check response containing no credentials or system paths."""
    status: str = Field(default="ok", description="Operational status.")
    service: str = Field(default="sugamgov-rag-api", description="Service identifier.")


class EvidenceChunkSchema(BaseModel):
    """Metadata and content for an individual retrieved evidence chunk."""
    chunk_id: str
    field_name: str
    chunk_text: str
    score: float
    keyword_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    keyword_score: Optional[float] = None
    vector_score: Optional[float] = None


class SchemeResultSchema(BaseModel):
    """Consolidated scheme result containing metadata and supporting evidence chunks."""
    scheme_id: str
    scheme_name: str
    level: Optional[str] = None
    state: Optional[str] = None
    states: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    score: float
    best_chunk_id: str
    best_field: str
    evidence_chunks: List[EvidenceChunkSchema] = Field(default_factory=list)


class RetrieveResponse(BaseModel):
    """Response returned by the /api/retrieve endpoint."""
    query: str
    results: List[SchemeResultSchema]
    total_results: int
    retrieval_method: str
    filters: Dict[str, Optional[str]]
    coverage_info: Dict[str, Any]


class EvidenceCitationSchema(BaseModel):
    """Granular evidence citation connecting answer claims to database chunk IDs."""
    scheme_id: str
    scheme_name: str
    chunk_id: str
    field_name: str


class ChatResponse(BaseModel):
    """
    Response returned by the /api/generate endpoint.
    Exposes the strictly grounded answer with citations and confidence metadata.
    """
    query: str
    answer: str
    language: str
    grounded: bool
    confidence: str
    schemes: List[str]
    evidence_used: List[EvidenceCitationSchema]
    limitations: Optional[str] = None
    retrieval_method: str
    total_results: int
    coverage_info: Dict[str, Any]


# Alias for backward-compatibility and expressive naming
GenerateResponse = ChatResponse
