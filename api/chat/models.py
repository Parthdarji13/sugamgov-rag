"""
api/chat/models.py
==================
Data models and Pydantic schemas for the Conversational Chat API Layer.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from api.schemas import SUPPORTED_LANGUAGES, EvidenceCitationSchema


# ============================================================================
# In-Memory Session Dataclasses
# ============================================================================

@dataclass
class ChatMessage:
    """Represents a single message in the conversational history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class SessionData:
    """Represents a conversation session with bounded message history."""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ============================================================================
# API Request & Response Schemas
# ============================================================================

class ChatRequest(BaseModel):
    """
    Request payload for multi-turn conversational chat endpoint (POST /api/chat).
    """
    message: str = Field(
        ...,
        description="User message or question (1 to 2,000 characters).",
        min_length=1,
        max_length=2000,
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for multi-turn conversation. If omitted, a new session is created.",
    )
    language: Optional[str] = Field(
        default="auto",
        description="Language code: 'auto' (detect), 'en' (English), 'hi' (Hindi), or 'gu' (Gujarati).",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of unique schemes to retrieve (1 to 10).",
    )
    candidate_k: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Candidate chunks retrieved per modality before consolidation (must be >= top_k).",
    )
    state: Optional[str] = Field(
        default=None,
        description="Optional state filter (e.g. 'Gujarat', 'Maharashtra').",
    )
    level: Optional[str] = Field(
        default=None,
        description="Optional administrative level filter ('Central' or 'State').",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional sector category filter (e.g. 'Agriculture', 'Education & Learning').",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("message cannot be empty or whitespace-only")
        if len(trimmed) > 2000:
            raise ValueError("message exceeds maximum limit of 2,000 characters")
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

    @field_validator("state", "level", "category", "session_id")
    @classmethod
    def sanitize_strings(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        trimmed = v.strip()
        return trimmed if trimmed else None

    @model_validator(mode="after")
    def validate_k_relationship(self) -> "ChatRequest":
        if self.candidate_k < self.top_k:
            raise ValueError(
                f"candidate_k ({self.candidate_k}) must be greater than or equal to top_k ({self.top_k})"
            )
        return self


class ChatResponse(BaseModel):
    """
    Response schema for POST /api/chat.
    Returns the session ID, strictly grounded answer, and evidence citations.
    """
    session_id: str
    message: str
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


class DeleteSessionResponse(BaseModel):
    """Response schema for DELETE /api/chat/{session_id}."""
    status: str = Field(default="deleted", description="Operation status.")
    session_id: str = Field(..., description="Deleted session ID.")
