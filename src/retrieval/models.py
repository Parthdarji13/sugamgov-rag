"""
src/retrieval/models.py
=======================
Structured data models for SugamGov retrieval results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class RetrievalResult:
    """
    Structured retrieval result representing a matched scheme chunk.
    
    Fields:
      chunk_id: Unique identifier of the chunk (e.g. 'S0001_scheme_name_0')
      scheme_id: Parent scheme ID (e.g. 'S0001')
      scheme_name: Official name of the scheme
      field_name: The originating scheme field (e.g. 'scheme_name', 'eligibility', 'benefits')
      chunk_text: Complete text content of the chunk
      score: Deterministic relevance score from PostgreSQL ts_rank_cd
      metadata: Associated chunk JSONB metadata (tags, char count, etc.)
      state: State name or None for Central schemes
      level: 'Central' or 'State'
      categories: List of scheme category strings
    """
    chunk_id: str
    scheme_id: str
    scheme_name: str
    field_name: str
    chunk_text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: Optional[str] = None
    level: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    keyword_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    keyword_score: Optional[float] = None
    vector_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts the retrieval result to a JSON-serializable dictionary."""
        d = {
            "chunk_id": self.chunk_id,
            "scheme_id": self.scheme_id,
            "scheme_name": self.scheme_name,
            "field_name": self.field_name,
            "chunk_text": self.chunk_text,
            "score": round(self.score, 6),
            "metadata": self.metadata,
            "state": self.state,
            "level": self.level,
            "categories": self.categories,
        }
        if self.keyword_rank is not None:
            d["keyword_rank"] = self.keyword_rank
        if self.vector_rank is not None:
            d["vector_rank"] = self.vector_rank
        if self.keyword_score is not None:
            d["keyword_score"] = round(self.keyword_score, 4)
        if self.vector_score is not None:
            d["vector_score"] = round(self.vector_score, 4)
        return d

    def preview(self, max_length: int = 140) -> str:
        """Returns a single-line preview of chunk_text truncated to max_length."""
        cleaned = " ".join(self.chunk_text.split())
        if len(cleaned) <= max_length:
            return cleaned
        return cleaned[:max_length - 3] + "..."
