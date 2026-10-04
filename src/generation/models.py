"""
src/generation/models.py
========================
Data models for SugamGov AI Grounded RAG Generation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class EvidenceCitation:
    """
    Structured citation reference pointing to specific retrieved evidence.

    Fields:
      scheme_id: Canonical scheme ID (e.g. 'S0126')
      scheme_name: Official scheme name
      chunk_id: Specific retrieved chunk ID (e.g. 'S0126_eligibility_0')
      field_name: Originating field ('eligibility', 'benefits', etc.)
    """
    scheme_id: str
    scheme_name: str
    chunk_id: str
    field_name: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts citation to a JSON-serializable dictionary."""
        return {
            "scheme_id": self.scheme_id,
            "scheme_name": self.scheme_name,
            "chunk_id": self.chunk_id,
            "field_name": self.field_name,
        }


@dataclass
class RAGAnswer:
    """
    Structured, grounded response generated from retrieved scheme evidence.

    Fields:
      query: User search query or question
      answer: Factual answer strictly grounded in retrieved evidence
      language: Response language code ('en', 'hi', 'gu')
      grounded: True if answer is grounded in retrieved evidence, False if insufficient
      confidence: Categorical indicator ('high', 'medium', 'low') based on evidence depth
      schemes: List of scheme summary dicts supporting the answer
      evidence_used: List of structured evidence citations supporting the answer
      limitations: Explicit disclaimer or limitations regarding data completeness
    """
    query: str
    answer: str
    language: str
    grounded: bool
    confidence: str
    schemes: List[Dict[str, Any]] = field(default_factory=list)
    evidence_used: List[Dict[str, Any]] = field(default_factory=list)
    limitations: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts RAGAnswer to a JSON-serializable dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "language": self.language,
            "grounded": self.grounded,
            "confidence": self.confidence,
            "schemes": self.schemes,
            "evidence_used": self.evidence_used,
            "limitations": self.limitations,
        }

    def preview(self, max_length: int = 140) -> str:
        """Returns a single-line summary preview."""
        cleaned = " ".join(self.answer.split())
        truncated_ans = cleaned if len(cleaned) <= max_length else cleaned[:max_length - 3] + "..."
        return (
            f"RAGAnswer(lang='{self.language}', grounded={self.grounded}, "
            f"confidence='{self.confidence}', schemes={len(self.schemes)}, "
            f"answer=\"{truncated_ans}\")"
        )
