"""
src/generation
==============
Grounded RAG Answer Generation Package for SugamGov AI.
Provides strict grounding prompts, data models, and LLM answer generation.
"""

from src.generation.models import RAGAnswer, EvidenceCitation
from src.generation.prompt import (
    SYSTEM_INSTRUCTION,
    detect_language,
    build_grounded_prompt,
)
from src.generation.rag_generator import (
    RAGGenerator,
    DEFAULT_GENERATION_MODEL,
)

__all__ = [
    "RAGAnswer",
    "EvidenceCitation",
    "SYSTEM_INSTRUCTION",
    "detect_language",
    "build_grounded_prompt",
    "RAGGenerator",
    "DEFAULT_GENERATION_MODEL",
]
