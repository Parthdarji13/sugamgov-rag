"""
src/retrieval/scheme_reranker.py
================================
Post-Retrieval Scheme-Level Reranking and Chunk Consolidation Layer for SugamGov AI RAG.

Architecture & Role in SugamGov:
--------------------------------
1. What Scheme Reranking & Consolidation Does:
   Raw retrieval pipelines (lexical FTS, dense vector search, or hybrid RRF) operate
   at the granular chunk level (e.g. S0126_scheme_name_0, S0126_details_0, S0126_benefits_0).
   If unmanaged, a single highly-scoring scheme with multiple matched chunks can dominate
   the top-k retrieved list, pushing out distinct, highly relevant government schemes.

   This module:
     - Accepts chunk-level RetrievalResult objects from any retriever (Hybrid, Keyword, Vector).
     - Consolidates retrieved chunks by unique scheme_id.
     - Preserves the highest-ranked chunk as the representative snippet for the scheme.
     - Retains all matching chunks as supporting `evidence_chunks`.
     - Produces unique, ranked `SchemeRetrievalResult` objects enforcing top_k.

2. Ranking Logic:
   Transparent, deterministic 5-tier ranking strategy:
     - Tier 1 (Primary): Best chunk retrieval score (or RRF score) descending.
     - Tier 2 (Secondary A): Multi-modal consensus (schemes with evidence in BOTH keyword
       and vector retrieval rank ahead of single-modality schemes when scores tie).
     - Tier 3 (Secondary B): Best keyword rank across evidence chunks ascending (lower is better).
     - Tier 4 (Secondary C): Best vector rank across evidence chunks ascending (lower is better).
     - Tier 5 (Tie-breaker): Lexicographical scheme_id ascending ("S0001" < "S0002").

3. Field Priority:
   Configurable, but by default None (0 boost) to strictly preserve the underlying
   empirical retrieval score.

4. Non-Machine-Learned Transparency:
   This is a deterministic consolidation and rank-fusion post-processor, NOT a trained
   or black-box machine-learning reranker.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class SchemeRetrievalResult:
    """
    Consolidated, scheme-level retrieval result.

    Fields:
      scheme_id: Unique identifier of the scheme (e.g. 'S0126')
      scheme_name: Official name of the scheme
      level: 'Central' or 'State'
      state: State name or None for Central schemes
      categories: List of scheme category strings
      best_chunk_id: chunk_id of the highest-ranked representative chunk
      best_field_name: originating field of the representative chunk (e.g. 'scheme_name')
      best_chunk_text: text content of the representative chunk
      best_retrieval_score: highest retrieval score (or RRF score) among the scheme's chunks
      keyword_rank: best (lowest) keyword rank across retrieved evidence chunks
      vector_rank: best (lowest) vector rank across retrieved evidence chunks
      rrf_score: best RRF score if hybrid retrieval was used
      evidence_chunks: all retrieved chunk-level results belonging to this scheme
    """
    scheme_id: str
    scheme_name: str
    level: Optional[str]
    state: Optional[str]
    categories: List[str]
    best_chunk_id: str
    best_field_name: str
    best_chunk_text: str
    best_retrieval_score: float
    keyword_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    evidence_chunks: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    states: List[str] = field(default_factory=list)

    @property
    def has_both_modalities(self) -> bool:
        """True if the scheme has evidence from both keyword and vector modalities."""
        return self.keyword_rank is not None and self.vector_rank is not None

    def to_dict(self, include_evidence: bool = True) -> Dict[str, Any]:
        """Converts the scheme retrieval result to a JSON-serializable dictionary."""
        d = {
            "scheme_id": self.scheme_id,
            "scheme_name": self.scheme_name,
            "level": self.level,
            "state": self.state,
            "states": self.states,
            "categories": self.categories,
            "best_chunk_id": self.best_chunk_id,
            "best_field_name": self.best_field_name,
            "best_chunk_text": self.best_chunk_text,
            "best_retrieval_score": round(self.best_retrieval_score, 6),
            "evidence_count": len(self.evidence_chunks),
            "has_both_modalities": self.has_both_modalities,
            "metadata": self.metadata,
        }
        if self.keyword_rank is not None:
            d["keyword_rank"] = self.keyword_rank
        if self.vector_rank is not None:
            d["vector_rank"] = self.vector_rank
        if self.rrf_score is not None:
            d["rrf_score"] = round(self.rrf_score, 6)
        if include_evidence:
            d["evidence_chunks"] = [
                c.to_dict() if hasattr(c, "to_dict") else str(c)
                for c in self.evidence_chunks
            ]
        return d

    def preview(self, max_length: int = 140) -> str:
        """Returns a single-line preview of best_chunk_text truncated to max_length."""
        cleaned = " ".join(self.best_chunk_text.split())
        if len(cleaned) <= max_length:
            return cleaned
        return cleaned[:max_length - 3] + "..."


class SchemeReranker:
    """
    Consolidates chunk-level retrieval results by scheme_id and performs
    deterministic multi-tier ranking.
    """
    def __init__(self, field_priority: Optional[Dict[str, int]] = None):
        """
        Initializes the scheme reranker.

        Args:
            field_priority: Optional dictionary mapping field_name to integer priority
                           (e.g. {'scheme_name': 1, 'details': 2}).
                           Defaults to None (preserving raw retrieval score ranking).
        """
        self.field_priority = field_priority or {}

    def rerank(
        self,
        results: List[Any],
        top_k: int = 5,
    ) -> List[SchemeRetrievalResult]:
        """
        Consolidates chunk-level results into unique scheme-level results.

        Args:
            results: List of chunk-level RetrievalResult objects.
            top_k: Number of unique schemes to return (must be >= 1).

        Returns:
            List of SchemeRetrievalResult objects, strictly unique by scheme_id,
            ordered deterministically.
        """
        if not results:
            return []

        top_k = max(1, top_k)

        # 1. Group chunks by scheme_id preserving incoming order
        scheme_groups: Dict[str, List[Any]] = {}
        for r in results:
            sid = r.scheme_id
            if sid not in scheme_groups:
                scheme_groups[sid] = []
            scheme_groups[sid].append(r)

        # 2. Consolidate each scheme group
        consolidated: List[SchemeRetrievalResult] = []

        for sid, chunks in scheme_groups.items():
            # Determine best representative chunk
            if self.field_priority:
                # If field priority configured, sort primarily by score descending,
                # secondarily by field priority ascending, tertiarily by chunk_id
                best_chunk = min(
                    chunks,
                    key=lambda c: (
                        -round(c.score, 6),
                        self.field_priority.get(c.field_name, 999),
                        c.chunk_id,
                    ),
                )
            else:
                # Default: the highest-scoring chunk (first in chunk order if tied)
                best_chunk = max(chunks, key=lambda c: c.score)

            # Determine best keyword rank across all chunks for this scheme
            kw_ranks = [c.keyword_rank for c in chunks if c.keyword_rank is not None]
            best_kw_rank = min(kw_ranks) if kw_ranks else None

            # Determine best vector rank across all chunks for this scheme
            vec_ranks = [c.vector_rank for c in chunks if c.vector_rank is not None]
            best_vec_rank = min(vec_ranks) if vec_ranks else None

            # RRF score if available from hybrid retrieval
            rrf_score = best_chunk.score if (best_chunk.keyword_rank is not None or best_chunk.vector_rank is not None) else None

            meta = dict(best_chunk.metadata) if hasattr(best_chunk, "metadata") and best_chunk.metadata else {}
            states_list = list(meta.get("states", []))

            consolidated.append(
                SchemeRetrievalResult(
                    scheme_id=sid,
                    scheme_name=best_chunk.scheme_name,
                    level=best_chunk.level,
                    state=best_chunk.state,
                    categories=list(best_chunk.categories) if best_chunk.categories else [],
                    best_chunk_id=best_chunk.chunk_id,
                    best_field_name=best_chunk.field_name,
                    best_chunk_text=best_chunk.chunk_text,
                    best_retrieval_score=float(best_chunk.score),
                    keyword_rank=best_kw_rank,
                    vector_rank=best_vec_rank,
                    rrf_score=rrf_score,
                    evidence_chunks=chunks,
                    metadata=meta,
                    states=states_list,
                )
            )

        # 3. Deterministic Multi-Tier Ranking
        # Tier 1 (Primary): best_retrieval_score descending
        # Tier 2: has_both_modalities (True before False)
        # Tier 3: best_kw_rank ascending (None treated as 999,999)
        # Tier 4: best_vec_rank ascending (None treated as 999,999)
        # Tier 5: scheme_id ascending (lexicographical tie-breaker)
        def scheme_sort_key(s: SchemeRetrievalResult):
            return (
                -round(s.best_retrieval_score, 6),
                0 if s.has_both_modalities else 1,
                s.keyword_rank if s.keyword_rank is not None else 999999,
                s.vector_rank if s.vector_rank is not None else 999999,
                s.scheme_id,
            )

        ranked_schemes = sorted(consolidated, key=scheme_sort_key)

        return ranked_schemes[:top_k]


def rerank_schemes(
    results: List[Any],
    top_k: int = 5,
    field_priority: Optional[Dict[str, int]] = None,
) -> List[SchemeRetrievalResult]:
    """
    Convenience function for scheme-level reranking and chunk consolidation.

    Args:
        results: List of chunk-level RetrievalResult objects.
        top_k: Number of unique schemes to return (default: 5).
        field_priority: Optional field priority dict (default: None).

    Returns:
        List of unique, consolidated SchemeRetrievalResult objects.
    """
    reranker = SchemeReranker(field_priority=field_priority)
    return reranker.rerank(results=results, top_k=top_k)
