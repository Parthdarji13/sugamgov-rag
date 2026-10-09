"""
scraper/embedder.py
====================
Chunk & Embed Pipeline for SugamGov Live Scheme Scraper.

Reuses EXACTLY the same:
  - Text chunking logic as scripts/06_chunk_schemes.py
  - Embedding model (intfloat/multilingual-e5-small, 384-dim)
  - 'passage: ' prefix for document embeddings
  - chunk_id format: {scheme_id}_{field_name}_{chunk_index}
  - metadata JSONB structure

This guarantees that new scraped chunks live in the same embedding
space as your 20,497 existing chunks — vector search works on both.
"""

import re
import json
import hashlib
import logging
import math
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("sugamgov_scraper.embedder")

# ── Chunking constants (identical to scripts/06_chunk_schemes.py) ──────────
CHUNK_TARGET_WORDS = 600
CHUNK_OVERLAP_WORDS = 60

TEXT_FIELDS = [
    "scheme_name",
    "details",
    "benefits",
    "eligibility",
    "application",
    "documents",
]


# ── Text Chunker (copied from scripts/06_chunk_schemes.py) ─────────────────

def _split_text_into_chunks(text_content: str, max_words: int = CHUNK_TARGET_WORDS,
                            overlap_words: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    """Deterministic sentence-aware text chunker with overlap."""
    words = text_content.split()
    if len(words) <= max_words:
        return [text_content]

    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_content) if s.strip()]

    units = []
    for s in raw_sentences:
        s_words = s.split()
        if len(s_words) > max_words:
            step = max_words - overlap_words if max_words > overlap_words else max_words
            for i in range(0, len(s_words), step):
                units.append(" ".join(s_words[i:i + max_words]))
        else:
            units.append(s)

    chunks, current_units, current_word_count = [], [], 0
    for unit in units:
        unit_words = len(unit.split())
        if current_word_count + unit_words > max_words and current_units:
            chunks.append(" ".join(current_units))
            overlap_units, overlap_count = [], 0
            for u in reversed(current_units):
                u_len = len(u.split())
                if overlap_count + u_len <= overlap_words:
                    overlap_units.insert(0, u)
                    overlap_count += u_len
                else:
                    break
            current_units = overlap_units + [unit]
            current_word_count = overlap_count + unit_words
        else:
            current_units.append(unit)
            current_word_count += unit_words

    if current_units:
        chunks.append(" ".join(current_units))
    return chunks


def build_chunks_for_scheme(scheme: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts a scheme dict into retrieval-ready chunk dicts.
    Matches the exact format produced by scripts/06_chunk_schemes.py.

    Each chunk dict has:
      scheme_id, chunk_id, field_name, chunk_text, chunk_index, metadata (JSON str)
    """
    scheme_id   = str(scheme.get("scheme_id", "")).strip()
    scheme_name = str(scheme.get("scheme_name", "")).strip()
    level       = scheme.get("level") or None
    state       = scheme.get("state") or None
    states      = scheme.get("states") or []
    categories  = scheme.get("categories") or []

    chunks: List[Dict[str, Any]] = []

    def _make_metadata(field_name: str, chunk_index: int, total: int, text: str) -> str:
        meta = {
            "scheme_id":          scheme_id,
            "scheme_name":        scheme_name,
            "level":              level,
            "state":              state,
            "states":             states,
            "categories":         categories,
            "source_field":       field_name,
            "chunk_index":        chunk_index,
            "total_field_chunks": total,
            "word_count":         len(text.split()),
            "char_count":         len(text),
            "data_source":        scheme.get("data_source", "live_scraper"),
        }
        return json.dumps(meta, ensure_ascii=False)

    # ── 1. scheme_name identity chunk ──────────────────────────────────────
    name_lines = [f"Scheme: {scheme_name}"]
    loc_parts = []
    if level:
        loc_parts.append(f"Level: {level}")
    if state:
        loc_parts.append(f"State: {state}")
    elif states:
        loc_parts.append(f"States: {', '.join(states[:5])}")
    if loc_parts:
        name_lines.append(" | ".join(loc_parts))
    if categories:
        name_lines.append(f"Categories: {', '.join(categories)}")

    name_text = "\n".join(name_lines)
    chunks.append({
        "scheme_id":   scheme_id,
        "chunk_id":    f"{scheme_id}_scheme_name_0",
        "field_name":  "scheme_name",
        "chunk_text":  name_text,
        "chunk_index": 0,
        "metadata":    _make_metadata("scheme_name", 0, 1, name_text),
    })

    # ── 2. Content fields ──────────────────────────────────────────────────
    for field_name in ["details", "benefits", "eligibility", "application", "documents"]:
        raw_text = str(scheme.get(field_name) or "").strip()
        if not raw_text or len(raw_text.split()) < 5:
            continue

        # Build contextual prefix (same format as scripts/06_chunk_schemes.py)
        prefix_lines = [
            f"Scheme: {scheme_name}",
            f"Section: {field_name.replace('_', ' ').title()}",
        ]
        if level:
            prefix_lines.append(f"Level: {level}")
        if state:
            prefix_lines.append(f"State: {state}")
        prefix = "\n".join(prefix_lines)

        sub_chunks = _split_text_into_chunks(raw_text)
        total = len(sub_chunks)

        for idx, sub in enumerate(sub_chunks):
            full_text = f"{prefix}\n\n{sub}"
            chunk_id  = f"{scheme_id}_{field_name}_{idx}"
            chunks.append({
                "scheme_id":   scheme_id,
                "chunk_id":    chunk_id,
                "field_name":  field_name,
                "chunk_text":  full_text,
                "chunk_index": idx,
                "metadata":    _make_metadata(field_name, idx, total, full_text),
            })

    return chunks


# ── Embedding ───────────────────────────────────────────────────────────────

def _get_model() -> Any:
    """Returns the cached SentenceTransformer model (same as RAG server uses)."""
    # Import at call time to avoid loading model unless needed
    from src.retrieval.vector_retriever import get_embedding_model
    return get_embedding_model()


def embed_chunks(chunks: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
    """
    Adds embedding_local (384-dim list of floats) to each chunk dict.

    Uses the exact same model and 'passage: ' prefix as the existing
    18_generate_local_embeddings.py script.

    Args:
        chunks: List of chunk dicts from build_chunks_for_scheme().
        batch_size: Number of chunks to embed in one model call.

    Returns:
        Same list with 'embedding_local' key added to each chunk.
        Chunks that fail to embed are returned with embedding_local = None.
    """
    if not chunks:
        return chunks

    model = _get_model()
    texts = [f"passage: {c['chunk_text']}" for c in chunks]

    logger.info("Embedding %d chunks in batches of %d ...", len(chunks), batch_size)
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            embeddings = model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for emb in embeddings:
                values = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                # Validate dimension and finiteness
                if len(values) == 384 and all(math.isfinite(v) for v in values):
                    all_embeddings.append([float(v) for v in values])
                else:
                    logger.warning("Bad embedding (dim=%d) — storing NULL", len(values))
                    all_embeddings.append(None)
        except Exception as e:
            logger.error("Embedding batch %d-%d failed: %s", i, i + batch_size, e)
            all_embeddings.extend([None] * len(batch_texts))

    # Attach embeddings to chunk dicts
    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding_local"] = emb

    embedded_count = sum(1 for c in chunks if c.get("embedding_local") is not None)
    logger.info("Embedding complete: %d/%d chunks embedded", embedded_count, len(chunks))
    return chunks


def process_schemes(
    schemes: List[Dict[str, Any]],
    embed: bool = True,
) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Full chunk + embed pipeline for a list of schemes.

    Args:
        schemes: List of scheme dicts (new or updated).
        embed: If True, generate embeddings. If False, skip (dry-run mode).

    Returns:
        List of (scheme_dict, [chunk_dict, ...]) tuples.
    """
    output = []

    for scheme in schemes:
        sid = scheme.get("scheme_id", "?")
        try:
            chunks = build_chunks_for_scheme(scheme)
            if not chunks:
                logger.warning("No chunks generated for scheme %s", sid)
                continue

            if embed:
                chunks = embed_chunks(chunks)

            output.append((scheme, chunks))
            logger.debug(
                "Processed scheme %s: %d chunks generated",
                sid, len(chunks),
            )
        except Exception as e:
            logger.error("Failed to process scheme %s: %s", sid, e, exc_info=True)

    return output
