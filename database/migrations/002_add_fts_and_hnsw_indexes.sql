-- =============================================================================
-- Migration 002: Add Persisted FTS Search Vector & HNSW Vector Index
-- =============================================================================
-- Description:
--   1. Adds generated tsvector column 'search_vector' on scheme_chunks combining
--      scheme_name (weight 'A') from metadata and chunk_text (weight 'B').
--   2. Creates GIN index on 'search_vector' for sub-millisecond lexical FTS.
--   3. Creates HNSW index on 'embedding_local' (vector_cosine_ops) for fast
--      approximate nearest neighbor dense vector search.
-- =============================================================================

-- PART A: Add generated tsvector column on scheme_chunks
ALTER TABLE scheme_chunks
ADD COLUMN IF NOT EXISTS search_vector tsvector
GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(metadata->>'scheme_name', '')), 'A') ||
    setweight(to_tsvector('english', chunk_text), 'B')
) STORED;

-- PART B: Create GIN index on search_vector
CREATE INDEX IF NOT EXISTS idx_scheme_chunks_search_vector
ON scheme_chunks
USING GIN (search_vector);

-- PART C: Create HNSW index on embedding_local (cosine distance)
CREATE INDEX IF NOT EXISTS idx_scheme_chunks_embedding_local_hnsw
ON scheme_chunks
USING hnsw (embedding_local vector_cosine_ops);
