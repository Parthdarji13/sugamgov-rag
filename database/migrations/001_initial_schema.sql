-- =============================================================================
-- Migration 001: Initial Schema for sugamgov
-- =============================================================================
-- Description:
--   1. Enables pgvector extension.
--   2. Creates core tables: schemes, sources, scheme_versions, scheme_chunks.
--   3. Sets up foreign keys, relational indexes, and automated updated_at triggers.
-- =============================================================================

-- PART A: Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- PART B: Main schemes table
CREATE TABLE IF NOT EXISTS schemes (
    id                          BIGSERIAL PRIMARY KEY,
    scheme_id                   VARCHAR(20) NOT NULL UNIQUE,
    scheme_name                 TEXT NOT NULL,
    slug                        TEXT,
    details                     TEXT,
    benefits                    TEXT,
    eligibility                 TEXT,
    application                 TEXT,
    documents                   TEXT,
    level                       VARCHAR(50),
    scheme_category             TEXT,
    tags                        TEXT,
    categories                  JSONB,
    normalized_tags             JSONB,
    state                       VARCHAR(100),
    states                      JSONB,
    state_extraction_method     VARCHAR(50),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- PART C: Sources and verification metadata table
CREATE TABLE IF NOT EXISTS sources (
    id                          BIGSERIAL PRIMARY KEY,
    scheme_id                   VARCHAR(20) NOT NULL,
    source_name                 TEXT NOT NULL,
    source_url                  TEXT,
    source_type                 VARCHAR(50) NOT NULL,
    is_official                 BOOLEAN NOT NULL DEFAULT FALSE,
    last_checked                TIMESTAMPTZ,
    last_verified               TIMESTAMPTZ,
    verification_status         VARCHAR(50),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_sources_scheme_id
        FOREIGN KEY (scheme_id)
        REFERENCES schemes(scheme_id)
        ON DELETE CASCADE
);

-- PART D: Historical scheme snapshots and versions
CREATE TABLE IF NOT EXISTS scheme_versions (
    id                          BIGSERIAL PRIMARY KEY,
    scheme_id                   VARCHAR(20) NOT NULL,
    version_number              INTEGER NOT NULL,
    scheme_snapshot             JSONB NOT NULL,
    source_id                   BIGINT,
    change_summary              TEXT,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_versions_scheme_id
        FOREIGN KEY (scheme_id)
        REFERENCES schemes(scheme_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_versions_source_id
        FOREIGN KEY (source_id)
        REFERENCES sources(id)
        ON DELETE SET NULL,
    CONSTRAINT uq_scheme_version
        UNIQUE(scheme_id, version_number)
);

-- PART E: Text chunks and vector embeddings (unspecified vector dimension for future model)
CREATE TABLE IF NOT EXISTS scheme_chunks (
    id                          BIGSERIAL PRIMARY KEY,
    scheme_id                   VARCHAR(20) NOT NULL,
    chunk_id                    VARCHAR(100) NOT NULL UNIQUE,
    field_name                  VARCHAR(50) NOT NULL,
    chunk_text                  TEXT NOT NULL,
    chunk_index                 INTEGER NOT NULL,
    embedding                   vector,
    metadata                    JSONB,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_chunks_scheme_id
        FOREIGN KEY (scheme_id)
        REFERENCES schemes(scheme_id)
        ON DELETE CASCADE
);

-- PART F: Indexes
-- Schemes indexes
CREATE INDEX IF NOT EXISTS idx_schemes_scheme_id ON schemes(scheme_id);
CREATE INDEX IF NOT EXISTS idx_schemes_slug ON schemes(slug);
CREATE INDEX IF NOT EXISTS idx_schemes_scheme_name ON schemes(scheme_name);
CREATE INDEX IF NOT EXISTS idx_schemes_level ON schemes(level);
CREATE INDEX IF NOT EXISTS idx_schemes_state ON schemes(state);
CREATE INDEX IF NOT EXISTS idx_schemes_scheme_category ON schemes(scheme_category);

-- Sources indexes
CREATE INDEX IF NOT EXISTS idx_sources_scheme_id ON sources(scheme_id);
CREATE INDEX IF NOT EXISTS idx_sources_source_url ON sources(source_url);
CREATE INDEX IF NOT EXISTS idx_sources_verification_status ON sources(verification_status);

-- Scheme chunks indexes
CREATE INDEX IF NOT EXISTS idx_scheme_chunks_scheme_id ON scheme_chunks(scheme_id);
CREATE INDEX IF NOT EXISTS idx_scheme_chunks_field_name ON scheme_chunks(field_name);

-- PART G: Automated updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_schemes_updated_at ON schemes;
CREATE TRIGGER trg_schemes_updated_at
    BEFORE UPDATE ON schemes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trg_sources_updated_at ON sources;
CREATE TRIGGER trg_sources_updated_at
    BEFORE UPDATE ON sources
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
