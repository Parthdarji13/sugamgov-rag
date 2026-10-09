-- =============================================================================
-- Migration 003: Add Scraper Tracking Columns
-- =============================================================================
-- Adds columns needed by the daily web scraper pipeline.
-- Safe to run multiple times (uses IF NOT EXISTS / DO $$ blocks).
-- Does NOT change or break any existing data or indexes.
-- =============================================================================

-- Add data_source: tracks where the row came from ('initial_dump' or 'live_scraper')
ALTER TABLE schemes
    ADD COLUMN IF NOT EXISTS data_source VARCHAR(50) DEFAULT 'initial_dump';

-- Add source_url: the URL the scraper fetched this scheme from
ALTER TABLE schemes
    ADD COLUMN IF NOT EXISTS source_url TEXT;

-- Add last_scraped_at: when the scraper last checked this scheme
ALTER TABLE schemes
    ADD COLUMN IF NOT EXISTS last_scraped_at TIMESTAMPTZ;

-- Add scrape_checksum: SHA-256 hash of the scraped content fields.
-- If checksum matches DB, the scheme is unchanged and skipped (saves compute).
ALTER TABLE schemes
    ADD COLUMN IF NOT EXISTS scrape_checksum VARCHAR(64);

-- Index for efficient "find schemes by data source" queries
CREATE INDEX IF NOT EXISTS idx_schemes_data_source ON schemes(data_source);

-- Index for finding stale schemes (not scraped in > N days)
CREATE INDEX IF NOT EXISTS idx_schemes_last_scraped_at ON schemes(last_scraped_at);

-- =============================================================================
-- Scrape run log table: one row per daily run, for monitoring
-- =============================================================================
CREATE TABLE IF NOT EXISTS scrape_runs (
    id              BIGSERIAL PRIMARY KEY,
    run_date        DATE NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    source          VARCHAR(100) NOT NULL,
    schemes_found   INTEGER DEFAULT 0,
    schemes_added   INTEGER DEFAULT 0,
    schemes_updated INTEGER DEFAULT 0,
    schemes_skipped INTEGER DEFAULT 0,
    errors          INTEGER DEFAULT 0,
    status          VARCHAR(20) DEFAULT 'running',  -- 'running', 'success', 'failed'
    error_message   TEXT,
    CONSTRAINT uq_scrape_run_date_source UNIQUE (run_date, source)
);

CREATE INDEX IF NOT EXISTS idx_scrape_runs_run_date ON scrape_runs(run_date);
CREATE INDEX IF NOT EXISTS idx_scrape_runs_source   ON scrape_runs(source);
