-- ============================================================
-- Evatt AI — Legal Casebase Schema v2
-- Relational + Graph model replacing flat ChromaDB metadata
-- ============================================================
-- Migrations are additive. If you are upgrading from v1,
-- run the "ALTER TABLE" blocks at the bottom.
-- ============================================================

-- ── Extensions ───────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- trigram index for fuzzy citation search

-- ── Enums ────────────────────────────────────────────────────

CREATE TYPE court_jurisdiction AS ENUM (
    'HCA',      -- High Court of Australia
    'FCAFC',    -- Federal Court (Full Court)
    'FCA',      -- Federal Court
    'NSWCA',    -- NSW Court of Appeal
    'VSCA',     -- Victorian Court of Appeal
    'QCA',      -- Queensland Court of Appeal
    'WASCA',    -- WA Court of Appeal
    'SASCFC',   -- SA Supreme Court (Full Court)
    'TASFC',    -- Tasmanian Full Court
    'ACTCA',    -- ACT Court of Appeal
    'UNKNOWN'
);

CREATE TYPE treatment_type AS ENUM (
    'followed',       -- Explicitly followed and applied
    'applied',        -- Applied without explicit endorsement
    'approved',       -- Principle approved (often obiter)
    'distinguished',  -- Distinguished on facts/law
    'considered',     -- Discussed but not decisive
    'doubted',        -- Correctness cast into doubt
    'not_followed',   -- Declined to follow (short of overruling)
    'overruled',      -- Expressly overruled
    'reversed',       -- Reversed on appeal
    'affirmed'        -- Affirmed on appeal
);

CREATE TYPE good_law_status AS ENUM (
    'green',   -- No negative treatment; good law
    'amber',   -- Distinguished, doubted, or not followed
    'red',     -- Overruled or reversed — no longer good law
    'unknown'  -- Insufficient data to determine
);

-- ── Core: cases ──────────────────────────────────────────────

CREATE TABLE cases (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Identification
    neutral_citation    TEXT NOT NULL UNIQUE,  -- e.g. [2024] HCA 1
    medium_citation     TEXT,                  -- e.g. (2024) 306 CLR 1
    case_name           TEXT NOT NULL,
    short_name          TEXT,                  -- e.g. "Mabo" for Mabo v Queensland

    -- Classification
    court               court_jurisdiction NOT NULL DEFAULT 'UNKNOWN',
    year                SMALLINT NOT NULL CHECK (year BETWEEN 1900 AND 2100),
    judgment_date       DATE,
    catchwords          TEXT[],                -- Array of subject-matter tags

    -- Parties
    plaintiff           TEXT,
    defendant           TEXT,

    -- Document
    source_file         TEXT,
    source_url          TEXT,                  -- AustLII or JADE URL
    full_text_stored    BOOLEAN DEFAULT FALSE,

    -- Citator status (denormalised for fast reads)
    good_law            good_law_status NOT NULL DEFAULT 'unknown',
    citing_count        INTEGER NOT NULL DEFAULT 0,   -- how many later cases cite this
    negative_count      INTEGER NOT NULL DEFAULT 0,   -- how many cite negatively
    status_updated_at   TIMESTAMPTZ,

    -- Vector store reference
    chroma_ids          TEXT[],   -- chunk IDs in ChromaDB for this case

    -- Audit
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cases_citation   ON cases USING gin (neutral_citation gin_trgm_ops);
CREATE INDEX idx_cases_year       ON cases (year);
CREATE INDEX idx_cases_court      ON cases (court);
CREATE INDEX idx_cases_good_law   ON cases (good_law);
CREATE INDEX idx_cases_name_trgm  ON cases USING gin (case_name gin_trgm_ops);

-- ── Core: citation_edges (the citation graph) ─────────────────
-- One row per directional citation relationship.
-- citing_case → cited_case with a treatment label.

CREATE TABLE citation_edges (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Graph edge
    citing_case_id      UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    cited_case_id       UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,

    -- Treatment
    treatment           treatment_type NOT NULL DEFAULT 'considered',
    is_negative         BOOLEAN GENERATED ALWAYS AS (
                            treatment IN ('doubted','not_followed','overruled','reversed')
                        ) STORED,

    -- Provenance
    context_text        TEXT,          -- Verbatim excerpt from the citing judgment
    page_number         INTEGER,       -- Page where the citation appears
    paragraph_ref       TEXT,          -- e.g. "[42]" if paragraph refs available

    -- Confidence from the citator pipeline
    confidence          NUMERIC(4,3) CHECK (confidence BETWEEN 0 AND 1),
    classifier_version  TEXT DEFAULT '1.0',
    human_verified      BOOLEAN DEFAULT FALSE,

    -- Audit
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT no_self_citation CHECK (citing_case_id != cited_case_id),
    CONSTRAINT unique_edge UNIQUE (citing_case_id, cited_case_id, treatment)
);

CREATE INDEX idx_edges_citing   ON citation_edges (citing_case_id);
CREATE INDEX idx_edges_cited    ON citation_edges (cited_case_id);
CREATE INDEX idx_edges_treatment ON citation_edges (treatment);
CREATE INDEX idx_edges_negative  ON citation_edges (is_negative) WHERE is_negative = TRUE;

-- ── Treatment log (audit trail) ──────────────────────────────

CREATE TABLE treatment_log (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    edge_id         UUID NOT NULL REFERENCES citation_edges(id) ON DELETE CASCADE,
    old_treatment   treatment_type,
    new_treatment   treatment_type NOT NULL,
    old_status      good_law_status,
    new_status      good_law_status,
    changed_by      TEXT DEFAULT 'citator_pipeline',
    note            TEXT,
    changed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── BM25 full-text search index ──────────────────────────────

CREATE TABLE case_text_index (
    case_id     UUID PRIMARY KEY REFERENCES cases(id) ON DELETE CASCADE,
    -- Full judgment text stored for BM25; embeddings live in ChromaDB
    body_text   TEXT NOT NULL,
    tsv         TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(body_text, ''))
                ) STORED
);

CREATE INDEX idx_text_tsv ON case_text_index USING gin (tsv);

-- ── Noteup index (materialised for fast lookup) ───────────────
-- Pre-aggregated for the GET /noteup/{citation} endpoint.
-- Refreshed by the citator pipeline after each new ingestion.

CREATE MATERIALIZED VIEW noteup_index AS
SELECT
    ci.cited_case_id                              AS case_id,
    c_cited.neutral_citation                      AS cited_citation,
    c_cited.case_name                             AS cited_name,
    COUNT(*)                                      AS total_citing,
    COUNT(*) FILTER (WHERE ci.is_negative)        AS negative_citing,
    ARRAY_AGG(DISTINCT ci.treatment::TEXT)        AS treatments_received,
    ARRAY_AGG(
        JSONB_BUILD_OBJECT(
            'citing_citation', c_citing.neutral_citation,
            'citing_name',     c_citing.case_name,
            'citing_year',     c_citing.year,
            'treatment',       ci.treatment,
            'context',         ci.context_text,
            'page',            ci.page_number,
            'confidence',      ci.confidence
        ) ORDER BY c_citing.year DESC
    )                                             AS citing_cases
FROM citation_edges ci
JOIN cases c_cited  ON c_cited.id  = ci.cited_case_id
JOIN cases c_citing ON c_citing.id = ci.citing_case_id
GROUP BY ci.cited_case_id, c_cited.neutral_citation, c_cited.case_name
WITH DATA;

CREATE UNIQUE INDEX idx_noteup_case_id ON noteup_index (case_id);
CREATE INDEX idx_noteup_citation       ON noteup_index (cited_citation);

-- ── Helper: refresh noteup index ─────────────────────────────
CREATE OR REPLACE FUNCTION refresh_noteup_index()
RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY noteup_index;
END;
$$;

-- ── Helper: recompute good_law status for one case ────────────
CREATE OR REPLACE FUNCTION recompute_good_law(p_case_id UUID)
RETURNS good_law_status LANGUAGE plpgsql AS $$
DECLARE
    v_overruled   INTEGER;
    v_negative    INTEGER;
    v_status      good_law_status;
BEGIN
    SELECT
        COUNT(*) FILTER (WHERE treatment IN ('overruled','reversed')),
        COUNT(*) FILTER (WHERE is_negative)
    INTO v_overruled, v_negative
    FROM citation_edges
    WHERE cited_case_id = p_case_id;

    IF v_overruled > 0 THEN
        v_status := 'red';
    ELSIF v_negative > 0 THEN
        v_status := 'amber';
    ELSE
        v_status := 'green';
    END IF;

    UPDATE cases
    SET good_law           = v_status,
        negative_count     = v_negative,
        status_updated_at  = NOW(),
        updated_at         = NOW()
    WHERE id = p_case_id;

    RETURN v_status;
END;
$$;

-- ── Trigger: auto-update citing_count on edge insert/delete ──

CREATE OR REPLACE FUNCTION update_citing_counts()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE cases SET citing_count = citing_count + 1,
                         updated_at   = NOW()
        WHERE id = NEW.cited_case_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE cases SET citing_count = GREATEST(citing_count - 1, 0),
                         updated_at   = NOW()
        WHERE id = OLD.cited_case_id;
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER trg_citing_count
AFTER INSERT OR DELETE ON citation_edges
FOR EACH ROW EXECUTE FUNCTION update_citing_counts();

-- ── Upgrade path from v1 ChromaDB-only setup ─────────────────
-- Run these if you already have a cases table without the new columns:
--
-- ALTER TABLE cases ADD COLUMN IF NOT EXISTS good_law       good_law_status NOT NULL DEFAULT 'unknown';
-- ALTER TABLE cases ADD COLUMN IF NOT EXISTS citing_count   INTEGER NOT NULL DEFAULT 0;
-- ALTER TABLE cases ADD COLUMN IF NOT EXISTS negative_count INTEGER NOT NULL DEFAULT 0;
-- ALTER TABLE cases ADD COLUMN IF NOT EXISTS chroma_ids     TEXT[];
