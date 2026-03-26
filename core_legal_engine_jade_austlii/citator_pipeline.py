"""
citator_pipeline.py — Evatt AI | Citation Analysis & Treatment Classifier
--------------------------------------------------------------------------
Analyses newly ingested judgments to:
  1. Extract every cited case from the judgment text.
  2. Classify the judicial treatment (followed / distinguished / overruled …).
  3. Upsert citation_edges into PostgreSQL.
  4. Recompute the good_law status for every cited case.
  5. Refresh the materialised noteup_index.

Usage:
    python citator_pipeline.py --case-id <uuid>
    python citator_pipeline.py --reindex-all          # full rebuild

Environment variables required:
    DATABASE_URL       postgresql://user:pass@host:5432/evatt
    ANTHROPIC_API_KEY  sk-ant-...
    OPENAI_API_KEY     sk-...     (for embeddings; used by ingest layer)
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("evatt.citator")

# ──────────────────────────────────────────────────────────────────────────────
# Treatment Lexicon
# ──────────────────────────────────────────────────────────────────────────────
# Each treatment is associated with keyword patterns.  We search a ±300-char
# window around each citation occurrence.  Claude is used as a fallback for
# ambiguous windows.

TREATMENT_PATTERNS: dict[str, re.Pattern] = {
    "overruled": re.compile(
        r"\b(overrul(?:ed|ing|e)|no longer (?:good )?law|expressly overrul(?:ed|ing)|"
        r"depart(?:ing|ed) from|depart from)\b",
        re.IGNORECASE,
    ),
    "reversed": re.compile(
        r"\b(revers(?:ed|ing|al)|set aside|appeal (?:allowed|granted))\b",
        re.IGNORECASE,
    ),
    "not_followed": re.compile(
        r"\b(decline[ds]? to follow|not follow(?:ed|ing)?|"
        r"prefer(?:red|ring)? not to follow|unable to follow)\b",
        re.IGNORECASE,
    ),
    "doubted": re.compile(
        r"\b(doubt(?:ed|ing|s|ful)|question(?:ed|ing|s)|cast(?:ing)? doubt|"
        r"with respect.{0,30}disagree|respectfully doubt)\b",
        re.IGNORECASE,
    ),
    "distinguished": re.compile(
        r"\b(distinguish(?:ed|able|ing)|distinguishable|"
        r"differs? (?:in|on)|factual(?:ly)? distinct|"
        r"present(?:ing)? case is different)\b",
        re.IGNORECASE,
    ),
    "followed": re.compile(
        r"\b(follow(?:ed|ing)|applied|reaffirm(?:ed|ing)|"
        r"endorse[ds]?|adopt(?:ed|ing)|consistent with)\b",
        re.IGNORECASE,
    ),
    "approved": re.compile(
        r"\b(approv(?:ed|ing)|affirm(?:ed|ing)|support(?:ed|ing)|"
        r"correct(?:ly)? decided|I agree)\b",
        re.IGNORECASE,
    ),
    "affirmed": re.compile(
        r"\b(affirm(?:ed|ing)|uphold|upheld)\b",
        re.IGNORECASE,
    ),
    "applied": re.compile(
        r"\b(applied|applying|gives? effect to|in accordance with)\b",
        re.IGNORECASE,
    ),
    "considered": re.compile(
        r"\b(consider(?:ed|ing)|refer(?:red|ring)|not(?:ed|ing)|"
        r"discuss(?:ed|ing)|examined|reviewed)\b",
        re.IGNORECASE,
    ),
}

# Negative treatments that trigger amber/red status
NEGATIVE_TREATMENTS = frozenset(
    {"overruled", "reversed", "not_followed", "doubted"}
)

# Priority order: higher = more legally significant
TREATMENT_PRIORITY: dict[str, int] = {
    "overruled":    10,
    "reversed":      9,
    "not_followed":  8,
    "doubted":       7,
    "distinguished": 6,
    "approved":      5,
    "affirmed":      4,
    "followed":      3,
    "applied":       2,
    "considered":    1,
}

# ──────────────────────────────────────────────────────────────────────────────
# Citation extraction patterns
# ──────────────────────────────────────────────────────────────────────────────

# Matches:  [2024] HCA 1   (2024) 306 CLR 1   [2018] FCAFC 45   etc.
_CITATION_RE = re.compile(
    r"""
    (?:
        \[(?P<nc_year>\d{4})\]\s*
        (?P<nc_court>HCA|FCAFC|FCA|NSWCA|VSCA|QCA|WASCA|SASCFC|TASFC|ACTCA)
        \s*(?P<nc_num>\d+)
    )
    |
    (?:
        \((?P<clr_year>\d{4})\)\s*(?P<clr_vol>\d+)\s*CLR\s*(?P<clr_page>\d+)
    )
    """,
    re.VERBOSE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CitationOccurrence:
    """A single occurrence of a citation in a judgment, with context window."""
    raw_citation:   str
    normalised:     str          # canonical form, e.g. "[2018] HCA 63"
    char_offset:    int
    context_window: str          # ±300 chars around the citation
    page_number:    int


@dataclass
class ClassifiedEdge:
    """One edge of the citation graph, ready to upsert."""
    citing_citation: str
    cited_citation:  str
    treatment:       str
    confidence:      float
    context_text:    str
    page_number:     int
    classifier_used: str         # "regex" | "llm" | "heuristic"


# ──────────────────────────────────────────────────────────────────────────────
# Database helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_connection() -> PgConnection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def lookup_case_id(conn: PgConnection, citation: str) -> Optional[str]:
    """Return the UUID for a citation, or None if not in the database."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM cases WHERE neutral_citation = %s", (citation,)
        )
        row = cur.fetchone()
        return str(row["id"]) if row else None


def upsert_citation_edge(conn: PgConnection, edge: ClassifiedEdge) -> None:
    """Insert or update a citation_edges row."""
    citing_id = lookup_case_id(conn, edge.citing_citation)
    cited_id  = lookup_case_id(conn, edge.cited_citation)

    if not citing_id or not cited_id:
        log.debug(
            f"Skipping edge {edge.citing_citation} → {edge.cited_citation}: "
            f"one or both cases not found in DB."
        )
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO citation_edges
                (id, citing_case_id, cited_case_id, treatment,
                 context_text, page_number, confidence, classifier_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (citing_case_id, cited_case_id, treatment)
            DO UPDATE SET
                context_text       = EXCLUDED.context_text,
                page_number        = EXCLUDED.page_number,
                confidence         = GREATEST(citation_edges.confidence, EXCLUDED.confidence),
                classifier_version = EXCLUDED.classifier_version
            """,
            (
                str(uuid.uuid4()),
                citing_id,
                cited_id,
                edge.treatment,
                edge.context_text[:2000],    # guard column width
                edge.page_number,
                edge.confidence,
                edge.classifier_used,
            ),
        )
    conn.commit()


def recompute_status(conn: PgConnection, citation: str) -> Optional[str]:
    """Call the DB function to recompute a case's good_law status."""
    case_id = lookup_case_id(conn, citation)
    if not case_id:
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT recompute_good_law(%s)", (case_id,))
        row = cur.fetchone()
        conn.commit()
        return row[0] if row else None


def refresh_noteup_index(conn: PgConnection) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT refresh_noteup_index()")
        conn.commit()
    log.info("Noteup index refreshed.")


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Citation extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_citation_occurrences(
    text: str, page_texts: list[str]
) -> list[CitationOccurrence]:
    """
    Scan *text* for all legal citations and return a structured occurrence
    for each, including a ±300-character context window and the source page.
    """

    def page_for_offset(offset: int) -> int:
        pos = 0
        for i, pt in enumerate(page_texts, start=1):
            pos += len(pt) + 2
            if offset < pos:
                return i
        return len(page_texts)

    occurrences: list[CitationOccurrence] = []
    seen_offsets: set[int] = set()

    for m in _CITATION_RE.finditer(text):
        start = m.start()
        if start in seen_offsets:
            continue
        seen_offsets.add(start)

        raw = m.group().strip()

        # Normalise to neutral citation format "[YEAR] COURT NUM"
        if m.group("nc_year"):
            normalised = f"[{m.group('nc_year')}] {m.group('nc_court')} {m.group('nc_num')}"
        elif m.group("clr_year"):
            normalised = f"({m.group('clr_year')}) {m.group('clr_vol')} CLR {m.group('clr_page')}"
        else:
            normalised = raw

        ctx_start = max(0, start - 300)
        ctx_end   = min(len(text), m.end() + 300)
        context   = text[ctx_start:ctx_end].replace("\n", " ").strip()
        page      = page_for_offset(start)

        occurrences.append(
            CitationOccurrence(
                raw_citation=raw,
                normalised=normalised,
                char_offset=start,
                context_window=context,
                page_number=page,
            )
        )

    log.info(f"Extracted {len(occurrences)} citation occurrence(s).")
    return occurrences


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Regex-based treatment classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_by_regex(context: str) -> tuple[str, float]:
    """
    Search the context window for treatment keywords.
    Returns (treatment_label, confidence).
    Picks the highest-priority match.
    """
    best_treatment  = "considered"
    best_priority   = TREATMENT_PRIORITY["considered"]
    best_confidence = 0.55   # baseline for default

    for label, pattern in TREATMENT_PATTERNS.items():
        if pattern.search(context):
            priority = TREATMENT_PRIORITY[label]
            if priority > best_priority:
                best_priority   = priority
                best_treatment  = label
                best_confidence = 0.82   # regex hit gives moderate confidence

    return best_treatment, best_confidence


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — LLM fallback classifier
# ──────────────────────────────────────────────────────────────────────────────

_CITATOR_SYSTEM = """\
You are a specialist Australian legal analyst. Your task is to classify the \
judicial treatment applied to a cited case based on the surrounding text excerpt.

Reply with ONLY a JSON object — no prose, no markdown, no backticks:
{
  "treatment": "<one of: followed|applied|approved|distinguished|considered|doubted|not_followed|overruled|reversed|affirmed>",
  "confidence": <float 0.0–1.0>,
  "reason": "<one sentence>"
}

Rules:
- "overruled" requires an explicit statement that the earlier case was wrongly decided.
- "distinguished" means the court accepts the cited case but finds it inapplicable here.
- "doubted" means uncertainty is expressed about correctness without a definitive finding.
- "followed" means the court adopts and applies the cited reasoning.
- Default to "considered" when the citation is merely referential.
"""

def classify_by_llm(
    citation: str, context: str, client: anthropic.Anthropic
) -> tuple[str, float]:
    """
    Call Claude to classify treatment for ambiguous context windows.
    Falls back to 'considered' on any error.
    """
    import json

    prompt = (
        f"Cited case: {citation}\n\n"
        f"Excerpt from the citing judgment:\n\"\"\"\n{context[:800]}\n\"\"\"\n\n"
        "Classify the judicial treatment."
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=_CITATOR_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        data = json.loads(raw)
        treatment  = data.get("treatment", "considered")
        confidence = float(data.get("confidence", 0.70))
        log.debug(f"LLM classified '{citation}' as '{treatment}' ({confidence:.2f})")
        return treatment, confidence

    except (json.JSONDecodeError, KeyError, Exception) as exc:
        log.warning(f"LLM classifier failed for '{citation}': {exc}. Defaulting.")
        return "considered", 0.50


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Master classification router
# ──────────────────────────────────────────────────────────────────────────────

LLM_CONFIDENCE_THRESHOLD = 0.75   # if regex confidence < this, escalate to LLM

def classify_treatment(
    occurrence: CitationOccurrence,
    llm_client: Optional[anthropic.Anthropic] = None,
) -> tuple[str, float, str]:
    """
    Route to regex first; if confidence is below threshold AND we have a
    Claude client, escalate to LLM classification.

    Returns (treatment, confidence, classifier_used).
    """
    treatment, confidence = classify_by_regex(occurrence.context_window)

    if confidence < LLM_CONFIDENCE_THRESHOLD and llm_client is not None:
        treatment, confidence = classify_by_llm(
            occurrence.normalised, occurrence.context_window, llm_client
        )
        return treatment, confidence, "llm"

    return treatment, confidence, "regex"


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Status propagation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_status_from_treatments(
    treatments: list[str],
) -> str:
    """
    Given all treatment labels received by a case, derive its good_law status.

    Priority: overruled/reversed → red > doubted/not_followed → amber > green
    """
    treatment_set = set(treatments)

    if treatment_set & {"overruled", "reversed"}:
        return "red"
    if treatment_set & {"doubted", "not_followed"}:
        return "amber"
    return "green"


def propagate_status(
    conn: PgConnection, cited_citations: set[str]
) -> dict[str, str]:
    """
    After new edges are written, recompute the good_law status for every
    case that was cited. Returns a dict of {citation: new_status}.
    """
    results: dict[str, str] = {}
    for citation in cited_citations:
        new_status = recompute_status(conn, citation)
        if new_status:
            results[citation] = new_status
            log.info(f"  Status updated: {citation} → {new_status}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_citator_pipeline(
    citing_citation: str,
    judgment_text:   str,
    page_texts:      list[str],
    use_llm:         bool = True,
) -> dict:
    """
    Full pipeline for a single newly-ingested judgment.

    Parameters
    ----------
    citing_citation
        Neutral citation of the new judgment being analysed.
    judgment_text
        Full concatenated text of the judgment.
    page_texts
        Per-page text list (used to recover page numbers).
    use_llm
        Set False to disable Claude fallback (faster, less accurate).

    Returns
    -------
    dict with keys: edges_written, statuses_updated, refresh_ok
    """
    log.info(f"Starting citator pipeline for: {citing_citation}")
    t0 = time.perf_counter()

    conn       = get_connection()
    llm_client = anthropic.Anthropic() if use_llm else None

    # ── 1. Extract ────────────────────────────────────────────
    occurrences = extract_citation_occurrences(judgment_text, page_texts)

    # Filter out self-references
    occurrences = [
        o for o in occurrences if o.normalised != citing_citation
    ]

    # ── 2 & 3. Classify ──────────────────────────────────────
    edges:           list[ClassifiedEdge] = []
    cited_citations: set[str]             = set()

    for occ in occurrences:
        treatment, confidence, classifier = classify_treatment(occ, llm_client)

        edges.append(
            ClassifiedEdge(
                citing_citation=citing_citation,
                cited_citation=occ.normalised,
                treatment=treatment,
                confidence=confidence,
                context_text=occ.context_window,
                page_number=occ.page_number,
                classifier_used=classifier,
            )
        )
        cited_citations.add(occ.normalised)

    log.info(f"Classified {len(edges)} edges.")

    # ── 4. Upsert edges ───────────────────────────────────────
    edges_written = 0
    for edge in edges:
        try:
            upsert_citation_edge(conn, edge)
            edges_written += 1
        except Exception as exc:
            log.warning(f"Edge upsert failed ({edge.cited_citation}): {exc}")

    log.info(f"Wrote {edges_written}/{len(edges)} edges.")

    # ── 5. Propagate status ───────────────────────────────────
    statuses = propagate_status(conn, cited_citations)

    # ── 6. Refresh noteup index ───────────────────────────────
    refresh_ok = False
    try:
        refresh_noteup_index(conn)
        refresh_ok = True
    except Exception as exc:
        log.warning(f"Noteup index refresh failed: {exc}")

    conn.close()

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    log.info(f"Pipeline complete in {elapsed} ms.")

    return {
        "citing_citation":  citing_citation,
        "edges_written":    edges_written,
        "citations_found":  len(occurrences),
        "statuses_updated": statuses,
        "refresh_ok":       refresh_ok,
        "elapsed_ms":       elapsed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full re-index (rebuild all edges from stored text)
# ──────────────────────────────────────────────────────────────────────────────

def reindex_all(use_llm: bool = False) -> None:
    """
    Re-run the citator pipeline over every case in the database.
    Designed for initial population or schema migrations.
    use_llm=False recommended for bulk runs (cost/speed).
    """
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.neutral_citation, t.body_text
            FROM cases c
            JOIN case_text_index t ON t.case_id = c.id
            ORDER BY c.year ASC
            """
        )
        rows = cur.fetchall()
    conn.close()

    log.info(f"Re-indexing {len(rows)} cases…")
    for row in rows:
        try:
            run_citator_pipeline(
                citing_citation=row["neutral_citation"],
                judgment_text=row["body_text"],
                page_texts=[row["body_text"]],   # simplified: single-page for bulk
                use_llm=use_llm,
            )
        except Exception as exc:
            log.error(f"Pipeline error for {row['neutral_citation']}: {exc}")

    log.info("Re-index complete.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evatt AI Citator Pipeline")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case-citation", type=str,
                       help="Neutral citation of the case to analyse, e.g. '[2024] HCA 1'")
    group.add_argument("--reindex-all", action="store_true",
                       help="Re-run pipeline over all cases (no LLM)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM fallback classifier")
    args = parser.parse_args()

    if args.reindex_all:
        reindex_all(use_llm=False)
    else:
        # For CLI testing: load text from DB
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.neutral_citation, t.body_text
                FROM cases c
                JOIN case_text_index t ON t.case_id = c.id
                WHERE c.neutral_citation = %s
                """,
                (args.case_citation,),
            )
            row = cur.fetchone()
        conn.close()

        if not row:
            print(f"Case '{args.case_citation}' not found in database.")
            raise SystemExit(1)

        result = run_citator_pipeline(
            citing_citation=row["neutral_citation"],
            judgment_text=row["body_text"],
            page_texts=[row["body_text"]],
            use_llm=not args.no_llm,
        )
        import json
        print(json.dumps(result, indent=2))
