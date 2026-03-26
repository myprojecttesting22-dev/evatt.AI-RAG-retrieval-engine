"""
main.py — Evatt AI | Legal RAG Retrieval API
---------------------------------------------
FastAPI application exposing semantic search and RAG-powered Q&A
over a local ChromaDB vector store of Australian High Court cases.

Endpoints:
    GET  /              → Health check
    GET  /search        → Hybrid semantic / citation-filtered search
    POST /ask           → RAG Q&A via Claude claude-sonnet-4-6
    GET  /collections   → List available ChromaDB collections + stats

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import anthropic
import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("evatt.api")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

VECTOR_DB_DIR = Path(
    os.getenv("VECTOR_DB_DIR", "../ingestion engine/vector_db")
).resolve()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_documents")
EMBEDDING_MODEL = "text-embedding-3-small"
CLAUDE_MODEL    = "claude-sonnet-4-6"

DEFAULT_TOP_K   = 5
MAX_TOP_K       = 20
MAX_CONTEXT_CHARS = 12_000   # hard cap on context fed to Claude

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    chunk_id:           str
    text:               str
    score:              float                        # cosine similarity (0–1)
    case_name:          str
    citation:           str
    court_jurisdiction: str
    year:               str
    page_number:        int
    source_file:        str


class SearchResponse(BaseModel):
    query:         str
    search_mode:   str                               # "semantic" | "citation_filter"
    filter_applied: Optional[dict[str, Any]]
    results:       list[SearchResult]
    latency_ms:    float


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000,
                          description="Legal question to answer using the case database.")
    top_k:    int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class Source(BaseModel):
    citation:    str
    case_name:   str
    page_number: int
    jurisdiction: str


class AskResponse(BaseModel):
    question:    str
    answer:      str
    sources:     list[Source]
    search_mode: str
    latency_ms:  float


class HealthResponse(BaseModel):
    status:      str
    collection:  str
    chunk_count: int
    vector_db:   str
    model:       str


# ──────────────────────────────────────────────────────────────────────────────
# Citation Detection (Hybrid Search Logic)
# ──────────────────────────────────────────────────────────────────────────────

# Full neutral citation  → [2018] HCA 63
_FULL_CITATION_RE = re.compile(
    r"""
    (?:
        \[(?P<year>\d{4})\]\s*
        (?P<court>HCA|FCAFC|FCA|NSWCA|VSCA|QCA|WASCA)
        \s*(?P<num>\d+)
    )
    |
    (?:(?P<insc_year>\d{4})\s+INSC\s+\d+)
    |
    (?:\((?P<scc_year>\d{4})\)\s*\d+\s*SCC\s+\d+)
    |
    (?:AIR\s+(?P<air_year>\d{4})\s+SC\s+\d+)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Partial court abbreviation → "HCA" / "FCAFC" in the query
_COURT_ABBREV_RE = re.compile(
    r"\b(HCA|FCAFC|FCA|NSWCA|VSCA|QCA|WASCA|INSC|SCC)\b",
    re.IGNORECASE,
)

# Court abbreviation → ChromaDB jurisdiction label
_COURT_TO_JURISDICTION: dict[str, str] = {
    "HCA":    "High Court of Australia",
    "FCAFC":  "Federal Court of Australia",
    "FCA":    "Federal Court of Australia",
    "NSWCA":  "New South Wales Court of Appeal",
    "VSCA":   "Court of Appeal of Victoria",
    "QCA":    "Queensland Court of Appeal",
    "WASCA":  "Western Australia Court of Appeal",
    "INSC":   "Supreme Court of India",
    "SCC":    "Supreme Court of India",
}


def detect_citation_filter(query: str) -> tuple[str, Optional[dict]]:
    """
    Analyse *query* and return (mode, chroma_where_filter).

    Modes:
        "citation_filter" — full citation matched  → filter on `citation`
        "jurisdiction_filter" — court abbreviation found → filter on `court_jurisdiction`
        "semantic"        — no citation signal     → plain vector search
    """
    # 1. Full citation match
    full_match = _FULL_CITATION_RE.search(query)
    if full_match:
        citation_str = full_match.group().strip()
        log.info(f"Citation detected: '{citation_str}' → citation filter")
        return "citation_filter", {"citation": {"$eq": citation_str}}

    # 2. Partial court abbreviation
    abbrev_match = _COURT_ABBREV_RE.search(query)
    if abbrev_match:
        abbrev = abbrev_match.group().upper()
        jurisdiction = _COURT_TO_JURISDICTION.get(abbrev)
        if jurisdiction:
            log.info(f"Court abbreviation '{abbrev}' → jurisdiction filter: '{jurisdiction}'")
            return "jurisdiction_filter", {"court_jurisdiction": {"$eq": jurisdiction}}

    # 3. Plain semantic search
    return "semantic", None


# ──────────────────────────────────────────────────────────────────────────────
# Client Singletons (initialised once at startup)
# ──────────────────────────────────────────────────────────────────────────────

_chroma_collection: Optional[chromadb.Collection] = None
_openai_client:     Optional[OpenAI]              = None
_anthropic_client:  Optional[anthropic.Anthropic] = None


def _get_collection() -> chromadb.Collection:
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    if not VECTOR_DB_DIR.exists():
        raise RuntimeError(
            f"Vector DB directory not found: {VECTOR_DB_DIR}\n"
            "Run the ingestion engine first, or set VECTOR_DB_DIR."
        )

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        _chroma_collection = client.get_collection(COLLECTION_NAME)
        log.info(
            f"ChromaDB collection '{COLLECTION_NAME}' loaded "
            f"({_chroma_collection.count()} chunks) from {VECTOR_DB_DIR}"
        )
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found in {VECTOR_DB_DIR}. "
            f"Original error: {exc}"
        ) from exc

    return _chroma_collection


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


# ──────────────────────────────────────────────────────────────────────────────
# Core Retrieval
# ──────────────────────────────────────────────────────────────────────────────

def _embed_query(query: str) -> list[float]:
    """Return the embedding vector for a single query string."""
    response = _get_openai().embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return response.data[0].embedding


def _run_chroma_query(
    query: str,
    top_k: int,
    where_filter: Optional[dict],
) -> tuple[list[SearchResult], Optional[dict]]:
    """
    Execute a ChromaDB query (with or without a metadata filter).

    Falls back to semantic-only if the filter returns 0 results.
    """
    collection = _get_collection()
    embedding  = _embed_query(query)

    query_kwargs: dict[str, Any] = dict(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    applied_filter: Optional[dict] = None

    if where_filter:
        query_kwargs["where"] = where_filter
        applied_filter = where_filter

    try:
        raw = collection.query(**query_kwargs)
    except Exception as exc:
        # Filter produced an error (e.g. no matching docs) — fall back
        log.warning(f"Filtered query failed ({exc}), falling back to semantic search.")
        query_kwargs.pop("where", None)
        applied_filter = None
        raw = collection.query(**query_kwargs)

    # Unpack first (and only) query result list
    docs      = raw["documents"][0]
    metas     = raw["metadatas"][0]
    distances = raw["distances"][0]

    results: list[SearchResult] = []
    for chunk_id_raw, doc, meta, dist in zip(
        raw.get("ids", [[]])[0], docs, metas, distances
    ):
        results.append(
            SearchResult(
                chunk_id=chunk_id_raw,
                text=doc,
                score=round(1.0 - float(dist), 4),
                case_name=meta.get("case_name", ""),
                citation=meta.get("citation", ""),
                court_jurisdiction=meta.get("court_jurisdiction", ""),
                year=str(meta.get("year", "")),
                page_number=int(meta.get("page_number", 0)),
                source_file=meta.get("source_file", ""),
            )
        )

    return results, applied_filter


# ──────────────────────────────────────────────────────────────────────────────
# Claude Prompt
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert Australian legal assistant for Evatt AI, a legal research platform.

Your role is to answer legal questions accurately and professionally using ONLY the \
case law excerpts provided to you in the user message. You must not draw on any \
external knowledge or make legal claims not supported by the provided text.

Guidelines:
- Cite specific cases and page numbers when referencing holdings or reasoning.
- If the provided excerpts do not contain sufficient information to answer the \
question, state clearly: "The provided case excerpts do not contain sufficient \
information to answer this question."
- Use formal legal writing style.
- Do not speculate or extrapolate beyond what is written in the excerpts.
- Structure your answer with a clear analysis section followed by a 'Sources' section.

Sources Section Format (REQUIRED at the end of every response):
---
**Sources**
- [Citation] — [Case Name], p. [page_number]
  (repeat for each unique source used)
"""


def _build_user_message(question: str, chunks: list[SearchResult]) -> str:
    """Assemble the user turn: question + numbered context excerpts."""
    context_parts: list[str] = []
    total_chars = 0

    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[Excerpt {i}]\n"
            f"Case: {chunk.case_name}\n"
            f"Citation: {chunk.citation}\n"
            f"Jurisdiction: {chunk.court_jurisdiction}\n"
            f"Page: {chunk.page_number}\n"
            f"---\n"
            f"{chunk.text.strip()}"
        )
        if total_chars + len(header) > MAX_CONTEXT_CHARS:
            log.warning("Context truncated at %d excerpts to stay within token budget.", i - 1)
            break
        context_parts.append(header)
        total_chars += len(header)

    context_block = "\n\n".join(context_parts)

    return (
        f"Question: {question}\n\n"
        f"Case Law Excerpts:\n\n"
        f"{context_block}\n\n"
        "Please answer the question using only the excerpts above. "
        "Include a Sources section at the end."
    )


def _extract_sources(chunks: list[SearchResult]) -> list[Source]:
    """Deduplicate and surface the source metadata for the API response."""
    seen: set[str] = set()
    sources: list[Source] = []
    for c in chunks:
        key = f"{c.citation}:{c.page_number}"
        if key not in seen:
            seen.add(key)
            sources.append(
                Source(
                    citation=c.citation,
                    case_name=c.case_name,
                    page_number=c.page_number,
                    jurisdiction=c.court_jurisdiction,
                )
            )
    return sources


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Evatt AI — Legal Retrieval API",
    description=(
        "Semantic and citation-aware search over Australian High Court case law, "
        "with RAG-powered Q&A via Claude."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Pre-warm all clients on startup so the first request is fast."""
    log.info("Starting Evatt AI Retrieval API…")
    try:
        col = _get_collection()
        log.info(f"ChromaDB ready  — {col.count()} chunks in '{COLLECTION_NAME}'")
    except RuntimeError as exc:
        log.error(f"ChromaDB init failed: {exc}")

    try:
        _get_openai()
        log.info("OpenAI client   — ready")
    except RuntimeError as exc:
        log.warning(f"OpenAI init: {exc}")

    try:
        _get_anthropic()
        log.info("Anthropic client — ready")
    except RuntimeError as exc:
        log.warning(f"Anthropic init: {exc}")


# ── GET /  ────────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Return service health and collection statistics."""
    try:
        col = _get_collection()
        chunk_count = col.count()
        db_status = str(VECTOR_DB_DIR)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return HealthResponse(
        status="ok",
        collection=COLLECTION_NAME,
        chunk_count=chunk_count,
        vector_db=db_status,
        model=CLAUDE_MODEL,
    )


# ── GET /search ───────────────────────────────────────────────────────────────

@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search(
    query: str = Query(..., min_length=2, max_length=500,
                       description="Natural language or citation query"),
    top_k: int = Query(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K,
                       description="Number of results to return"),
) -> SearchResponse:
    """
    **Hybrid search endpoint.**

    - If the query contains a full legal citation (e.g. `[2018] HCA 63`),
      ChromaDB is filtered to that exact citation.
    - If the query contains a court abbreviation (e.g. `HCA`, `FCA`),
      results are restricted to that jurisdiction.
    - Otherwise, a standard cosine-similarity semantic search is performed.
    """
    t0 = time.perf_counter()

    try:
        search_mode, where_filter = detect_citation_filter(query)
        results, applied_filter   = _run_chroma_query(query, top_k, where_filter)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.exception("Unexpected error during /search")
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return SearchResponse(
        query=query,
        search_mode=search_mode,
        filter_applied=applied_filter,
        results=results,
        latency_ms=latency_ms,
    )


# ── POST /ask ─────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse, tags=["RAG Q&A"])
async def ask(body: AskRequest) -> AskResponse:
    """
    **RAG-powered legal Q&A.**

    Retrieves the most relevant case law excerpts using hybrid search,
    then asks Claude to answer using only those excerpts.
    """
    t0 = time.perf_counter()

    # ── Retrieve ─────────────────────────────────────────────────────────
    try:
        search_mode, where_filter = detect_citation_filter(body.question)
        chunks, _                 = _run_chroma_query(body.question, body.top_k, where_filter)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.exception("Retrieval failed in /ask")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant case law found for this question.",
        )

    # ── Generate ──────────────────────────────────────────────────────────
    user_message = _build_user_message(body.question, chunks)

    try:
        claude = _get_anthropic()
        response = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = response.content[0].text
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Anthropic rate limit reached. Retry shortly.")
    except anthropic.APIStatusError as exc:
        log.exception("Anthropic API error")
        raise HTTPException(status_code=502, detail=f"Claude API error: {exc.message}")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return AskResponse(
        question=body.question,
        answer=answer,
        sources=_extract_sources(chunks),
        search_mode=search_mode,
        latency_ms=latency_ms,
    )


# ── GET /collections ──────────────────────────────────────────────────────────

@app.get("/collections", tags=["Admin"])
async def list_collections() -> JSONResponse:
    """
    List all ChromaDB collections in the local vector store,
    with document counts and sample metadata keys.
    """
    if not VECTOR_DB_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Vector DB not found at {VECTOR_DB_DIR}",
        )

    try:
        client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        collections_info = []
        for col in client.list_collections():
            c = client.get_collection(col.name)
            sample = c.get(limit=1, include=["metadatas"])
            meta_keys = list(sample["metadatas"][0].keys()) if sample["metadatas"] else []
            collections_info.append(
                {
                    "name":       col.name,
                    "count":      c.count(),
                    "meta_keys":  meta_keys,
                }
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(
        content={
            "vector_db": str(VECTOR_DB_DIR),
            "collections": collections_info,
        }
    )
