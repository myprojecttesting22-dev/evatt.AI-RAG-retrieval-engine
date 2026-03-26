"""
ingest.py — Evatt AI | Legal RAG Ingestion Engine (LOCAL BOSS EDITION)
--------------------------------------------------
Processes High Court of Australia and Indian Judiciary PDFs into a
ChromaDB vector store using 100% local HuggingFace embeddings.
No OpenAI keys required.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("./data")
VECTOR_DB_DIR = Path("./vector_db")
COLLECTION_NAME = "legal_documents"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Lightning-fast local model

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("evatt.ingest")
console = Console()

# ──────────────────────────────────────────────────────────────────────────────
# Legal Citation Patterns
# ──────────────────────────────────────────────────────────────────────────────

HCA_PATTERN = re.compile(
    r"""
    (?:
        \[(?P<hca_year1>\d{4})\]\s*
        (?:HCA|FCAFC|FCA|NSWCA|VSCA|QCA|WASCA|SASCFC|TASFC|ACTCA|NTCA)
        \s*\d+
    )
    |
    (?:
        \((?P<hca_year2>\d{4})\)\s*\d+\s*CLR\s*\d+
    )
    """,
    re.VERBOSE,
)

INSC_PATTERN = re.compile(
    r"""
    (?:
        (?P<insc_year>\d{4})\s+INSC\s+\d+
    )
    |
    (?:
        \((?P<scc_year>\d{4})\)\s*\d+\s*SCC\s*\d+
    )
    |
    (?:
        AIR\s+(?P<air_year>\d{4})\s+SC\s+\d+
    )
    """,
    re.VERBOSE,
)

NEUTRAL_PATTERN = re.compile(
    r"\[(?P<neutral_year>\d{4})\]\s+[A-Z]{2,6}(?:\s+[A-Za-z]+)?\s+\d+",
)

ALL_CITATION_PATTERNS = [HCA_PATTERN, INSC_PATTERN, NEUTRAL_PATTERN]

# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LegalDocument:
    file_path: Path
    case_name: str
    citation: str
    court_jurisdiction: str
    year: str
    full_text: str
    page_texts: list[str] = field(default_factory=list)


@dataclass
class LegalChunk:
    chunk_id: str
    text: str
    case_name: str
    citation: str
    court_jurisdiction: str
    year: str
    page_number: int
    source_file: str
    chunk_index: int
    contains_citation: bool

# ──────────────────────────────────────────────────────────────────────────────
# Citation & Metadata Utilities
# ──────────────────────────────────────────────────────────────────────────────

def extract_citations(text: str) -> list[str]:
    found: list[str] = []
    for pattern in ALL_CITATION_PATTERNS:
        found.extend(m.group() for m in pattern.finditer(text))
    seen: set[str] = set()
    unique: list[str] = []
    for c in found:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def detect_jurisdiction(text: str, filename: str) -> tuple[str, str]:
    if re.search(r"\bHCA\b", text) or "high-court" in filename.lower():
        citations = re.findall(r"\[\d{4}\]\s*HCA\s*\d+", text)
        return "High Court of Australia", (citations[0] if citations else "")

    if re.search(r"\bFCAFC\b|\bFCA\b", text):
        citations = re.findall(r"\[\d{4}\]\s*FCA(?:FC)?\s*\d+", text)
        return "Federal Court of Australia", (citations[0] if citations else "")

    if re.search(r"\bINSC\b", text) or "supreme-court-india" in filename.lower():
        citations = re.findall(r"\d{4}\s+INSC\s+\d+", text)
        return "Supreme Court of India", (citations[0] if citations else "")

    if re.search(r"\bSCC\b", text):
        citations = re.findall(r"\(\d{4}\)\s*\d+\s*SCC\s*\d+", text)
        return "Supreme Court of India", (citations[0] if citations else "")

    if re.search(r"\bAIR\b.*\bSC\b", text):
        citations = re.findall(r"AIR\s+\d{4}\s+SC\s+\d+", text)
        return "Supreme Court of India", (citations[0] if citations else "")

    m = NEUTRAL_PATTERN.search(text)
    if m:
        return "Unknown Jurisdiction", m.group()

    return "Unknown Jurisdiction", ""


def infer_year(citation: str, text: str) -> str:
    m = re.search(r"\b(19|20)\d{2}\b", citation)
    if m:
        return m.group()
    m = re.search(r"\b(19|20)\d{2}\b", text[:500])
    return m.group() if m else "Unknown"


def extract_case_name(text: str, filename: str) -> str:
    head = text[:800]
    m = re.search(
        r"(?:Between\s+)?([A-Z][A-Za-z\s,\.]+)\s+v\s+([A-Z][A-Za-z\s,\.]+)",
        head,
    )
    if m:
        raw = f"{m.group(1).strip()} v {m.group(2).strip()}"
        return re.sub(r"\s{2,}", " ", raw)[:120]

    lines = head.split("\n")
    for line in lines[:15]:
        stripped = line.strip()
        if len(stripped) > 8 and stripped.isupper() and "v" in stripped.split():
            return stripped[:120]

    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ").title()

# ──────────────────────────────────────────────────────────────────────────────
# PDF Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_pdf(pdf_path: Path) -> Optional[LegalDocument]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        log.error(f"Could not open [bold]{pdf_path.name}[/]: {exc}")
        return None

    page_texts: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        page_text = re.sub(r"[ \t]+", " ", page_text)
        page_text = re.sub(r"\n{3,}", "\n\n", page_text)
        page_texts.append(page_text.strip())

    full_text = "\n\n".join(page_texts)

    if not full_text.strip():
        log.warning(f"[yellow]No extractable text in {pdf_path.name}. Scanned PDF?[/]")
        return None

    jurisdiction, citation = detect_jurisdiction(full_text, pdf_path.name)
    year = infer_year(citation, full_text)
    case_name = extract_case_name(full_text, pdf_path.name)

    return LegalDocument(
        file_path=pdf_path,
        case_name=case_name,
        citation=citation,
        court_jurisdiction=jurisdiction,
        year=year,
        full_text=full_text,
        page_texts=page_texts,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Smart Chunking
# ──────────────────────────────────────────────────────────────────────────────

_SEPARATORS = [
    "\n\n", "\n", ". ", ", ", " ", ""
]

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=_SEPARATORS,
    length_function=len,
)

def _page_for_char_offset(page_texts: list[str], offset: int) -> int:
    running = 0
    for i, pt in enumerate(page_texts, start=1):
        running += len(pt) + 2
        if offset < running:
            return i
    return len(page_texts)


def chunk_document(doc: LegalDocument) -> list[LegalChunk]:
    raw_chunks = _splitter.create_documents([doc.full_text])
    chunks: list[LegalChunk] = []

    for idx, lc_doc in enumerate(raw_chunks):
        text: str = lc_doc.page_content
        char_offset = sum(len(c.text) - CHUNK_OVERLAP for c in chunks) if chunks else 0
        page_num = _page_for_char_offset(doc.page_texts, char_offset)

        citations_in_chunk = extract_citations(text)
        chunk_id = f"{doc.file_path.stem}_chunk_{idx:04d}"

        chunks.append(
            LegalChunk(
                chunk_id=chunk_id,
                text=text,
                case_name=doc.case_name,
                citation=doc.citation or (citations_in_chunk[0] if citations_in_chunk else ""),
                court_jurisdiction=doc.court_jurisdiction,
                year=doc.year,
                page_number=page_num,
                source_file=doc.file_path.name,
                chunk_index=idx,
                contains_citation=bool(citations_in_chunk),
            )
        )

    log.info(
        f"  [cyan]{doc.file_path.name}[/] → {len(chunks)} chunks "
        f"| jurisdiction: [green]{doc.court_jurisdiction}[/] "
        f"| citation: [yellow]{doc.citation or 'not found'}[/]"
    )
    return chunks

# ──────────────────────────────────────────────────────────────────────────────
# Local Embeddings (Zero Cost)
# ──────────────────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[LegalChunk]) -> list[list[float]]:
    texts = [c.text for c in chunks]
    
    console.print(f"[bold blue]Loading local AI model ({EMBEDDING_MODEL}) & embedding {len(texts)} chunks...[/]")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # HuggingFace handles its own fast processing, no manual batching needed
    all_embeddings = embedder.embed_documents(texts)
    
    return all_embeddings

# ──────────────────────────────────────────────────────────────────────────────
# ChromaDB Storage
# ──────────────────────────────────────────────────────────────────────────────

def get_or_create_collection(reset: bool = False) -> chromadb.Collection:
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    if reset:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            log.info(f"[red]Collection '{COLLECTION_NAME}' reset.[/]")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def upsert_chunks(
    collection: chromadb.Collection,
    chunks: list[LegalChunk],
    embeddings: list[list[float]],
) -> None:
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "case_name": c.case_name,
            "citation": c.citation,
            "court_jurisdiction": c.court_jurisdiction,
            "year": c.year,
            "page_number": c.page_number,
            "source_file": c.source_file,
            "chunk_index": c.chunk_index,
            "contains_citation": str(c.contains_citation),
        }
        for c in chunks
    ]

    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    log.info(f"  [green]✓ Upserted {len(chunks)} chunks into '{COLLECTION_NAME}'[/]")

# ──────────────────────────────────────────────────────────────────────────────
# Verification Helper
# ──────────────────────────────────────────────────────────────────────────────

def verify_database(sample_query: str = "constitutional law fundamental rights") -> None:
    console.rule("[bold cyan]Database Verification")

    chroma_client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    total = collection.count()

    console.print(Panel(
        f"[bold]Collection:[/] {COLLECTION_NAME}\n"
        f"[bold]Location:[/]   {VECTOR_DB_DIR.resolve()}\n"
        f"[bold]Total chunks:[/] [green]{total}[/]",
        title="📚 Vector Store Summary",
        border_style="cyan",
    ))

    if total == 0:
        console.print("[yellow]No chunks found. Run ingestion first.[/]")
        return

    result = collection.get(include=["metadatas"])
    jurisdictions: dict[str, int] = {}
    cases: set[str] = set()
    for meta in result["metadatas"]:
        j = meta.get("court_jurisdiction", "Unknown")
        jurisdictions[j] = jurisdictions.get(j, 0) + 1
        cases.add(meta.get("case_name", ""))

    table = Table(title="Jurisdiction Breakdown", header_style="bold magenta")
    table.add_column("Jurisdiction", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    for j, count in sorted(jurisdictions.items(), key=lambda x: -x[1]):
        table.add_row(j, str(count))
    console.print(table)

    console.print(f"\n[bold]Sample query:[/] '{sample_query}'")
    
    # Use local HuggingFace model for the search query
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    q_embedding = embedder.embed_query(sample_query)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    res_table = Table(title="Top 3 Semantic Results", header_style="bold blue")
    res_table.add_column("#", width=3)
    res_table.add_column("Case", style="cyan")
    res_table.add_column("Citation", style="yellow")
    res_table.add_column("Jurisdiction", style="green")
    res_table.add_column("Page", justify="right")
    res_table.add_column("Score", justify="right")

    for rank, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        score = round(1 - dist, 4)
        res_table.add_row(
            str(rank),
            meta.get("case_name", "")[:40],
            meta.get("citation", ""),
            meta.get("court_jurisdiction", "")[:30],
            str(meta.get("page_number", "")),
            f"{score:.4f}",
        )

    console.print(res_table)
    console.rule("[bold cyan]Verification Complete")

# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def ingest(data_dir: Path, reset: bool = False) -> int:
    console.rule("[bold green]Evatt AI — Legal Ingestion Engine (LOCAL EDITION)")

    pdf_files = sorted(data_dir.glob("**/*.pdf"))
    if not pdf_files:
        log.warning(f"No PDFs found in [bold]{data_dir.resolve()}[/]. Nothing to do.")
        return 0

    console.print(f"Found [bold]{len(pdf_files)}[/] PDF(s) in [cyan]{data_dir.resolve()}[/]")

    collection = get_or_create_collection(reset=reset)
    all_chunks: list[LegalChunk] = []

    console.rule("[dim]Step 1 / 3 — Extract & Chunk")
    for pdf_path in pdf_files:
        log.info(f"Processing [bold]{pdf_path.name}[/]…")
        doc = extract_pdf(pdf_path)
        if doc is None:
            continue
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    console.print(f"\n[bold]Total chunks to embed:[/] [green]{len(all_chunks)}[/]")

    if not all_chunks:
        log.warning("No chunks produced. Exiting.")
        return 0

    console.rule("[dim]Step 2 / 3 — Embed")
    # Pass only the chunks, the function handles the local model init
    embeddings = embed_chunks(all_chunks)

    console.rule("[dim]Step 3 / 3 — Store")
    upsert_chunks(collection, all_chunks, embeddings)

    console.rule("[bold green]Ingestion Complete")
    console.print(
        f"[bold green]✓[/] {len(all_chunks)} chunks from "
        f"{len(pdf_files)} file(s) stored in [cyan]{VECTOR_DB_DIR.resolve()}[/]"
    )
    return len(all_chunks)

# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evatt AI — Legal RAG Ingestion Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing PDF files (default: ./data)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run database verification after ingestion",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip ingestion and only verify existing DB",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing collection before re-ingesting",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="constitutional law fundamental rights",
        help="Sample query string used during --verify (default: constitutional law…)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    if args.verify_only:
        verify_database(sample_query=args.query)
    else:
        ingest(data_dir=args.data_dir, reset=args.reset)
        if args.verify:
            verify_database(sample_query=args.query)