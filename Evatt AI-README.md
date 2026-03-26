# Evatt AI — Legal RAG Ingestion Engine

Production-grade ingestion pipeline for High Court of Australia and Indian Judiciary PDFs.

---

## Project Layout

```
.
├── ingest.py          # Main ingestion engine
├── requirements.txt   # Python dependencies
├── data/              # ← drop your PDF files here (subdirs supported)
└── vector_db/         # ← auto-created ChromaDB store
```

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI key
export OPENAI_API_KEY="sk-..."     # Windows: set OPENAI_API_KEY=sk-...

# 4. Drop PDFs into ./data
mkdir -p data
cp /path/to/your/legal-pdfs/*.pdf data/

# 5. Run ingestion
python ingest.py

# 6. Verify the database
python ingest.py --verify-only --query "duty of care negligence"
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir PATH` | `./data` | Source directory for PDFs |
| `--verify` | off | Run DB verification after ingestion |
| `--verify-only` | off | Skip ingestion; only verify existing DB |
| `--reset` | off | Wipe collection before re-ingesting |
| `--query TEXT` | constitutional law… | Sample query used during `--verify` |

---

## Supported Citation Formats

| Jurisdiction | Examples |
|---|---|
| High Court of Australia | `[2024] HCA 1`, `(2024) 306 CLR 1` |
| Federal Court of Australia | `[2023] FCA 456`, `[2022] FCAFC 78` |
| Supreme Court of India | `2023 INSC 45`, `(2024) 5 SCC 100`, `AIR 2023 SC 1` |
| Generic Neutral | `[2024] UKSC 1`, `[2023] EWCA Civ 123` |

---

## Metadata Stored Per Chunk

| Field | Description |
|---|---|
| `case_name` | Extracted or inferred case name |
| `citation` | Primary citation string |
| `court_jurisdiction` | Court / jurisdiction label |
| `year` | 4-digit year |
| `page_number` | Page in the source PDF |
| `source_file` | PDF filename |
| `chunk_index` | 0-based position within document |
| `contains_citation` | `"True"` if chunk holds a citation |

---

## Chunking Strategy

- **Size:** 1200 characters, 250 overlap  
- **Separators (coarse → fine):** paragraph → line → sentence → clause → word → char  
- Citations at paragraph boundaries are preserved intact thanks to the overlap window.

---

## Programmatic Use

```python
# Ingest from a custom folder
from ingest import ingest
total = ingest(data_dir=Path("./my_pdfs"), reset=False)

# Verify the DB with a custom query
from ingest import verify_database
verify_database("reasonable person standard tort")
```
