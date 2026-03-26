from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from hybrid_search import get_search_engine

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("evatt.noteup_api")

app = FastAPI(title="Evatt AI Local API")

# Fix 2: The Security Gate (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

@app.on_event("startup")
async def startup_event():
    log.info("Initialising Hybrid Search Engine...")
    engine = get_search_engine()
    engine.warm_bm25()

@app.post("/ask")
async def ask(payload: SearchQuery):
    try:
        engine = get_search_engine()
        # This uses your 23M parameter local model!
        hits = engine.search(payload.query, top_k=payload.top_k)
        
        # Build the 'answer' string that the frontend is looking for
        citation_lines = "\n".join([f"• {h.document[:100]}..." for h in hits])
        fallback_answer = f"I found {len(hits)} relevant legal chunks in the local database.\n\nKey results:\n{citation_lines}"

        # THE CRITICAL FIX: Ensure 'answer' is always returned
        return {
            "answer": fallback_answer,
            "hits": [
                {
                    "case_name": h.metadata.get("case_name", "Unknown Case"),
                    "neutral_citation": h.metadata.get("neutral_citation", "N/A"),
                    "year": h.metadata.get("year", "N/A"),
                    "document": h.document,
                    "score": h.score
                } for h in hits
            ]
        }
    except Exception as e:
        log.error(f"Search failed: {e}")
        return {"answer": "Error: Could not retrieve local results.", "hits": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
