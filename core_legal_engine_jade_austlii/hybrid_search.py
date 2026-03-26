import logging
import chromadb
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings

# Setup logging
log = logging.getLogger("evatt.hybrid_search")

# Data structures required by noteup_api.py
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5
    jurisdiction: Optional[str] = None

class SearchHit(BaseModel):
    document: str
    metadata: Dict[str, Any]
    score: float

class HybridSearchEngine:
    """
    100% Local Search Engine for Evatt AI.
    Zero OpenAI dependencies. Zero API costs.
    """
    def __init__(self, vector_db_dir="../vector_db", collection_name="legal_documents"):
        log.info(f"Initializing Local Hybrid Search Engine (all-MiniLM-L6-v2)...")
        
        # 1. Connect to the existing ChromaDB
        db_path = Path(vector_db_dir).resolve()
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # 2. Load the Local Model (384 Dimensions)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        log.info("✓ Local Embedder Loaded (384 Dimensions)")

    def warm_bm25(self):
        """Warms up the BM25 index. Local version skips this as it uses pure semantic search."""
        log.info("✓ BM25 Warming (Local Stub Active)")
        pass

    def semantic_search(self, query: str, n_results: int = 5) -> List[SearchHit]:
        """Translates query to 384-dim vector and searches ChromaDB locally."""
        query_vector = self.embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        hits = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                hits.append(SearchHit(
                    document=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=results['distances'][0][i]
                ))
        return hits

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        """Standard search method called by noteup_api."""
        return self.semantic_search(query, n_results=top_k)

_engine = None

def get_search_engine():
    global _engine
    if _engine is None:
        _engine = HybridSearchEngine()
    return _engine
