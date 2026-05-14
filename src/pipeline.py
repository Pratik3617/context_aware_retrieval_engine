"""
RAGPipeline - orchestrates document ingrstion and retrieval.

Ingestion:
    file path -> DocumentChunker -> List[{id, text}] -> embed -> FAISSVectorStore

Retrieval:
    Strategy A - embed(query) -> search
    Strategy B - expand query -> embed -> search

"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config import CHUNK_CFG, RETRIEVAL_CFG
from src.chunker import DocumentChunker
from src.embeddings import BaseEmbedder
from src.query_expander import MockGenerativeModel
from src.vector_store import FAISSVectorStore, SearchResult

@dataclass
class RetrievalOutput:
    query: str
    strategy: str
    expanded_query: Optional[str]
    results: List[SearchResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "expanded_query": self.expanded_query,
            "results": [
                {
                    "rank": r.rank,
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 6),
                    "text_preview": r.text[:120] + ("..." if len(r.text) > 120 else "")
                }
                for r in self.results
            ]
        }
    
class RAGPipeline:
    """
    Args:
        embedder: Any BaseEmbedder implementation.
        expander: MockGenerativeModel for Strategy B query expansion.
        chunker:  DocumentChunker instance. Defaults to one built from CHUNK_CFG.
        top_k:    Chunks returned per query. Defaults to RETRIEVAL_CFG.top_k.
    """

    def __init__(
            self,
            embedder: BaseEmbedder,
            expander: MockGenerativeModel,
            chunker:  Optional[DocumentChunker] = None,
            top_k:    int = RETRIEVAL_CFG.top_k,
        ) -> None:
        self._embedder = embedder
        self._expander = expander
        self._chunker = chunker or DocumentChunker(
            chunk_size=CHUNK_CFG.chunk_size,
            chunk_overlap=CHUNK_CFG.chunk_overlap,
            strategy=CHUNK_CFG.strategy,
        )
        self._top_k  = top_k
        self._store  = FAISSVectorStore(dim=embedder.dim)
        self._chunks: List[Dict[str, str]] = []
    
    # Ingestion
    def ingest_file(self, file_path: str | Path) -> int:
        """Chunk a .txt/.pdf file, embed and index. Returns chunk count."""
        return self._ingest(self._chunker.chunk_file(file_path))
    
    def ingest_text(self, text: str) -> int:
        """Chunk a raw string, embed and index. Returns chunk count."""
        return self._ingest(self._chunker.chunk_text(text))
    
    # Retrieval
    def retrieve_strategy_a(self, query: str) -> RetrievalOutput:
        """Strategy A — Raw Vector Search: embed query directly, then search."""
        self._require_ingested()
        vec     = self._embedder.embed([query])[0]
        results = self._store.search(vec, self._top_k)
        return RetrievalOutput(query=query, strategy="A", expanded_query=None, results=results)
    
    def retrieve_strategy_b(self, query: str) -> RetrievalOutput:
        """Strategy B — AI-Enhanced: expand via mock LLM, then embed and search."""
        self._require_ingested()
        expanded = self._expander.expand(query)
        vec      = self._embedder.embed([expanded])[0]
        results  = self._store.search(vec, self._top_k)
        return RetrievalOutput(query=query, strategy="B", expanded_query=expanded, results=results)
    
    # Properties
    @property
    def store_size(self) -> int:
        return self._store.size

    @property
    def is_ingested(self) -> bool:
        return len(self._chunks) > 0

    @property
    def chunks(self) -> List[Dict[str, str]]:
        return list(self._chunks)
    
    # Private methods
    def _ingest(self, chunks: List[Dict[str,str]]) -> int:
        if not chunks:
            raise ValueError("Chunker produced zero chunks - check input document")
        
        vectors = self._embedder.embed([c["text"] for c in chunks])
        self._store.add(vectors, chunks)
        self._chunks.extend(chunks)
        return len(chunks)
    
    def _require_ingested(self) -> None:
        if not self.is_ingested:
            raise RuntimeError(
                "Pipeline has no indexed data."
                "Call ingest_file() or ingest_text() first."
            )