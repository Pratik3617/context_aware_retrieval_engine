"""
FAISS-backed vector store.

Design:
  - Stores L2-normalised vectors under an IndexFlatIP (inner-product) index.
    Since vectors are unit-length, inner-product == cosine similarity.
  - Keeps a parallel list of metadata dicts (id + text) in insertion order.
  - No external state; the index lives entirely in memory.

Why cosine over Euclidean for language embeddings:
  Euclidean distance conflates magnitude with direction.  Language model
  embeddings vary in magnitude across sequences of different lengths, so
  cosine similarity (direction only) gives more stable semantic rankings.
  Euclidean would be preferred for metric-learning tasks where absolute
  distances are calibrated (e.g., face verification).
"""

import faiss
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    score: float
    rank: int    # 1-indexed

class FAISSVectorStore:
    """
    Thread-safe-read, single-writer FAISS store with metadata tracking.
    """
    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._metadata: List[Dict[str, Any]] = []


    # Ingestion
    def add(self, vectors: NDArray[np.float32], metadata: List[Dict[str, Any]]) -> None:
        """
        Add pre-normalised embeddings with corresponding metadata.

        Args:
            vectors:  shape (n, dim), L2-normalised float32
            metadata: list of length n; each entry must contain 'id' and 'text'
        """
        if len(vectors) != len(metadata):
            raise ValueError(
                f"vectors and metadata length mismatch: {len(vectors)} vs {len(metadata)}"
            )
        self._assert_shape(vectors)
        self._index.add(vectors)
        self._metadata.extend(metadata)

    # Retrieval
    def search(self, query_vector: NDArray[np.float32], top_k: int) -> List[SearchResult]:
        """
        Return the top-k most similar chunks for a single query vector.

        Args:
            query_vector: shape (dim,) or (1, dim), L2-normalised float32
            top_k:        number of results to return

        Returns:
            List of SearchResult sorted by descending cosine score.
        """
        if self._index.ntotal == 0:
            raise RuntimeError("Vector store is empty, call add() before search().")
        
        query_2d = query_vector.reshape(1, -1).astype(np.float32)
        self._assert_shape(query_2d)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_2d, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0],scores[0]), start=1):
            meta = self._metadata[idx]
            results.append(
                SearchResult(
                    chunk_id=meta["id"],
                    text=meta["text"],
                    score=float(score),
                    rank=rank
                )
            )
        return results
    
    # Properties

    @property
    def size(self) -> int:
        return self._index.ntotal

    @property
    def dim(self) -> int:
        return self._dim

    # Internal helpers
    def _assert_shape(self, vectors: NDArray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(
                f"Expected shape (n, {self._dim}), got {vectors.shape}"
            )