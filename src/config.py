"""
configuration file for the RAG pipeline.
All tuneable constants live here.

Covers:
  - EmbeddingConfig  : model name, dimensionality, Vertex AI equivalent
  - ChunkingConfig   : chunk size, overlap, strategy, source document path
  - RetrievalConfig  : top_k, similarity metric
  - BenchmarkConfig  : queries, output file names
"""
import os
from dataclasses import dataclass, field
from typing import List
from enum import Enum

class ChunkStrategy(str, Enum):
    SLIDING_WINDOW = "sliding_window"
    RECURSIVE_SPLIT = "rescursive_split"

@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 350
    chunk_overlap: int = 80
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE_SPLIT

    document_path: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "technical_document.txt"
    )

    def __post_init__(self)-> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})."
            )

@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # vertex AI equivalent referenced in mock
    vertex_model_name: str = 'textembedding-gecko@003'

@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 3
    similarity_metric: str = "cosine"

@dataclass(frozen=True)
class BenchmarkConfig:
    queries: List[str] = field(default_factory=lambda: [
        "How does the system handle peak load?",
        "What are the strategies for optimizing vector search latency?",
        "How is data consistency maintained during distributed ingestion?"
    ])

    output_json: str = "benchmark_results.json"

CHUNK_CFG     = ChunkingConfig()
EMBEDDING_CFG = EmbeddingConfig()
RETRIEVAL_CFG = RetrievalConfig()
BENCHMARK_CFG = BenchmarkConfig()

