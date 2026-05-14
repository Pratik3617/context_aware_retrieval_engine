"""
Shared pytest fixtures.

Uses the real SentenceTransformerEmbedder — requires the model to be
cached locally (run download_models.py once before running tests).
All paths and chunking parameters come from CHUNK_CFG.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config          import CHUNK_CFG
from src.embeddings      import SentenceTransformerEmbedder, MockVertexAIEmbedder
from src.chunker         import DocumentChunker
from src.vector_store    import FAISSVectorStore
from src.query_expander  import MockGenerativeModel
from src.pipeline        import RAGPipeline


@pytest.fixture(scope="session")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


@pytest.fixture(scope="session")
def mock_vertex_embedder() -> MockVertexAIEmbedder:
    return MockVertexAIEmbedder()


@pytest.fixture(scope="session")
def expander() -> MockGenerativeModel:
    return MockGenerativeModel()


@pytest.fixture(scope="session")
def chunker() -> DocumentChunker:
    return DocumentChunker(
        chunk_size=CHUNK_CFG.chunk_size,
        chunk_overlap=CHUNK_CFG.chunk_overlap,
        strategy=CHUNK_CFG.strategy,
    )


@pytest.fixture(scope="session")
def populated_store(embedder, chunker) -> FAISSVectorStore:
    chunks  = chunker.chunk_file(CHUNK_CFG.document_path)
    store   = FAISSVectorStore(dim=embedder.dim)
    vectors = embedder.embed([c["text"] for c in chunks])
    store.add(vectors, chunks)
    return store


@pytest.fixture(scope="session")
def pipeline(embedder, expander) -> RAGPipeline:
    p = RAGPipeline(embedder=embedder, expander=expander)
    p.ingest_file(CHUNK_CFG.document_path)
    return p
