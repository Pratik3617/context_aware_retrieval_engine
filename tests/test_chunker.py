"""
Tests for Document Chunker - both strategies, file reading, error cases.
"""

import os
import tempfile
import pytest
from src.chunker import DocumentChunker
from src.config import CHUNK_CFG, ChunkStrategy

SAMPLE_TEXT = (
    "Distributed systems must balance consistency and availability.\n\n"
    "Vector databases store high-dimensional embeddings for semantic search. "
    "FAISS and ChromaDB are popular open-source options.\n\n"
    "Query expansion improves retrieval recall by augmenting the user query "
    "with semantically related terms before embedding the query vector."
)

class TestRecursiveSplit:
    def test_produces_chunks(self):
        chunks = DocumentChunker(chunk_size=200, chunk_overlap=50).chunk_text(SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_chunk_size_respected(self):
        chunks = DocumentChunker(chunk_size=150, chunk_overlap=30).chunk_text(SAMPLE_TEXT)
        for ch in chunks:
            assert len(ch["text"]) <= 200

    def test_no_leading_newlines(self):
        chunks = DocumentChunker(chunk_size=300, chunk_overlap=50).chunk_text(SAMPLE_TEXT)
        for ch in chunks:
            assert not ch["text"].startswith("\n")

    def test_ids_unique(self):
        chunks = DocumentChunker(chunk_size=200, chunk_overlap=50).chunk_text(SAMPLE_TEXT)
        ids = [ch["id"] for ch in chunks]
        assert len(ids) == len(set(ids))

    def test_ids_sequential(self):
        chunks = DocumentChunker(chunk_size=200, chunk_overlap=50).chunk_text(SAMPLE_TEXT)
        assert [ch["id"] for ch in chunks] == [f"chunk_{i:04d}" for i in range(len(chunks))]

    def test_no_empty_chunks(self):
        chunks = DocumentChunker(chunk_size=200, chunk_overlap=50).chunk_text(SAMPLE_TEXT)
        assert all(ch["text"].strip() for ch in chunks)


class TestSlidingWindow:
    def test_produces_chunks(self):
        chunks = DocumentChunker(
            chunk_size=200, chunk_overlap=50, strategy=ChunkStrategy.SLIDING_WINDOW
        ).chunk_text(SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_chunk_size_hard_limit(self):
        chunks = DocumentChunker(
            chunk_size=150, chunk_overlap=40, strategy=ChunkStrategy.SLIDING_WINDOW
        ).chunk_text(SAMPLE_TEXT)
        for ch in chunks:
            assert len(ch["text"]) <= 150

    def test_overlap_produces_shared_tokens(self):
        words  = " ".join([f"word{i}" for i in range(200)])
        chunks = DocumentChunker(
            chunk_size=100, chunk_overlap=40, strategy=ChunkStrategy.SLIDING_WINDOW
        ).chunk_text(words)
        if len(chunks) >= 2:
            tail = set(chunks[0]["text"].split()[-5:])
            head = set(chunks[1]["text"].split()[:10])
            assert tail & head


class TestFileIngestion:
    def test_returns_records(self, chunker):
        chunks = chunker.chunk_file(CHUNK_CFG.document_path)
        assert len(chunks) > 0
        assert all("id" in ch and "text" in ch for ch in chunks)

    def test_reasonable_chunk_count(self, chunker):
        assert len(chunker.chunk_file(CHUNK_CFG.document_path)) >= 5

    def test_key_terms_present(self, chunker):
        combined = " ".join(ch["text"] for ch in chunker.chunk_file(CHUNK_CFG.document_path)).lower()
        for term in ["faiss", "embedding", "retrieval", "consistency", "kubernetes"]:
            assert term in combined

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            DocumentChunker().chunk_file("/nonexistent/file.txt")

    def test_unsupported_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            f.write(b"dummy")
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                DocumentChunker().chunk_file(tmp)
        finally:
            os.unlink(tmp)

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="empty"):
            DocumentChunker().chunk_text("   ")


class TestValidation:
    def test_overlap_equal_to_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            DocumentChunker(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            DocumentChunker(chunk_size=100, chunk_overlap=200)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
