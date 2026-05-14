"""
Tests for FAISSVectorStore — add/search, score ordering, edge cases.

Run directly:   python tests/test_vector_store.py
Run via pytest: pytest tests/test_vector_store.py -v
"""
import numpy as np
import pytest

from src.vector_store import FAISSVectorStore, SearchResult

SMALL_CORPUS = [
    {"id": "s1", "text": "Fast vector search with FAISS indexing."},
    {"id": "s2", "text": "Autoscaling handles peak load in Kubernetes."},
    {"id": "s3", "text": "Embeddings capture semantic meaning of text."},
    {"id": "s4", "text": "Write-ahead logs ensure durability of writes."},
    {"id": "s5", "text": "Query expansion adds related terms to the search."},
]


@pytest.fixture
def store(embedder):
    return FAISSVectorStore(dim=embedder.dim)


class TestFAISSVectorStore:
    def test_initial_size_zero(self, embedder):
        assert FAISSVectorStore(dim=embedder.dim).size == 0

    def test_add_increments_size(self, store, embedder):
        vecs = embedder.embed([c["text"] for c in SMALL_CORPUS])
        store.add(vecs, SMALL_CORPUS)
        assert store.size == len(SMALL_CORPUS)

    def test_search_returns_top_k(self, populated_store, embedder):
        qv = embedder.embed(["vector search"])[0]
        assert len(populated_store.search(qv, top_k=3)) == 3

    def test_results_are_search_result_instances(self, populated_store, embedder):
        qv = embedder.embed(["embedding model"])[0]
        assert all(isinstance(r, SearchResult) for r in populated_store.search(qv, top_k=2))

    def test_scores_descending(self, populated_store, embedder):
        qv     = embedder.embed(["autoscaling peak load"])[0]
        scores = [r.score for r in populated_store.search(qv, top_k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_rank_sequential_from_1(self, populated_store, embedder):
        qv    = embedder.embed(["FAISS index"])[0]
        ranks = [r.rank for r in populated_store.search(qv, top_k=3)]
        assert ranks == [1, 2, 3]

    def test_top_k_capped_at_store_size(self, embedder):
        s    = FAISSVectorStore(dim=embedder.dim)
        vecs = embedder.embed([c["text"] for c in SMALL_CORPUS])
        s.add(vecs, SMALL_CORPUS)
        assert len(s.search(vecs[0], top_k=100)) == len(SMALL_CORPUS)

    def test_empty_store_raises(self, store, embedder):
        qv = embedder.embed(["anything"])[0]
        with pytest.raises(RuntimeError, match="empty"):
            store.search(qv, top_k=3)

    def test_shape_mismatch_on_add_raises(self, store, embedder):
        bad = np.random.rand(2, embedder.dim + 1).astype(np.float32)
        with pytest.raises(ValueError, match="Expected shape"):
            store.add(bad, [{"id": "x", "text": "a"}, {"id": "y", "text": "b"}])

    def test_metadata_length_mismatch_raises(self, store, embedder):
        vecs = embedder.embed(["a", "b"])
        with pytest.raises(ValueError, match="mismatch"):
            store.add(vecs, [{"id": "one", "text": "only one"}])

    def test_scores_bounded(self, populated_store, embedder):
        qv = embedder.embed(["bounds check"])[0]
        for r in populated_store.search(qv, top_k=5):
            assert -1.01 <= r.score <= 1.01


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
