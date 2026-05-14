"""
Tests for the embedding layer.

Run directly:   python tests/test_embeddings.py
Run via pytest: pytest tests/test_embeddings.py -v
"""
import numpy as np
import pytest

from src.embeddings import SentenceTransformerEmbedder, MockVertexAIEmbedder
from dotenv import load_dotenv

class TestSentenceTransformerEmbedder:
    def test_single_text_shape(self, embedder):
        assert embedder.embed(["hello world"]).shape == (1, embedder.dim)

    def test_batch_shape(self, embedder):
        assert embedder.embed(["a", "b", "c"]).shape == (3, embedder.dim)

    def test_dtype_float32(self, embedder):
        assert embedder.embed(["check dtype"]).dtype == np.float32

    def test_l2_normalised(self, embedder):
        vecs  = embedder.embed(["norm test", "another"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)

    def test_dim_property_matches_output(self, embedder):
        assert embedder.embed(["dim check"]).shape[1] == embedder.dim

    def test_semantic_similarity_ordering(self, embedder):
        """Semantically related pair should score higher than unrelated pair."""
        query = embedder.embed(["vector similarity search"])[0]
        related = embedder.embed(["semantic nearest neighbour retrieval"])[0]
        unrelated = embedder.embed(["the capital of France is Paris"])[0]

        related_score = np.dot(query, related)
        unrelated_score = np.dot(query, unrelated)

        assert related_score > unrelated_score


class TestMockVertexAIEmbedder:
    def test_delegates_to_sdk(self, mock_vertex_embedder):
        mock_vertex_embedder.embed(["test delegation"])
        mock_vertex_embedder.sdk_model.get_embeddings.assert_called_once()

    def test_output_shape(self, mock_vertex_embedder):
        assert mock_vertex_embedder.embed(["a", "b"]).shape == (2, mock_vertex_embedder.dim)

    def test_output_normalised(self, mock_vertex_embedder):
        vec = mock_vertex_embedder.embed(["norm check"])[0]
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_dtype_float32(self, mock_vertex_embedder):
        assert mock_vertex_embedder.embed(["dtype"]).dtype == np.float32

    def test_call_count_increments(self, mock_vertex_embedder):
        from src.embeddings import MockVertexAIEmbedder as MVE
        fresh = MVE()
        fresh.embed(["first"])
        fresh.embed(["second"])
        assert fresh.sdk_model.get_embeddings.call_count == 2

    def test_dim_matches_sentence_transformer(self, embedder, mock_vertex_embedder):
        assert embedder.dim == mock_vertex_embedder.dim


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
