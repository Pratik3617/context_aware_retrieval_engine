"""
Embedding abstractions.
Class hierarchy:
    BaseEmbedder               (ABC — defines the interface)
    ├── SentenceTransformerEmbedder   local model, used in production and benchmark
    └── MockVertexAIEmbedder          mocks vertexai.TextEmbeddingModel SDK surface
"""

import abc
import numpy as np
from numpy.typing import NDArray
from typing import List
from unittest.mock import MagicMock
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_CFG

class BaseEmbedder(abc.ABC):
    """
    Interface that every embeder must implement.
    """
    @abc.abstractmethod
    def embed(self, texts: List[str]) -> NDArray[np.float32]:
        """Return L2-normalised embeddings, shape (len(texts), dim)."""
    
    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality"""

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local embedder backed by sentence-transformers.
    Mimics the behaviour of Vertex AI text embedding-gecko for offline use.
    """

    def __init__(self, model_name: str = EMBEDDING_CFG.model_name)-> None:
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim
    
    def embed(self, texts: List[str]) -> NDArray[np.float32]:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True, # L2-norm -> cosine via inner product
            show_progress_bar=False
        ).astype(np.float32)

        return vectors
    
class MockVertexAIEmbedder(BaseEmbedder):
    """
    Mocks ``vertexai.language_models.TextEmbeddingModel``.

    The underlying computation is still done by sentence-transformers so that
    retrieval results are meaningful.  The mock wraps the real model inside
    the Vertex AI SDK surface so tests can assert on SDK call patterns.
    """

    def __init__(self) -> None:
        self._real_embedder = SentenceTransformerEmbedder()

        # build a mock that mirrors the vertex AI interface
        self._sdk_model = MagicMock(name="TextEmbeddingModel")
        self._sdk_model.get_embeddings.side_effect = self._sdk_get_embeddings

    @property
    def dim(self) -> int:
        return self._real_embedder.dim
    
    # Internal Helpers

    def _sdk_get_embeddings(self, texts: List[str]):
        """
        Simulates the list[TextEmbedding] response from the Vertex AI SDK.
        """
        vectors = self._real_embedder.embed(texts)
        mock_responses = []
        for vec in vectors:
            resp = MagicMock()
            resp.values = vec.tolist()
            mock_responses.append(resp)

        return mock_responses
    
    # Public API
    def embed(self, texts: List[str]) -> NDArray[np.float32]:
        sdk_responses = self._sdk_model.get_embeddings(texts)
        vectors = np.array(
            [r.values for r in sdk_responses], dtype=np.float32
        )
        # Normalise in case the mock path skips it (defensive)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms
    
    @property
    def sdk_model(self) -> MagicMock:
        """Expose the mock for test assertions."""
        return self._sdk_model
