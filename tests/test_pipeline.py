"""
Tests for RAGPipeline — ingestion, Strategy A, Strategy B, serialisation.

Run directly:   python tests/test_pipeline.py
Run via pytest: pytest tests/test_pipeline.py -v
"""
import json

import pytest

from src.config          import CHUNK_CFG
from src.embeddings      import SentenceTransformerEmbedder
from src.pipeline        import RAGPipeline, RetrievalOutput
from src.query_expander  import MockGenerativeModel


class TestIngestion:
    def test_ingest_file_returns_positive_count(self, pipeline):
        assert pipeline.store_size > 0

    def test_is_ingested_flag(self):
        p = RAGPipeline(
            embedder=SentenceTransformerEmbedder(),
            expander=MockGenerativeModel(),
        )
        assert not p.is_ingested
        p.ingest_file(CHUNK_CFG.document_path)
        assert p.is_ingested

    def test_ingest_text_works(self, embedder, expander):
        p = RAGPipeline(embedder=embedder, expander=expander)
        n = p.ingest_text(
            "Horizontal scaling distributes load across replicas. "
            "Vector databases index embeddings for fast retrieval. "
            "Query expansion augments search queries with related terms."
        )
        assert n > 0
        assert p.store_size == n

    def test_chunks_accessible(self, pipeline):
        assert len(pipeline.chunks) == pipeline.store_size

    def test_chunks_have_required_keys(self, pipeline):
        assert all("id" in ch and "text" in ch for ch in pipeline.chunks)

    def test_empty_text_raises(self, embedder, expander):
        p = RAGPipeline(embedder=embedder, expander=expander)
        with pytest.raises((ValueError, RuntimeError)):
            p.ingest_text("   ")

    def test_missing_file_raises(self, embedder, expander):
        p = RAGPipeline(embedder=embedder, expander=expander)
        with pytest.raises(FileNotFoundError):
            p.ingest_file("/nonexistent/document.txt")


class TestStrategyA:
    def test_returns_retrieval_output(self, pipeline):
        assert isinstance(pipeline.retrieve_strategy_a("peak load"), RetrievalOutput)

    def test_strategy_label(self, pipeline):
        assert pipeline.retrieve_strategy_a("test").strategy == "A"

    def test_no_expanded_query(self, pipeline):
        assert pipeline.retrieve_strategy_a("test").expanded_query is None

    def test_top_k_results(self, pipeline):
        assert len(pipeline.retrieve_strategy_a("vector database").results) == 3

    def test_raises_before_ingest(self, embedder, expander):
        p = RAGPipeline(embedder=embedder, expander=expander)
        with pytest.raises(RuntimeError, match="ingest"):
            p.retrieve_strategy_a("anything")


class TestStrategyB:
    def test_returns_retrieval_output(self, pipeline):
        assert isinstance(pipeline.retrieve_strategy_b("peak load"), RetrievalOutput)

    def test_strategy_label(self, pipeline):
        assert pipeline.retrieve_strategy_b("test").strategy == "B"

    def test_expanded_query_populated(self, pipeline):
        result = pipeline.retrieve_strategy_b("How does the system handle peak load?")
        assert result.expanded_query is not None
        assert len(result.expanded_query) >= len(result.query)

    def test_top_k_results(self, pipeline):
        assert len(pipeline.retrieve_strategy_b("vector search latency").results) == 3

    def test_expander_called(self):
        expander = MockGenerativeModel()
        p = RAGPipeline(
            embedder=SentenceTransformerEmbedder(),
            expander=expander,
        )
        p.ingest_file(CHUNK_CFG.document_path)
        p.retrieve_strategy_b("peak load")
        expander.sdk_model.generate_content.assert_called_once()

    def test_raises_before_ingest(self, embedder, expander):
        p = RAGPipeline(embedder=embedder, expander=expander)
        with pytest.raises(RuntimeError, match="ingest"):
            p.retrieve_strategy_b("anything")


class TestSerialization:
    def test_to_dict_json_serialisable(self, pipeline):
        json.dumps(pipeline.retrieve_strategy_a("embedding model").to_dict())

    def test_to_dict_keys(self, pipeline):
        d = pipeline.retrieve_strategy_b("vector search").to_dict()
        assert set(d.keys()) == {"query", "strategy", "expanded_query", "results"}

    def test_result_entry_keys(self, pipeline):
        first = pipeline.retrieve_strategy_a("consistency").to_dict()["results"][0]
        assert {"rank", "chunk_id", "score", "text_preview"}.issubset(first.keys())


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
