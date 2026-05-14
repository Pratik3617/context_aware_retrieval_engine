"""
Tests for MockGenerativeModel query expander.

The mock simulates GenerativeModel output via pure string formatting —
no static word lists, no regex pattern matching, no NLP preprocessing.

Run directly:   python tests/test_query_expander.py
Run via pytest: pytest tests/test_query_expander.py -v
"""
from unittest.mock import MagicMock

import pytest

from src.query_expander import MockGenerativeModel, _simulate_llm_expansion


class TestSimulateLLMExpansion:
    QUERIES = [
        "How does the system handle peak load?",
        "What are the strategies for optimizing vector search latency?",
        "How is data consistency maintained during distributed ingestion?",
        "explain transformer attention mechanism",
        "FAISS index types",
        "why use cosine similarity",
        "Byzantine fault tolerance in Paxos consensus protocol",
        "kubernetes horizontal pod autoscaler",
    ]

    def test_always_longer_than_input(self):
        for q in self.QUERIES:
            expanded = _simulate_llm_expansion(q)
            assert len(expanded.split()) > len(q.split()), (
                f"Expansion not longer for: {q!r}"
            )

    def test_original_query_preserved(self):
        q = "How does the system handle peak load?"
        assert q in _simulate_llm_expansion(q)

    def test_contains_all_three_reformulations(self):
        expanded = _simulate_llm_expansion("vector search latency")
        assert "system implementation" in expanded
        assert "concepts methods best practices" in expanded
        assert "architecture design performance" in expanded

    def test_deterministic(self):
        q = "distributed transaction consistency"
        assert _simulate_llm_expansion(q) == _simulate_llm_expansion(q)

    def test_different_queries_produce_different_expansions(self):
        assert (
            _simulate_llm_expansion("peak load autoscaling")
            != _simulate_llm_expansion("product quantization FAISS index")
        )

    def test_returns_non_empty_string(self):
        assert isinstance(_simulate_llm_expansion("any query"), str)
        assert _simulate_llm_expansion("x").strip() != ""

    def test_no_static_preprocessing(self):
        """
        Queries with unusual vocabulary that would trip up stop-word
        removal or keyword matching must still produce a longer expansion.
        """
        q = "raft consensus leader election timeout quorum"
        assert len(_simulate_llm_expansion(q).split()) > len(q.split())


class TestMockGenerativeModel:
    def test_expand_calls_sdk(self):
        m = MockGenerativeModel()
        m.expand("How does the system handle peak load?")
        m.sdk_model.generate_content.assert_called_once()

    def test_prompt_contains_original_query(self):
        m     = MockGenerativeModel()
        query = "vector search latency"
        m.expand(query)
        prompt = m.sdk_model.generate_content.call_args[0][0]
        assert query in prompt

    def test_expansion_longer_than_query(self):
        m     = MockGenerativeModel()
        query = "How does the system handle peak load?"
        assert len(m.expand(query).split()) > len(query.split())

    def test_call_count_per_expand(self):
        m = MockGenerativeModel()
        for i in range(4):
            m.expand(f"query number {i}")
        assert m.sdk_model.generate_content.call_count == 4

    def test_sdk_is_magic_mock(self):
        assert isinstance(MockGenerativeModel().sdk_model, MagicMock)

    def test_model_name_stored(self):
        assert MockGenerativeModel(model_name="gemini-pro").model_name == "gemini-pro"

    def test_different_queries_different_expansions(self):
        m = MockGenerativeModel()
        assert m.expand("peak load autoscaling") != m.expand("vector quantization FAISS")

    def test_same_query_deterministic(self):
        m = MockGenerativeModel()
        q = "distributed ingestion consistency"
        assert m.expand(q) == m.expand(q)

    def test_does_not_use_real_vertex_sdk(self):
        m = MockGenerativeModel()
        m.expand("test isolation")
        assert isinstance(m.sdk_model, MagicMock)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
