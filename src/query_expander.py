"""
Semantic query expansion via a mocked Vertex AI GenerativeModel.

Purpose
-------
Improves semantic retrieval quality by expanding user queries with:
    - domain terminology
    - architectural vocabulary
    - implementation-related concepts
    - synonymous technical phrases

Mocked SDK Surface
------------------
Mirrors:
    vertexai.generative_models.GenerativeModel

Usage:
    model.generate_content(prompt) -> response.text
"""

from __future__ import annotations

import re
from typing import Dict, List
from unittest.mock import MagicMock


# Semantic Expansion Knowledge Base
QUERY_EXPANSIONS: Dict[str, List[str]] = {

    # Scalability / Load Handling
    "peak load": [
        "traffic spikes",
        "autoscaling",
        "horizontal scaling",
        "load balancing",
        "high concurrency",
        "throughput optimization",
        "resource allocation",
        "distributed scaling"
    ],

    "high traffic": [
        "autoscaling",
        "load balancing",
        "traffic spikes",
        "distributed systems",
        "horizontal scaling"
    ],

    # Vector Search / Retrieval
    "vector search latency": [
        "approximate nearest neighbor",
        "ANN indexing",
        "FAISS optimization",
        "embedding retrieval",
        "low latency similarity search",
        "semantic retrieval throughput",
        "vector indexing performance",
        "dense retrieval optimization"
    ],

    "retrieval latency": [
        "ANN search",
        "vector database optimization",
        "embedding similarity search",
        "low latency retrieval",
        "query throughput"
    ],

    "semantic search": [
        "vector embeddings",
        "dense retrieval",
        "similarity matching",
        "FAISS index",
        "embedding search"
    ],

    # Distributed Systems / Consistency
    "data consistency": [
        "replication consistency",
        "synchronized ingestion",
        "distributed data synchronization",
        "consistent state management",
        "distributed write coordination"
    ],

    "distributed ingestion": [
        "parallel data pipelines",
        "replication",
        "partitioning",
        "distributed coordination",
        "stream ingestion",
        "fault tolerant ingestion"
    ],

    # Authentication / Security
    "authentication": [
        "identity verification",
        "token validation",
        "authorization",
        "access control",
        "JWT tokens",
        "OAuth",
        "user credentials"
    ],

    "security": [
        "authentication",
        "authorization",
        "encryption",
        "access control",
        "secure communication",
        "identity management"
    ],

    # Performance / Caching
    "slow response": [
        "latency optimization",
        "caching",
        "performance bottlenecks",
        "query optimization",
        "throughput improvement"
    ],

    "performance": [
        "throughput",
        "latency optimization",
        "resource efficiency",
        "scalability",
        "system optimization"
    ],

    # Reliability / Fault Tolerance
    "fault tolerance": [
        "high availability",
        "system resilience",
        "redundancy",
        "failure recovery",
        "replication",
        "distributed recovery"
    ]
}


# Core Expansion Logic
def _simulate_llm_expansion(query: str) -> str:
    """
    Simulates an LLM-based semantic query expansion.

    Strategy:
        1. Preserve original query
        2. Detect semantic intent using trigger phrases
        3. Inject relevant architectural terminology
        4. Return enriched embedding-friendly query

    Args:
        query:
            Original user query

    Returns:
        Expanded semantic query string
    """

    query_clean = query.strip()
    query_lower = query_clean.lower()

    expanded_terms: List[str] = []

    # Match semantic triggers
    for trigger, terms in QUERY_EXPANSIONS.items():
        if trigger in query_lower:
            expanded_terms.extend(terms)

    # Fallback expansion if no trigger matched
    if not expanded_terms:
        expanded_terms.extend([
            "system design",
            "architecture",
            "implementation",
            "scalability",
            "performance optimization"
        ])

    # Deduplicate while preserving order
    seen = set()
    unique_terms = []

    for term in expanded_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)

    expanded_query = (
        f"{query_clean}. "
        f"Related concepts: {' '.join(unique_terms)}"
    )

    return expanded_query


# Mock Vertex AI GenerativeModel
class MockGenerativeModel:
    """
    Mocks vertexai.generative_models.GenerativeModel.

    SDK Compatibility:
        model.generate_content(prompt) -> response.text

    Internally:
        Uses semantic expansion logic to simulate
        LLM-powered query rewriting.
    """

    def __init__(self, model_name: str = "gemini-1.5-pro-001") -> None:

        self._model_name = model_name

        self._sdk = MagicMock(
            name=f"GenerativeModel({model_name})"
        )

        self._sdk.generate_content.side_effect = (
            self._generate_content
        )

    # Internal SDK Mock
    def _generate_content(self, prompt: str) -> MagicMock:

        raw_query = self._extract_query_from_prompt(prompt)

        expanded_query = _simulate_llm_expansion(raw_query)

        response = MagicMock()
        response.text = expanded_query

        return response

    @staticmethod
    def _extract_query_from_prompt(prompt: str) -> str:
        """
        Extract quoted query from prompt.

        Supports:
            "query"
            “query”
        """

        match = re.search(
            r'["\u201c\u201d](.*?)["\u201c\u201d]',
            prompt
        )

        return match.group(1) if match else prompt.strip()

    # Public API
    def expand(self, query: str) -> str:
        """
        Expand a user query using mocked SDK flow.
        """

        prompt = (
            "Rewrite and expand the following search query "
            "to include related technical terminology for "
            "better semantic retrieval.\n\n"
            f'Query: "{query}"'
        )

        return (
            self._sdk
            .generate_content(prompt)
            .text
            .strip()
        )

    # Properties
    @property
    def sdk_model(self) -> MagicMock:
        """
        Expose mocked SDK for unit testing.
        """
        return self._sdk

    @property
    def model_name(self) -> str:
        return self._model_name