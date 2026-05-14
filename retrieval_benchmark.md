# Retrieval Benchmark Report

## Benchmark Configuration

- **Generated At:** 2026-05-14 12:42 UTC
- **Embedding Model:** all-MiniLM-L6-v2
- **Vertex AI Equivalent:** textembedding-gecko@003
- **Chunk Size:** 350
- **Chunk Overlap:** 80
- **Chunk Strategy:** rescursive_split
- **Similarity Metric:** cosine
- **Top-K Retrieval:** 3
- **Corpus Chunks:** 46

---

## Similarity Score Comparison

| Query | Strategy A | Strategy B | Relative Gain |
|---|---|---|---|
| How does the system handle peak load? | 0.6139 | 0.7870 | +28.20% |
| What are the strategies for optimizing vector search latency? | 0.5646 | 0.6952 | +23.11% |
| How is data consistency maintained during distributed ingestion? | 0.6705 | 0.6717 | +0.18% |

**Average Relative Improvement:** +17.16%

---

## Detailed Retrieval Analysis

### Query 1
**Input Query:** `How does the system handle peak load?`

#### Strategy A — Raw Vector Search

- Rank 1 | chunk_0003 | score=0.6139
  - Preview: Handling peak load is one of the most critical operational challenges in distributed infrastructure. When traffic spikes...
- Rank 2 | chunk_0009 | score=0.5376
  - Preview: Engineers use load testing tools such as Locust, k6, or Gatling to drive synthetic traffic at multiples of expected peak...
- Rank 3 | chunk_0006 | score=0.4339
  - Preview: Load balancers distribute incoming requests across available replicas using algorithms such as round-robin, least-connec...

#### Strategy B — AI-Enhanced Retrieval

**Expanded Query:**

> How does the system handle peak load?. Related concepts: traffic spikes autoscaling horizontal scaling load balancing high concurrency throughput optimization resource allocation distributed scaling

- Rank 1 | chunk_0003 | score=0.7870
  - Preview: Handling peak load is one of the most critical operational challenges in distributed infrastructure. When traffic spikes...
- Rank 2 | chunk_0009 | score=0.5831
  - Preview: Engineers use load testing tools such as Locust, k6, or Gatling to drive synthetic traffic at multiples of expected peak...
- Rank 3 | chunk_0004 | score=0.5799
  - Preview: Horizontal scaling, where additional service replicas are spun up in response to demand, is the standard mitigation. Kub...

#### Analysis

- Strategy B improved top-1 similarity by `0.1731`.
- Query expansion introduced additional technical terminology intended to improve semantic alignment between user intent and indexed corpus language.

---

### Query 2
**Input Query:** `What are the strategies for optimizing vector search latency?`

#### Strategy A — Raw Vector Search

- Rank 1 | chunk_0022 | score=0.5646
  - Preview: Vector databases store high-dimensional embeddings and support approximate nearest neighbour queries with sub-millisecon...
- Rank 2 | chunk_0000 | score=0.5614
  - Preview: Distributed Systems and Vector Search: Architecture, Scalability, and Modern Retrieval Techniques
- Rank 3 | chunk_0025 | score=0.5266
  - Preview: FAISS, developed by Meta AI, is the most widely used open-source library for dense vector search. It supports both exact...

#### Strategy B — AI-Enhanced Retrieval

**Expanded Query:**

> What are the strategies for optimizing vector search latency?. Related concepts: approximate nearest neighbor ANN indexing FAISS optimization embedding retrieval low latency similarity search semantic retrieval throughput vector indexing performance dense retrieval optimization

- Rank 1 | chunk_0022 | score=0.6952
  - Preview: Vector databases store high-dimensional embeddings and support approximate nearest neighbour queries with sub-millisecon...
- Rank 2 | chunk_0000 | score=0.6125
  - Preview: Distributed Systems and Vector Search: Architecture, Scalability, and Modern Retrieval Techniques
- Rank 3 | chunk_0023 | score=0.6001
  - Preview: Flat indices perform exact nearest neighbour search by comparing the query vector against every stored vector; this guar...

#### Analysis

- Strategy B improved top-1 similarity by `0.1305`.
- Query expansion introduced additional technical terminology intended to improve semantic alignment between user intent and indexed corpus language.

---

### Query 3
**Input Query:** `How is data consistency maintained during distributed ingestion?`

#### Strategy A — Raw Vector Search

- Rank 1 | chunk_0011 | score=0.6705
  - Preview: Data consistency during high-throughput ingestion pipelines requires careful coordination. Write-ahead logs ensure durab...
- Rank 2 | chunk_0002 | score=0.4832
  - Preview: Systems like Apache Cassandra and DynamoDB are designed around this tradeoff, using consistent hashing to distribute dat...
- Rank 3 | chunk_0001 | score=0.4379
  - Preview: Modern distributed systems must balance consistency, availability, and partition tolerance — a tradeoff formalized by th...

#### Strategy B — AI-Enhanced Retrieval

**Expanded Query:**

> How is data consistency maintained during distributed ingestion?. Related concepts: replication consistency synchronized ingestion distributed data synchronization consistent state management distributed write coordination parallel data pipelines replication partitioning distributed coordination stream ingestion fault tolerant ingestion

- Rank 1 | chunk_0011 | score=0.6717
  - Preview: Data consistency during high-throughput ingestion pipelines requires careful coordination. Write-ahead logs ensure durab...
- Rank 2 | chunk_0001 | score=0.5192
  - Preview: Modern distributed systems must balance consistency, availability, and partition tolerance — a tradeoff formalized by th...
- Rank 3 | chunk_0013 | score=0.5153
  - Preview: Conflict-free replicated data types allow concurrent updates across replicas without coordination, resolving conflicts d...

#### Analysis

- Strategy B improved top-1 similarity by `0.0012`.
- Query expansion introduced additional technical terminology intended to improve semantic alignment between user intent and indexed corpus language.

---

## Final Observations

The benchmark demonstrates that AI-assisted query expansion can improve semantic retrieval quality by bridging vocabulary gaps between user queries and technical document terminology.

The largest gains were observed for abstract or high-level user queries where the indexed corpus used more implementation-oriented vocabulary.

Smaller improvements were observed for already well-specified technical queries, indicating that raw vector retrieval remains effective when query vocabulary closely matches document terminology.

Cosine similarity was used because embedding direction is more semantically meaningful than vector magnitude for transformer-based language embeddings.

In production, this architecture can be migrated to Vertex AI Matching Engine by replacing the local SentenceTransformer embedder with Vertex AI `textembedding-gecko` models and replacing FAISS with a managed vector index.
