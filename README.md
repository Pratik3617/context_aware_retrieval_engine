# Semantic RAG & Vector Search Pipeline

RAG pipeline comparing **raw vector retrieval (Strategy A)** vs **AI-enhanced query-expanded retrieval (Strategy B)** using local embeddings, FAISS vector search, and mocked Vertex AI SDK surfaces.

---

## Project Objective

This project implements a local Retrieval-Augmented Generation (RAG)
pipeline for semantic document retrieval.

The system benchmarks two retrieval strategies:

- **Strategy A — Raw Vector Search**
  Direct embedding-based semantic retrieval.

- **Strategy B — AI-Enhanced Retrieval**
  Query expansion using a mocked Vertex AI `GenerativeModel`
  before vector search.

The objective is to evaluate whether semantic query expansion
improves retrieval quality and embedding alignment.

---

## Retrieval Architecture

```text
Raw Document
    ↓
DocumentChunker
    ↓
SentenceTransformerEmbedder
    ↓
FAISS Vector Store
    ↓
 ┌──────────────────────────────┐
 │ Strategy A                   │
 │ Query → Embed → Search       │
 └──────────────────────────────┘

 ┌──────────────────────────────┐
 │ Strategy B                   │
 │ Query → Query Expansion      │
 │       → Embed → Search       │
 └──────────────────────────────┘
```

---

## Structure

```text
context_aware_retrieval_engine/
├── data/
│   └── technical_document.txt
│
├── src/
│   ├── config.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── stub_embedder.py
│   ├── vector_store.py
│   ├── query_expander.py
│   └── pipeline.py
│
├── tests/
│
├── benchmark.py
├── retrieval_benchmark.md
├── benchmark_results.json
├── download_models.py
├── requirements.txt
└── README.md
```

---

## Component Overview

| Component | Description |
|---|---|
| `DocumentChunker` | Splits `.txt` / `.pdf` documents into overlapping semantic chunks |
| `SentenceTransformerEmbedder` | Local embedding model using `all-MiniLM-L6-v2` |
| `MockVertexAIEmbedder` | Mocked `TextEmbeddingModel` SDK surface |
| `FAISSVectorStore` | In-memory cosine similarity vector index |
| `MockGenerativeModel` | Query expansion using mocked Vertex AI generative interface |
| `RAGPipeline` | End-to-end ingestion and retrieval orchestration |
| `benchmark.py` | Runs Strategy A vs Strategy B evaluation |

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional: pre-download embedding models for offline execution.

```bash
python download_models.py
```

---

## Run Tests

```bash
pytest tests/ -v
```

84 tests with fully offline unit coverage.

Optional integration tests validate real SentenceTransformer embeddings:

```bash
INTEGRATION=1 pytest tests/ -v
```

---

## Run Benchmark

```bash
python benchmark.py
```

Outputs:
- `benchmark_results.json`
- `retrieval_benchmark.md`

---

## Benchmark Summary

| Query | Strategy A | Strategy B | Relative Gain |
|---|---|---|---|
| Peak load handling | 0.6139 | 0.7870 | +28.2% |
| Vector search latency | 0.5646 | 0.6952 | +23.1% |
| Distributed ingestion consistency | 0.6705 | 0.6717 | +0.18% |

Average improvement across benchmark queries: **+17.16%**

---

## Query Expansion Examples

### Example 1

```text
Input Query:
How does the system handle peak load?

Expanded Query:
autoscaling, load balancing,
traffic spikes, horizontal scaling,
throughput optimization
```

### Example 2

```text
Input Query:
What are the strategies for optimizing vector search latency?

Expanded Query:
ANN indexing, FAISS optimization,
embedding retrieval, low latency similarity search
```

---

## Key Design Decisions

### Cosine Similarity over Euclidean Distance

Transformer embeddings primarily encode semantic direction rather than absolute vector magnitude.

Vectors are L2-normalized before insertion into FAISS `IndexFlatIP`, making inner-product mathematically equivalent to cosine similarity.

Why cosine similarity:
- stable semantic ranking
- magnitude-independent comparison
- standard practice for transformer embeddings
- efficient retrieval with normalized vectors

---

### Semantic Query Expansion

`MockGenerativeModel` simulates a Vertex AI `GenerativeModel`
by expanding user queries with semantically related technical terminology.

Examples:
- `"peak load"`  
  → autoscaling, load balancing, traffic spikes

- `"vector search latency"`  
  → ANN indexing, FAISS optimization, embedding retrieval

This improves semantic alignment between user intent
and indexed corpus vocabulary.

---

### Dynamic Runtime Chunking

`DocumentChunker` reads raw `.txt` / `.pdf` documents at runtime.

Chunking occurs during ingestion rather than being manually pre-generated, ensuring:
- realistic retrieval evaluation
- unbiased semantic chunk boundaries
- production-like ingestion behavior

Supported strategies:
- `RECURSIVE_SPLIT`
- `SLIDING_WINDOW`

---

### Mocked Vertex AI SDK Surfaces

The mocked Vertex AI SDK interfaces preserve production-style architecture while remaining fully offline and self-contained.

Mocked services:
- `TextEmbeddingModel`
- `GenerativeModel`

This enables:
- deterministic testing
- offline execution
- cloud-provider abstraction
- production migration readiness

---


## Key Retrieval Insight

Semantic query expansion was most effective for
high-level or abstract user queries where the
document corpus used different implementation-oriented terminology.

The benchmark also demonstrates that aggressive query
expansion can occasionally introduce semantic drift,
highlighting the importance of balanced expansion
strategies in production RAG systems.

---

## Production Migration (Vertex AI)

This local implementation mirrors a production-grade Vertex AI retrieval architecture.

| Local Component | Production Equivalent |
|---|---|
| `SentenceTransformerEmbedder` | Vertex AI `textembedding-gecko` |
| `FAISSVectorStore` | Vertex AI Matching Engine |
| `MockGenerativeModel` | Gemini / Vertex AI `GenerativeModel` |

Potential production enhancements:
- distributed vector indexing
- metadata filtering
- hybrid BM25 + dense retrieval
- scalable ANN retrieval
- online embedding refresh
- multi-tenant vector serving

---

## Benchmark Outputs

### `benchmark_results.json`

Structured machine-readable retrieval comparison.

Contains:
- benchmark metadata
- Strategy A retrieval results
- Strategy B retrieval results
- similarity scores
- expanded queries

---

### `retrieval_benchmark.md`

Human-readable benchmark report including:
- retrieval rankings
- top-k chunk previews
- similarity comparisons
- semantic analysis
- benchmark observations

---

## Technologies Used

| Technology | Purpose |
|---|---|
| `sentence-transformers` | Embedding generation |
| `FAISS` | Vector similarity search |
| `NumPy` | Vector operations |
| `pytest` | Testing |
| `unittest.mock` | SDK mocking |
| `pypdf` | PDF ingestion |
| `python-dotenv` | Environment configuration |

---

## Future Improvements

Potential extensions:
- hybrid sparse+dense retrieval
- reranking models
- semantic chunking
- metadata-aware filtering
- cross-encoder retrieval reranking
- distributed FAISS indexing
- streaming ingestion pipeline
- evaluation metrics (MRR / Recall@K)

---

## Final Notes

This project focuses on:
- semantic retrieval engineering
- query understanding
- vector database workflows
- retrieval benchmarking
- production-oriented RAG architecture

The system is intentionally designed to remain:
- fully local
- reproducible
- deterministic
- cloud-migration friendly
- testable offline