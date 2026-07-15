# rag-qdrant

A modular **document retrieval** library built on Qdrant. Loads documents, splits them into chunks, generates embeddings, and indexes them for vector search.

---

## Architecture

```
Loader → Chunker → Embedder → VectorStore → Retriever
```

| Module | Implementations |
|--------|----------------|
| **Loaders** | `TextLoader`, `PDFLoader`, `WebLoader` |
| **Chunkers** | `FixedSizeChunker`, `RecursiveCharacterChunker`, `SemanticChunker` |
| **Embeddings** | `local` (SentenceTransformers), `openai`, `cohere` |
| **VectorStore** | `qdrant` (in-memory or server) |
| **Retrieval** | `Retriever` (dense), `HybridRetriever` (dense + keyword via RRF), `Reranker` |

---

## Project Structure

```
src/rag_qdrant/
├── loaders/          # Document loading
├── chunking/         # Text splitting
├── embeddings/       # Embedding generation
├── vectorstore/      # Qdrant integration
├── retrieval/        # Search (dense, hybrid, reranker)
├── interfaces/       # Pydantic schemas and configuration
├── pipeline.py       # Main orchestrator
└── exceptions.py     # Custom exceptions
```

---

## Installation

```bash
# Core
pip install .

# With optional dependencies
pip install ".[local]"      # SentenceTransformers
pip install ".[openai]"     # OpenAI embeddings
pip install ".[cohere]"     # Cohere embeddings
pip install ".[pdf]"        # PDF support
pip install ".[web]"        # Web scraping
pip install ".[all]"        # Everything
```

---

## Prerequisites

Start Qdrant with Docker:

```bash
docker compose up -d
```

This exposes the REST API on `localhost:6333`.

---

## Quick Start

### 1. Programmatic Setup

```python
import asyncio
from rag_qdrant import (
    TextLoader, RecursiveCharacterChunker,
    create_embedder, create_vector_store,
    Retriever, RAGPipeline
)

async def main():
    loader = TextLoader("documents.txt")
    chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
    embedder = create_embedder("local", model_name="all-MiniLM-L6-v2")
    store = create_vector_store("qdrant", url="http://localhost", port=6333)
    retriever = Retriever(store=store, embedder=embedder, collection_name="docs")

    pipeline = RAGPipeline(
        store=store, embedder=embedder,
        chunker=chunker, retriever=retriever
    )

    await pipeline.async_create_collection()
    await pipeline.async_ingest(loader)

    results = await pipeline.async_retrieve("What is the main topic?")
    for r in results:
        print(f"[{r.score:.4f}] {r.chunk.content}")

asyncio.run(main())
```

### 2. JSON Configuration Setup

Create a `pipeline_config.json` file:

```json
{
  "vectorstore": {
    "provider": "qdrant",
    "url": "http://localhost",
    "port": 6333,
    "api_key": "optional"
  },
  "embeddings": {
    "provider": "local",
    "model": "all-MiniLM-L6-v2"
  },
  "chunker": {
    "type": "recursive",
    "chunk_size": 500,
    "chunk_overlap": 50
  },
  "retriever": {
    "collection_name": "docs",
    "limit": 5,
    "hybrid": false
  }
}
```

Load and use:

```python
import json
from rag_qdrant import RAGConfig, RAGPipeline

with open("pipeline_config.json") as f:
    config = RAGConfig.model_validate(json.load(f))

pipeline = RAGPipeline.from_config(config)
```

---

## VectorStore Configuration

| Field | Description | Default |
|-------|-------------|---------|
| `provider` | Vector store type | `"qdrant"` |
| `url` | Qdrant server URL | `None` |
| `host` | Alternative hostname | `None` |
| `port` | REST API port | `6333` |
| `api_key` | API key (Qdrant Cloud) | `None` |
| `location` | `":memory:"` for in-memory | `None` |

## Embeddings Configuration

| Field | Description |
|-------|-------------|
| `provider` | `"local"`, `"openai"`, `"cohere"` |
| `model` | Model name |
| `api_key` | API key (for openai/cohere) |
| `device` | Device for local (`"cpu"`, `"cuda"`) |

## Retriever Configuration

| Field | Description | Default |
|-------|-------------|---------|
| `collection_name` | Qdrant collection name | *required* |
| `limit` | Max number of results | `5` |
| `hybrid` | Enable hybrid search (dense + keyword via RRF) | `false` |
| `reranker_provider` | Reranker provider (`"local"`, `"cohere"`) | `None` |
| `reranker_model` | Cross-encoder model | `None` |

---

## Pipeline API

| Method | Description |
|--------|-------------|
| `RAGPipeline.from_config(config)` | Create pipeline from `RAGConfig` |
| `create_collection()` / `async_create_collection()` | Create vector collection |
| `delete_collection()` / `async_delete_collection()` | Delete collection |
| `ingest(loader)` / `async_ingest(loader)` | Load, chunk, embed and index |
| `retrieve(query)` / `async_retrieve(query)` | Search for relevant chunks |

---

## Quality Checks

```bash
mypy src          # Type checking
ruff check src    # Linting
pytest tests/     # Tests
```
