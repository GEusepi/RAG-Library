"""Configuration schemas for the RAG pipeline."""

from typing import Any

from pydantic import BaseModel, Field


class EmbedderConfig(BaseModel):
    """Configuration for text embedding models."""
    provider: str  # 'openai', 'cohere', 'local'
    model: str | None = None
    api_key: str | None = None
    device: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(BaseModel):
    """Configuration for the vector database client."""
    provider: str = "qdrant"
    location: str | None = None
    url: str | None = None
    host: str | None = None
    port: int = 6333
    api_key: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ChunkerConfig(BaseModel):
    """Configuration for chunking text."""
    type: str = "recursive"  # 'fixed', 'recursive', 'semantic'
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] | None = None
    threshold_percentile: float = 95.0
    min_chunk_size: int = 100


class RetrieverConfig(BaseModel):
    """Configuration for search and retrieval logic."""
    collection_name: str
    limit: int = 5
    hybrid: bool = False
    rrf_k: int = 60
    reranker_provider: str | None = None  # 'local', 'cohere'
    reranker_model: str | None = None
    reranker_api_key: str | None = None
    reranker_kwargs: dict[str, Any] = Field(default_factory=dict)


class RAGConfig(BaseModel):
    """Full configurations for a RAGPipeline instance."""
    vectorstore: VectorStoreConfig
    embeddings: EmbedderConfig
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    retriever: RetrieverConfig
