"""rag-qdrant package root."""

from rag_qdrant.chunking.base import BaseChunker
from rag_qdrant.chunking.factory import create_chunker
from rag_qdrant.chunking.fixed_size import FixedSizeChunker
from rag_qdrant.chunking.recursive import RecursiveCharacterChunker
from rag_qdrant.chunking.semantic import SemanticChunker
from rag_qdrant.embeddings.base import BaseEmbedder
from rag_qdrant.embeddings.factory import create_embedder
from rag_qdrant.exceptions import (
    ChunkingError,
    CollectionNotFoundError,
    ConfigurationError,
    EmbeddingError,
    LoaderError,
    RagQdrantError,
    VectorStoreError,
)
from rag_qdrant.interfaces.config import (
    ChunkerConfig,
    EmbedderConfig,
    RAGConfig,
    RetrieverConfig,
    VectorStoreConfig,
)
from rag_qdrant.interfaces.schemas import Chunk, Document, Filter, SearchResult
from rag_qdrant.loaders.base import BaseLoader
from rag_qdrant.loaders.pdf import PDFLoader
from rag_qdrant.loaders.text import TextLoader
from rag_qdrant.loaders.web import WebLoader
from rag_qdrant.pipeline import RAGPipeline
from rag_qdrant.retrieval.hybrid import HybridRetriever
from rag_qdrant.retrieval.reranker import Reranker
from rag_qdrant.retrieval.retriever import Retriever
from rag_qdrant.vectorstore.base import BaseVectorStore
from rag_qdrant.vectorstore.factory import create_vector_store

__all__ = [
    # Exceptions
    "RagQdrantError",
    "VectorStoreError",
    "CollectionNotFoundError",
    "EmbeddingError",
    "ConfigurationError",
    "ChunkingError",
    "LoaderError",
    # Schema
    "Document",
    "Chunk",
    "Filter",
    "SearchResult",
    # Loaders
    "BaseLoader",
    "TextLoader",
    "PDFLoader",
    "WebLoader",
    # Chunking
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveCharacterChunker",
    "SemanticChunker",
    "create_chunker",
    # Embeddings
    "BaseEmbedder",
    "create_embedder",
    # Vector store
    "BaseVectorStore",
    "create_vector_store",
    # Retrieval
    "Retriever",
    "HybridRetriever",
    "Reranker",
    # Configs
    "RAGConfig",
    "EmbedderConfig",
    "VectorStoreConfig",
    "ChunkerConfig",
    "RetrieverConfig",
    # Pipeline
    "RAGPipeline",
]
