"""Custom exceptions for the rag-qdrant library."""

class RagQdrantError(Exception):
    """Base exception for all errors in the rag-qdrant library."""
    pass


class VectorStoreError(RagQdrantError):
    """Exception raised for vector store errors."""
    pass


class CollectionNotFoundError(VectorStoreError):
    """Exception raised when a requested collection does not exist."""
    pass


class EmbeddingError(RagQdrantError):
    """Exception raised for embedding generation errors."""
    pass


class ConfigurationError(RagQdrantError):
    """Exception raised for invalid configurations."""
    pass


class ChunkingError(RagQdrantError):
    """Exception raised during document chunking."""
    pass


class LoaderError(RagQdrantError):
    """Exception raised when document loading fails."""
    pass
