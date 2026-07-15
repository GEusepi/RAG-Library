"""Factory for creating vector store clients."""

from typing import Any

from rag_qdrant.exceptions import ConfigurationError
from rag_qdrant.vectorstore.base import BaseVectorStore


def create_vector_store(provider: str, **kwargs: Any) -> BaseVectorStore:
    """Create a vector store instance based on the provider name.

    Usage:
        >>> # In-memory Qdrant Client (ideal for tests / experiments)
        >>> store = create_vector_store("qdrant", location=":memory:")
        
        >>> # Local Server
        >>> store = create_vector_store("qdrant", host="localhost", port=6333)

        >>> # Cloud / Hosted Instance
        >>> store = create_vector_store("qdrant", url="https://...", api_key="...")

    Args:
        provider: The provider name ('qdrant').
        **kwargs: Configuration arguments passed to the vector store constructor.

    Returns:
        BaseVectorStore: An instance of the requested vector store.

    Raises:
        ConfigurationError: If the provider name is unknown or invalid.
    """
    provider_lower = provider.lower()
    
    if provider_lower == "qdrant":
        from rag_qdrant.vectorstore.qdrant import QdrantVectorStore
        return QdrantVectorStore(**kwargs)
    raise ConfigurationError(
        f"Unknown vector store provider: '{provider}'. "
        "Supported providers are: 'qdrant'."
    )
