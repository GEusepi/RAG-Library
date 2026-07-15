"""Factory for creating embedding generators."""

from typing import Any

from rag_qdrant.embeddings.base import BaseEmbedder
from rag_qdrant.exceptions import ConfigurationError


def create_embedder(provider: str, **kwargs: Any) -> BaseEmbedder:
    """Create an embedder instance based on the provider name.

    Usage:
        >>> # OpenAI Embedder
        >>> embedder = create_embedder("openai", api_key="sk-...", model="text-embedding-3-small")
        
        >>> # Cohere Embedder
        >>> embedder = create_embedder("cohere", api_key="...", model="embed-english-v3.0")

        >>> # Local Embedder (SentenceTransformers)
        >>> embedder = create_embedder("local", model_name="all-MiniLM-L6-v2", device="cpu")

    Args:
        provider: The provider name ('openai', 'cohere', 'local').
        **kwargs: Configuration arguments passed to the embedder constructor.

    Returns:
        BaseEmbedder: An instance of the requested embedder.

    Raises:
        ConfigurationError: If the provider name is unknown or invalid.
    """
    provider_lower = provider.lower()
    
    if provider_lower == "openai":
        from rag_qdrant.embeddings.openai import OpenAIEmbedder
        return OpenAIEmbedder(**kwargs)
        
    if provider_lower == "cohere":
        from rag_qdrant.embeddings.cohere import CohereEmbedder
        return CohereEmbedder(**kwargs)
        
    if provider_lower == "local":
        from rag_qdrant.embeddings.local import LocalEmbedder
        return LocalEmbedder(**kwargs)
        
    raise ConfigurationError(
        f"Unknown embedding provider: '{provider}'. "
        "Supported providers are: 'openai', 'cohere', 'local'."
    )
