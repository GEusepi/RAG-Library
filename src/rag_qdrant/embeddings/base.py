"""Base interface for embedding generators."""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for all text embedding providers."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of document texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a query text.

        Args:
            text: A single string query to embed.

        Returns:
            List[float]: The generated embedding.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        pass

    @abstractmethod
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of document texts asynchronously.

        Args:
            texts: List of strings to embed.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        pass

    @abstractmethod
    async def aembed_query(self, text: str) -> list[float]:
        """Generate an embedding for a query text asynchronously.

        Args:
            text: A single string query to embed.

        Returns:
            List[float]: The generated embedding.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        pass
